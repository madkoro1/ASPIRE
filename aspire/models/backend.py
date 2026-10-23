"""Local HF/LoRA, OpenAI-compatible API and offline mock generation backends.

Two roles share one backend:
  generator  - the fine-tuned model (LoRA adapter active);
  feedback   - the feedback LLM. Locally it is the SAME Qwen2.5-32B Base weights
               with the adapter disabled (Sec. 3.2); over the API it may be a
               different model (--feedback-model).
Every call is cached in generation_cache.jsonl, keyed by prompt, role,
metadata and decoding settings, so interrupted runs resume without re-sampling.
"""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time

from aspire.common import append_jsonl, digest, file_hash, read_jsonl

ROLES = ("generator", "feedback")


def render(text, role, style, tokenizer=None, system=None):
    """Return (prompt_string, add_special_tokens) for local models.

    'qa' reproduces the fine-tuning frame `Question: {q}\\nAnswer: {a}`. The
    prompt ends at 'Answer:'; the following space belongs to the first answer
    token under the BPE tokenizer, exactly as during training.
    'raw' is plain text completion (Base feedback model with few-shot prompt).
    'chat' applies the tokenizer chat template (Instruct models).
    """
    if style == "qa":
        return f"Question: {text}\nAnswer:", False
    if style == "raw":
        return text, False
    if style == "chat":
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), False
    raise ValueError(f"unknown prompt style: {style}")


class Backend:
    def __init__(self, args, output):
        self.args = args
        self.output = Path(output)
        self.path = self.output / "generation_cache.jsonl"
        records = read_jsonl(self.path, recover_tail=True)
        self.cache = {r["request_hash"]: r for r in records}
        if len(self.cache) != len(records):
            raise ValueError("duplicate generation cache keys")
        self.model = self.tokenizer = None
        self.session = None
        self.grammar_factory = None

    # ------------------------------------------------------------------ local
    def _load_local(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        a = self.args
        if not torch.cuda.is_available():
            raise RuntimeError("local generation requires a CUDA GPU; use --backend api or mock otherwise")
        if a.max_gpu_gb > 0:
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            torch.cuda.set_per_process_memory_fraction(min(1.0, a.max_gpu_gb / total), 0)
        if a.adapter:
            cfg = json.loads((Path(a.adapter) / "adapter_config.json").read_text(encoding="utf-8"))
            trained = Path(cfg["base_model_name_or_path"]).name.lower()
            if trained != Path(a.model).name.lower():
                raise ValueError(f"adapter trained on {trained}; supplied backbone is {a.model}")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs = {"device_map": {"": 0}, "torch_dtype": dtype}  # unquantized, same precision as training
        self.tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True, **kwargs)
        if a.adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, a.adapter, local_files_only=True)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _local(self, text, role, stop, seed):
        import torch
        a = self.args
        self._load_local()
        style = a.generator_style if role == "generator" else a.feedback_style
        prompt, special = render(text, role, style, self.tokenizer, a.system_prompt if style == "chat" else None)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=special).to(self.model.device)
        count = inputs["input_ids"].shape[1]
        limit = getattr(self.model.config, "max_position_embeddings", 32768)
        if count + a.max_tokens > limit:
            raise ValueError(f"prompt exceeds model context: {count}+{a.max_tokens}>{limit}")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        kwargs = {"max_new_tokens": a.max_tokens, "do_sample": False,
                  "pad_token_id": self.tokenizer.pad_token_id}
        if stop:
            kwargs.update(stop_strings=list(stop), tokenizer=self.tokenizer)
        if role == "generator" and getattr(a, "grammar", "none") == "asp":
            from transformers import LogitsProcessorList
            if self.grammar_factory is None:
                from aspire.models.grammar import processor_factory
                self.grammar_factory = processor_factory(self.tokenizer)
            kwargs["logits_processor"] = LogitsProcessorList([self.grammar_factory()])
        disable = role == "feedback" and a.adapter and a.feedback_base
        ctx = self.model.disable_adapter() if disable else nullcontext()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with ctx, torch.inference_mode():
            generated = self.model.generate(**inputs, **kwargs)
        torch.cuda.synchronize()
        new = generated[0][count:]
        content = self.tokenizer.decode(new, skip_special_tokens=True)
        return content, {"rendered_prompt": prompt, "prompt_tokens": count,
                         "completion_tokens": int(new.shape[0]),
                         "elapsed_seconds": time.perf_counter() - started,
                         "finish_reason": "length" if new.shape[0] >= a.max_tokens else "stop",
                         "adapter_disabled": bool(disable)}

    # -------------------------------------------------------------------- api
    def _api(self, text, role, stop, seed):
        import requests
        a = self.args
        if self.session is None:
            self.session = requests.Session()
        key = os.environ.get(a.api_key_env, "").strip()
        if not key:
            raise RuntimeError(f"missing API key: export {a.api_key_env}=...")
        model = (a.feedback_model or a.model) if role == "feedback" else a.model
        if not a.endpoint:
            raise RuntimeError("API calls require --endpoint")
        messages = ([{"role": "system", "content": a.system_prompt}] if a.system_prompt else []) + \
            [{"role": "user", "content": text}]
        body = {"model": model, "messages": messages, "temperature": 0.0, "top_p": 1.0,
                "max_tokens": a.max_tokens}
        if stop:
            body["stop"] = list(stop)[:4]
        if a.send_api_seed:
            body["seed"] = seed
        started, last = time.perf_counter(), None
        for attempt in range(4):
            try:
                r = self.session.post(a.endpoint, json=body, timeout=(15, a.api_timeout),
                                      headers={"Authorization": f"Bearer {key}"})
                if r.status_code == 429 or r.status_code >= 500:
                    last = RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
                    time.sleep(2 * 2 ** attempt)
                    continue
                r.raise_for_status()
                payload = r.json()
                usage = payload.get("usage") or {}
                choice = payload["choices"][0]
                return choice["message"]["content"] or "", {
                    "request": body, "response_model": payload.get("model"),
                    "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                    "completion_tokens": int(usage.get("completion_tokens") or 0),
                    "elapsed_seconds": time.perf_counter() - started,
                    "finish_reason": choice.get("finish_reason")}
            except (requests.RequestException, KeyError, ValueError) as exc:
                last = exc
                time.sleep(2 * 2 ** attempt)
        raise RuntimeError(f"API call failed: {last}")

    # ------------------------------------------------------------------- mock
    def _mock(self, text, role, stop, seed):
        """Deterministic offline stand-in used for plumbing smoke tests only."""
        content = ("<feedback>\nCategory: missing condition\nRule: the output rule\n"
                   "Violated condition: the condition stated in the problem\nPhase: not applicable\n"
                   "Fix direction: add the condition stated in the problem.\n</feedback>" if role == "feedback"
                   else " out(X) :- in(X).")
        return content, {"prompt_tokens": len(text.split()), "completion_tokens": len(content.split()),
                         "elapsed_seconds": 0.0, "finish_reason": "stop"}

    def generate(self, text, role, metadata, stop=()):
        if role not in ROLES:
            raise ValueError(role)
        a = self.args
        request = {"text": text, "role": role, "metadata": metadata, "stop": list(stop),
                   "seed": a.seed, "max_tokens": a.max_tokens,
                   "style": a.generator_style if role == "generator" else a.feedback_style}
        if role == "generator" and getattr(a, "grammar", "none") != "none":  # absent key keeps old caches valid
            request["grammar"] = a.grammar
        key = digest(request)
        if key in self.cache:
            return self.cache[key]["content"], key
        seed = (a.seed + int(key[:8], 16)) % (2**31)
        backend = a.backend
        if role == "feedback" and getattr(a, "feedback_backend", "same") == "api":
            backend = "api"
        content, info = {"local": self._local, "api": self._api, "mock": self._mock}[backend](text, role, stop, seed)
        for s in stop:  # identical truncation for every backend
            if s in content:
                content = content[:content.index(s) + len(s)]
        record = {"request_hash": key, "role": role, "content": content, "prompt": text,
                  "call_metadata": metadata, "effective_seed": seed, **info}
        append_jsonl(self.path, record)
        self.cache[key] = record
        return content, key

    def summary(self):
        result = {}
        for role in ROLES:
            records = [r for r in self.cache.values() if r["role"] == role]
            result[role] = {"calls": len(records),
                            "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in records),
                            "completion_tokens": sum(r.get("completion_tokens", 0) for r in records),
                            "seconds": sum(r.get("elapsed_seconds", 0) for r in records),
                            "length_limited_calls": sum(r.get("finish_reason") == "length" for r in records)}
        if self.model is not None:
            import torch
            result["gpu_peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3
        return result


def model_identity(args):
    identity = {"backend": args.backend, "model": args.model, "adapter": args.adapter,
                "generator_style": args.generator_style, "feedback_style": args.feedback_style,
                "dtype": "bfloat16", "feedback_base": args.feedback_base,
                "feedback_model": args.feedback_model, "feedback_backend": args.feedback_backend,
                "endpoint": args.endpoint if "api" in (args.backend, args.feedback_backend) else None}
    if args.backend == "local":
        paths = [Path(args.model) / "config.json", Path(args.model) / "tokenizer_config.json"]
        if args.adapter:
            paths += sorted(Path(args.adapter).glob("adapter*"))
        identity["local_files"] = {str(p): file_hash(p) for p in paths if p.is_file()}
    return identity
