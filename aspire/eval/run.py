"""Run one resumable arm (generation or refinement) on a frozen suite.

Order per sample: initial candidate (generated, or read from --initial) ->
refinement against the VisibleOracle -> final strict evaluation on F0 and the
held-out contexts. Hidden contexts are evaluated only after every decision is fixed.
"""
import argparse
from contextlib import contextmanager
import importlib.metadata
import json
import os
from pathlib import Path

from aspire.common import append_jsonl, code_hashes, digest, file_hash, pin_json, read_jsonl, write_json
from aspire.data.suite import load_suite
from aspire.models import prompts as P
from aspire.models.backend import Backend, model_identity
from aspire.refine.loop import METHODS, REFINE_METHODS, SOURCES, refine
from aspire.verify.extraction import parse_answer
from aspire.verify.oracle import Solver, VisibleOracle, evaluate


@contextmanager
def run_lock(directory):
    path = Path(directory) / "RUN.lock"
    with open(path, "a+b") as stream:
        stream.seek(0)
        if stream.read(1) == b"":
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"another process is writing {directory}") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def versions():
    result = {}
    for name in ("clingo", "pandas", "requests", "torch", "transformers", "peft", "bitsandbytes"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", required=True)
    p.add_argument("--split", choices=("iid", "ood"), default="iid")
    p.add_argument("--families", choices=("paper", "all"), default="paper")
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--initial", help="completed base run whose candidates are refined")
    p.add_argument("--backend", choices=("local", "api", "mock"), default="local")
    p.add_argument("--model", required=True, help="local weights directory or API model name")
    p.add_argument("--adapter", default="", help="LoRA adapter of the fine-tuned generator")
    p.add_argument("--max-gpu-gb", type=float, default=0, help="per-process GPU cap in GB; 0 = no cap")
    p.add_argument("--generator-style", choices=("qa", "raw", "chat"), default="qa")
    p.add_argument("--feedback-style", choices=("qa", "raw", "chat"), default="raw")
    p.add_argument("--feedback-base", action=argparse.BooleanOptionalAction, default=True,
                   help="disable the LoRA adapter for feedback-LLM calls (Sec. 3.2)")
    p.add_argument("--feedback-model", default="", help="API model name for the feedback role")
    p.add_argument("--feedback-backend", choices=("same", "api"), default="same",
                   help="api: send feedback-role calls to --endpoint (e.g. an Instruct feedback model)")
    p.add_argument("--baseline-role", choices=("generator", "feedback"), default="generator",
                   help="who explains/fixes in self_debug and pgs and acts in llm_asp")
    p.add_argument("--system-prompt", default="")
    p.add_argument("--endpoint", default=os.environ.get("ASPIRE_API_URL", ""))
    p.add_argument("--api-key-env", default="ASPIRE_API_KEY")
    p.add_argument("--api-timeout", type=float, default=300)
    p.add_argument("--send-api-seed", action="store_true")
    p.add_argument("--fewshot-k", type=int, default=3)
    p.add_argument("--seed", type=int, default=56)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--attempts", type=int, default=3, help="refinement budget T")
    p.add_argument("--selection", choices=("shortest", "random", "first"), default="shortest")
    p.add_argument("--visible", choices=("all", "instance"), default="all")
    p.add_argument("--keep", choices=("best", "last"), default="best")
    p.add_argument("--verify-projection", choices=("on", "off"), default="on",
                   help="off: refinement compares whole answer sets (No Projection ablation)")
    p.add_argument("--verify-phases", choices=("both", "optimal"), default="both",
                   help="optimal: refinement skips the feasible phase (Optimal Only ablation)")
    p.add_argument("--grammar", choices=("none", "asp"), default="none",
                   help="asp: grammar-constrained decoding of generator calls (local backend)")
    p.add_argument("--clingo-timeout", type=float, default=20)
    p.add_argument("--limit", type=int, default=0, help="first N samples only (smoke tests)")
    a = p.parse_args(argv)
    if a.attempts < 0 or a.max_tokens < 1 or a.clingo_timeout <= 0:
        p.error("invalid budget/tokens/timeout")
    if (a.backend == "api" or a.feedback_backend == "api") and not a.endpoint:
        p.error("API runs require --endpoint (an OpenAI-compatible /chat/completions URL)")
    if a.backend != "local" and a.adapter:
        p.error("--adapter requires --backend local")
    if a.grammar != "none" and a.backend != "local":
        p.error("--grammar requires --backend local")
    if a.method in REFINE_METHODS and not a.initial:
        p.error("refinement methods require --initial from a completed base run")
    if a.method in ("base", "fewshot") and a.initial:
        p.error("generation methods do not take --initial")
    return a


def fewshot_bank(k, seed, forbidden_questions):
    """k training-vocabulary exemplars per family; any exemplar equal to a test question is dropped."""
    from aspire.data.generate import TRACK_A_SPECS, TRAIN_VOCAB, test_rows
    bank = {}
    for row in test_rows(TRACK_A_SPECS, 4 * k, seed, TRAIN_VOCAB, "fewshot", "iid"):
        if row["prompt"] in forbidden_questions or len(bank.get(row["task"], [])) >= k:
            continue
        bank.setdefault(row["task"], []).append((row["prompt"] + "\nExample input facts (do not copy into your program):\n"
                                                 + row["fact"], row["answer"]))
    return bank


def run(a):
    manifest, rows, private = load_suite(a.suite, a.split, "test", a.families)
    if a.limit:
        rows = rows[:a.limit]
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    identity = model_identity(a)
    initial = {}
    initial_prov = None
    if a.initial:
        d = Path(a.initial)
        initial_prov = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
        done = json.loads((d / "COMPLETE.json").read_text(encoding="utf-8"))
        if digest(initial_prov) != done["config_sha256"] or done["results_sha256"] != file_hash(d / "protocol_samples.jsonl"):
            raise ValueError("initial run changed after completion")
        for key in ("suite_id", "split"):
            if initial_prov[key] != {"suite_id": manifest["suite_id"], "split": a.split}[key]:
                raise ValueError(f"initial run mismatch in {key}")
        initial = {r["sample_id"]: r for r in read_jsonl(d / "protocol_samples.jsonl")}
        missing = [r["sample_id"] for r in rows if r["sample_id"] not in initial]
        if missing:
            raise ValueError(f"initial run lacks {len(missing)} samples of this selection")
    config = {"schema_version": 1, "suite_id": manifest["suite_id"],
              "split": a.split, "families": a.families, "method": a.method,
              "source": SOURCES.get(a.method, "generation"), "model": identity,
              "baseline_role": a.baseline_role, "seed": a.seed, "max_tokens": a.max_tokens,
              "attempts": a.attempts, "selection": a.selection, "visible": a.visible, "keep": a.keep,
              "verify_projection": a.verify_projection, "verify_phases": a.verify_phases, "grammar": a.grammar,
              "fewshot_k": a.fewshot_k if a.method == "fewshot" else None,
              "clingo_timeout": a.clingo_timeout, "limit": a.limit,
              "initial_run_sha256": digest(initial_prov) if initial_prov else None,
              "code": code_hashes(), "versions": versions(), "samples": [r["sample_id"] for r in rows]}
    bank = fewshot_bank(a.fewshot_k, a.seed + 7, {r["question"] for r in rows}) if a.method == "fewshot" else None
    with run_lock(out):
        pin_json(out / "run_config.json", config)
        done = {r["sample_id"]: r for r in read_jsonl(out / "protocol_samples.jsonl", recover_tail=True)}
        backend, solver = Backend(a, out), Solver(a.clingo_timeout)
        try:
            for pos, public in enumerate(rows, 1):
                sid = public["sample_id"]
                if sid in done:
                    continue
                solver.cache.clear()
                before = solver.measure()
                meta = {"sample_id": sid, "method": a.method, "round": 0, "phase": "initial"}
                spec = P.specification(public)
                if a.method == "base":
                    raw, _ = backend.generate(spec, "generator", meta)
                    start = parse_answer(raw)[0]
                elif a.method == "fewshot":
                    raw, _ = backend.generate(P.fewshot_prompt(bank[public["task"]], spec), "generator", meta)
                    start = parse_answer(raw)[0]
                else:
                    raw, start = initial[sid].get("initial_raw_generation", ""), initial[sid]["candidate_program"]
                # Schema-2 suites verify against frozen expected outputs only; the
                # reference program is passed solely for legacy (schema-1) suites.
                gold = None if manifest.get("schema_version", 1) >= 2 else private[sid]["gold_program"]
                # The no-projection ablation needs full reference answer sets, which are not frozen.
                oracle_gold = private[sid]["gold_program"] if a.verify_projection == "off" else gold
                oracle = VisibleOracle(solver, public, oracle_gold, a.visible,
                                       project=a.verify_projection == "on", phases_mode=a.verify_phases)
                if a.method in REFINE_METHODS:
                    candidate, trace, trajectory = refine(a.method, public, start, backend, oracle, a)
                else:
                    candidate, trace, trajectory = start, [], [start]
                # Reported visible score always uses the standard verifier, also in verifier ablations.
                scorer = oracle if oracle.project and oracle.phase_list is None else VisibleOracle(solver, public, gold, a.visible)
                visible_final = scorer.score(candidate)
                after_inference = solver.measure()
                # Every inference decision is final; only now are held-out contexts used.
                private_eval = private[sid] if gold else {k: v for k, v in private[sid].items() if k != "gold_program"}
                result = evaluate(solver, public, private_eval, candidate)
                initial_result = evaluate(solver, public, private_eval, start)
                last_result = evaluate(solver, public, private_eval, trajectory[-1])
                after = solver.measure()
                result.update({
                    "sample_id": sid, "sample_index": public["sample_index"], "task": public["task"],
                    "level": public["level"], "candidate_program": candidate,
                    "visible_score": visible_final, "visible_total": len(scorer.contexts),
                    "initial_candidate_program": start, "initial_raw_generation": raw,
                    "initial_result": _brief(initial_result), "last_candidate_program": trajectory[-1],
                    "last_result": _brief(last_result), "trajectory": trajectory, "refinement_trace": trace,
                    "inference_solver": {k: after_inference[k] - before[k] for k in before},
                    "evaluation_solver": {k: after[k] - after_inference[k] for k in before}})
                append_jsonl(out / "protocol_samples.jsonl", result)
                done[sid] = result
                write_json(out / "usage.json", backend.summary())
                print(f"[{a.method} {a.split}] {pos}/{len(rows)} {public['task']} "
                      f"strict={int(result['end_to_end_correct'])} rounds={len(trace)}", flush=True)
            write_json(out / "usage.json", backend.summary())
            write_json(out / "COMPLETE.json", {"samples": len(done), "config_sha256": digest(config),
                                                "results_sha256": file_hash(out / "protocol_samples.jsonl")})
            print(f"COMPLETE {out}: {sum(r['end_to_end_correct'] for r in done.values())}/{len(done)} strict")
        finally:
            solver.close()


def _brief(result):
    keep = ("instance_correct", "context_correct", "end_to_end_correct", "extraction", "parse_ground",
            "timed_out")
    return {k: result[k] for k in keep} | {"contexts_correct": [c["correct"] for c in result["contexts"]]}


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
