"""BF16 LoRA fine-tuning (unquantized base weights) for plain SFT, Random-FIM and AST-FIM (Sec. 3.2, Sec. 4.2).

Loss (Eq. 2): causal LM cross-entropy on answer tokens and EOS only, summed over
every micro-batch in one optimizer update (gradient accumulation included) and
divided by the number of supervised tokens in that update. Trainer computes the
token count over the accumulation window (`num_items_in_batch`); `token_sum_loss`
uses it explicitly, so short FIM answers are not up-weighted relative to full
programs as a per-micro-batch mean would do.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time

from aspire.training.common import file_hash, write_json


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, help='local original Qwen2.5-32B Base weights')
    p.add_argument('--train', required=True, help='frozen plain.jsonl / random_fim.jsonl / ast_fim.jsonl')
    p.add_argument('--validation', required=True, help='validation.jsonl from the same frozen directory')
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=56)
    p.add_argument('--seq-length', type=int, default=1024)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--gradient-accumulation', type=int, default=16)
    p.add_argument('--learning-rate', type=float, default=2e-5)
    p.add_argument('--scheduler', choices=['cosine', 'constant'], default='cosine')
    p.add_argument('--epochs', type=float, default=1)
    p.add_argument('--max-steps', type=int, default=-1)
    p.add_argument('--rank', type=int, default=32)
    p.add_argument('--alpha', type=int, default=64)
    p.add_argument('--dropout', type=float, default=.05)
    p.add_argument('--max-gpu-gb', type=float, default=0, help='per-process GPU cap in GB; 0 = no cap')
    p.add_argument('--eval-steps', type=int, default=100)
    p.add_argument('--resume', help='checkpoint directory inside --output, same configuration only')
    return p.parse_args(argv)


def token_sum_loss(outputs, labels, num_items_in_batch=None):
    """Sum of answer-token NLL / supervised tokens in the whole optimizer update."""
    import torch.nn.functional as F
    logits = outputs['logits'] if isinstance(outputs, dict) else outputs[0]
    logits = logits[..., :-1, :].contiguous().float()
    labels = labels[..., 1:].contiguous().to(logits.device)
    total = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1),
                            ignore_index=-100, reduction='sum')
    if num_items_in_batch is None:  # evaluation: per-batch token mean
        num_items_in_batch = labels.ne(-100).sum()
    return total / num_items_in_batch


def main(argv=None):
    a = arguments(argv)
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('independent single-GPU jobs only; do not use torchrun')
    if a.seq_length < 2 or min(a.batch_size, a.gradient_accumulation, a.eval_steps) < 1 or a.max_gpu_gb < 0:
        raise ValueError('invalid training dimensions/memory cap')
    if Path(a.train).resolve() == Path(a.validation).resolve():
        raise ValueError('training and validation must be distinct')
    from aspire.training.frozen import encode_frozen, verify_frozen
    out = Path(a.output)
    config = {k: v for k, v in vars(a).items() if k != 'resume'}
    config['data_hashes'] = {k: file_hash(getattr(a, k)) for k in ('train', 'validation')}
    config['source_hashes'] = {p.name: file_hash(p) for p in Path(__file__).parent.glob('*.py')}
    config['fim_manifest_sha256'] = verify_frozen(a.train, a.validation, a.model)
    model_config = json.loads((Path(a.model) / 'config.json').read_text(encoding='utf-8'))
    if model_config.get('quantization_config'):
        raise ValueError('BF16 LoRA requires original weights, not AWQ/GPTQ')
    if out.exists() and any(out.iterdir()):
        if not a.resume:
            raise ValueError('nonempty output requires --resume; choose a new output for new experiments')
        previous = json.loads((out / 'experiment.json').read_text(encoding='utf-8'))
        if previous['config'] != config:
            raise ValueError('resume configuration/data/code mismatch')
        if Path(a.resume).resolve().parent != out.resolve():
            raise ValueError('resume checkpoint must belong to this output directory')
    elif a.resume:
        raise ValueError('resume requires the original experiment directory')

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model
    from aspire.training.fim import AnswerCollator
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('set CUDA_VISIBLE_DEVICES to exactly ONE GPU before launching')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 GPU required')
    if a.max_gpu_gb > 0:
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        torch.cuda.set_per_process_memory_fraction(min(a.max_gpu_gb / total, 1), 0)
    set_seed(a.seed)
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train, train_stats = encode_frozen(a.train, tokenizer, 'sft', a.seq_length, False, a.seed)
    validation, val_stats = encode_frozen(a.validation, tokenizer, 'sft', a.seq_length, False, a.seed)
    # Unquantized BF16 base weights (~61 GiB for 32B); only the LoRA parameters are trained.
    model = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True, device_map={'': 0},
                                                 torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout,
                                             bias='none', task_type='CAUSAL_LM', target_modules='all-linear'))
    model.config.use_cache = False
    model.print_trainable_parameters()
    versions = {k: importlib.metadata.version(k)
                for k in ['torch', 'transformers', 'peft', 'bitsandbytes', 'datasets', 'accelerate']}
    write_json(out / 'experiment.json', {
        'config': config, 'versions': versions, 'train': train_stats, 'validation': val_stats,
        'gpu': torch.cuda.get_device_name(0), 'effective_batch': a.batch_size * a.gradient_accumulation,
        'loss': 'answer+EOS token NLL summed over the optimizer update / supervised tokens in the update'})
    args = TrainingArguments(
        output_dir=str(out), num_train_epochs=a.epochs, max_steps=a.max_steps,
        per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=1,
        gradient_accumulation_steps=a.gradient_accumulation, learning_rate=a.learning_rate,
        lr_scheduler_type=a.scheduler, warmup_ratio=.03, bf16=True, fp16=False,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        optim='paged_adamw_8bit', max_grad_norm=.3, weight_decay=.001,
        eval_strategy='steps', eval_steps=a.eval_steps, save_strategy='steps', save_steps=a.eval_steps,
        save_total_limit=2, logging_steps=5, logging_first_step=True, report_to='none',
        seed=a.seed, data_seed=a.seed, prediction_loss_only=True, label_names=['labels'],
        remove_unused_columns=False, include_num_input_tokens_seen=True)
    trainer = Trainer(model=model, args=args, train_dataset=Dataset.from_list(train),
                      eval_dataset=Dataset.from_list(validation),
                      data_collator=AnswerCollator(tokenizer.pad_token_id),
                      compute_loss_func=token_sum_loss)
    start = time.monotonic()
    before = trainer.evaluate()
    result = trainer.train(resume_from_checkpoint=a.resume)
    after = trainer.evaluate()
    trainer.save_model(str(out / 'adapter'))
    tokenizer.save_pretrained(out / 'adapter')
    trainer.save_state()
    write_json(out / 'metrics.json', {
        'before': before, 'after': after, 'train': result.metrics,
        'elapsed_seconds': time.monotonic() - start,
        'peak_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3,
        'global_step': trainer.state.global_step, 'input_tokens_seen': trainer.state.num_input_tokens_seen})
    write_json(out / 'COMPLETE.json', {
        'adapter': str(out / 'adapter'),
        'files': {p.name: file_hash(p) for p in (out / 'adapter').glob('adapter*') if p.is_file()},
        'experiment_sha256': file_hash(out / 'experiment.json')})


if __name__ == '__main__':
    main()
