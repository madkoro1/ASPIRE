"""Read-only cloud preflight. Does not load weights or start training."""
import argparse
import json
from pathlib import Path


def check_files(model):
    model = Path(model)
    config = json.loads((model / 'config.json').read_text(encoding='utf-8'))
    if config.get('quantization_config'):
        raise ValueError('Use original weights, not AWQ/GPTQ/other prequantized checkpoint')
    if (config.get('model_type'), config.get('num_hidden_layers'), config.get('hidden_size')) != ('qwen2', 64, 5120):
        raise ValueError('Config does not match expected Qwen2.5-32B architecture; check download')
    index = json.loads((model / 'model.safetensors.index.json').read_text(encoding='utf-8'))
    shards = sorted(set(index['weight_map'].values()))
    if not shards:
        raise ValueError('Empty weight index')
    for shard in shards:
        path = (model / shard).resolve()
        if not path.is_relative_to(model.resolve()) or not path.is_file():
            raise ValueError(f'Missing or invalid shard: {shard}')
    if not (model / 'tokenizer.json').is_file():
        raise ValueError('Missing tokenizer.json')
    return dict(model=str(model.resolve()), shards=len(shards),
                weight_files_gib=sum((model / s).stat().st_size for s in shards) / 1024**3,
                note='Config and shard presence checked; no weight integrity or memory-fit guarantee')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, help='local original Qwen2.5-32B Base weights')
    p.add_argument('--max-gpu-gb', type=float, default=0, help='0 = no cap')
    args = p.parse_args()
    result = check_files(args.model)
    import torch
    import transformers
    import peft
    import bitsandbytes  # paged 8-bit AdamW only; model weights are not quantized
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one GPU with CUDA_VISIBLE_DEVICES=0')
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if not 0 <= args.max_gpu_gb <= total:
        raise ValueError('GPU cap must be 0 (no cap) or within physical GPU memory')
    budget = args.max_gpu_gb or total
    if budget < result['weight_files_gib'] + 4:
        raise ValueError(f"BF16 weights ({result['weight_files_gib']:.1f} GiB) do not fit a {budget:.1f} GiB budget")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 not available in this environment')
    tok = transformers.AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    if not tok.is_fast or tok.eos_token_id is None:
        raise ValueError('Fast tokenizer and EOS required')
    result.update(gpu=torch.cuda.get_device_name(0), total_gib=total, planned_cap_gib=args.max_gpu_gb,
                  torch=torch.__version__, cuda=torch.version.cuda,
                  transformers=transformers.__version__, peft=peft.__version__,
                  bitsandbytes=bitsandbytes.__version__, training_started=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
