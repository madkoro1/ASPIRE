"""Verified frozen FIM datasets, shared by all model sizes."""
import json
from pathlib import Path
from aspire.training.common import file_hash, rows


FROZEN_ARMS = ('plain.jsonl', 'random_fim.jsonl', 'ast_fim.jsonl', 'ast_any.jsonl')


def verify_frozen(train, validation, model):
    root = Path(train).resolve().parent
    if Path(train).name not in FROZEN_ARMS:
        raise ValueError('FIM train must be a frozen arm JSONL')
    if Path(validation).resolve().parent != root or Path(validation).name not in (
            'validation.jsonl', 'validation_exact_nonoverlap.jsonl'):
        raise ValueError('FIM validation must belong to the same frozen dataset')
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if not manifest.get('complete') or manifest.get('schema') != 1:
        raise ValueError('Incomplete or unsupported FIM manifest')
    # ast_any exists only in FIM directories built by the current prepare.py.
    required = ('plain.jsonl', 'random_fim.jsonl', 'ast_fim.jsonl', Path(train).name, Path(validation).name)
    for name in required:
        if name not in manifest['files']:
            raise ValueError(f'Missing manifest file: {name}')
    for name, expected in manifest['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or file_hash(path) != expected:
            raise ValueError(f'Frozen data hash mismatch: {name}')
    # Token-length matching depends on tokenizer behavior, not model size.
    tok_hashes = manifest.get('tokenizer_hashes', {})
    if not tok_hashes or 'tokenizer.json' not in tok_hashes:
        raise ValueError('Frozen data lacks real tokenizer fingerprints; rerun fim_prepare')
    for name, expected in tok_hashes.items():
        path = (Path(model) / name).resolve()
        if not path.is_relative_to(Path(model).resolve()) or not path.is_file() or file_hash(path) != expected:
            raise ValueError(f'Tokenizer mismatch: {name}; prepare a NEW dataset directory with this model tokenizer')
    return file_hash(root / 'manifest.json')


def encode_frozen(path, tokenizer, stage, seq_length, packing, seed=56):
    from aspire.training.fim import encode_answer
    if stage != 'sft' or packing:
        raise ValueError('FIM requires unpacked SFT')
    records = list(rows(path))
    if not records:
        raise ValueError('Empty FIM dataset')
    # Keep source row order identical across arms; Trainer provides seeded sampling.
    output, counts = [], dict(documents=len(records), tokens=0, supervised_tokens=0, fim_documents=0)
    for row in records:
        encoded, stats = encode_answer(tokenizer, row['question'], row['answer'], seq_length)
        output.append(encoded)
        counts['tokens'] += stats['input_tokens']
        counts['supervised_tokens'] += stats['supervised_tokens']
        counts['fim_documents'] += row.get('task') == 'fim'
    counts['sequences'] = len(output)
    return output, counts
