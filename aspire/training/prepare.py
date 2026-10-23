"""Freeze the paired SFT / Random-FIM / AST-FIM training sets (Sec. 2.2-2.3, 3.2). No training.

All three arms share source rows. A fixed fraction rho of rows is selected; in
those slots AST-FIM replaces the full-program target by an AST span (complete
rule, body literal, comparison literal), and Random-FIM by a contiguous span on
tokenizer boundaries whose supervised token count (answer + EOS) is identical.
If either arm cannot be built, BOTH fall back to full-program supervision.
"""
import argparse
from collections import Counter
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import random
import sys

from aspire.training.common import digest, file_hash, rows, write_json
from aspire.training.fim import (ast_all_spans, ast_spans, choose_ast, encode_answer, infill_record,
                                 match_random, split_source)
from aspire.training.asp_ast import parse


def normalized_program(answer):
    # Formatting-normalized only; neither alpha-invariant nor semantic equivalence.
    try:
        return digest([str(n) for n in parse(answer)])
    except (RuntimeError, ValueError):
        return None


def write_jsonl(path, records):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')


def distribution(values):
    values = sorted(values)
    if not values:
        return dict(n=0)
    return dict(n=len(values), min=values[0], max=values[-1], mean=sum(values) / len(values),
                median=values[len(values) // 2], p95=values[min(len(values) - 1, int(len(values) * .95))])


def prepare(train_path, validation_path, output, tokenizer, rate=.2, seed=20260924,
            limit=1024, attempts=512, tokenizer_path=None):
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('output must be empty; frozen datasets are never overwritten')
    if Path(train_path).resolve() == Path(validation_path).resolve():
        raise ValueError('distinct train and validation sources required')
    if not 0 <= rate <= 1 or attempts < 1 or limit < 2:
        raise ValueError('invalid rate, attempts or sequence length')
    train, validation = list(rows(train_path)), list(rows(validation_path))
    if not train or not validation:
        raise ValueError('empty source split')
    for row in train + validation:
        if not all(isinstance(row.get(k), str) and row[k].strip() for k in ('question', 'answer')):
            raise ValueError('each source row needs nonempty question and answer')
    selected = set(random.Random(seed).sample(range(len(train)), int(len(train) * rate)))
    # ast_any (generic AST-subtree baseline) is converted in exactly the slots where
    # the AST-FIM/Random-FIM pair was built; its span is uniform over all AST nodes
    # and its length is not matched. If it cannot be built there, only it falls back.
    arms = {name: [] for name in ('plain', 'random_fim', 'ast_fim', 'ast_any')}
    any_fallbacks = Counter()
    totals = {name: Counter() for name in arms}
    evidence, failures, available, chosen = [], Counter(), Counter(), Counter()
    normalized_train = set()
    qa_train = {digest([r['question'], r['answer']]) for r in train}
    raw_programs = {digest(r['answer']) for r in train}
    source_hash = file_hash(train_path)
    for index, row in enumerate(train):
        source_id = digest([source_hash, index])
        common = dict(source_id=source_id, source_row=index + 1,
                      program_sha256=digest(row['answer']), split='train')
        ordinary = dict(question=row['question'], answer=row['answer'], task='sft', **common)
        _, plain_stats = encode_answer(tokenizer, row['question'], row['answer'], limit)
        current = {name: dict(ordinary) for name in arms}
        stats = {name: plain_stats for name in arms}
        error = None
        try:
            spans = ast_spans(row['answer'])
            normalized_train.add(normalized_program(row['answer']))
        except (RuntimeError, ValueError) as exc:
            spans, error = [], type(exc).__name__
        available.update({kind: 1 for kind in {s['kind'] for s in spans}})
        audit = dict(**common, selected=index in selected,
                     available_span_count=len(spans), parse_error=error)
        if index in selected:
            rng = random.Random(digest([seed, source_id, 'ast']))
            span = choose_ast(spans, rng)
            reason = 'parse_error' if error else 'no_supported_span'
            if span:
                ast_record = infill_record(row['question'], row['answer'], span)
                try:
                    _, ast_stats = encode_answer(tokenizer, **ast_record, limit=limit)
                    random_span, random_stats, searched = match_random(
                        tokenizer, row['question'], row['answer'], ast_stats['supervised_tokens'],
                        random.Random(digest([seed, source_id, 'random'])), limit, attempts)
                    audit['random_candidates_checked'] = searched
                    reason = 'no_exact_length_match'
                    if random_span:
                        random_record = infill_record(row['question'], row['answer'], random_span)
                        for name, record, unit, st in (
                            ('ast_fim', ast_record, span, ast_stats),
                            ('random_fim', random_record, random_span, random_stats)):
                            p, m, s = split_source(row['answer'], unit)
                            current[name] = dict(**record, **common, task='fim', span=unit,
                                                 prefix=p, suffix=s)
                            assert p + m + s == row['answer']
                            stats[name] = st
                        ast_bounds = {(s['start'], s['end']) for s in spans}
                        audit.update(ast_span=span, random_span=random_span,
                                     random_is_supported_ast_span=(random_span['start'], random_span['end']) in ast_bounds,
                                     random_equals_selected_ast=(span['start'], span['end']) == (random_span['start'], random_span['end']),
                                     ast_relative_start=span['start'] / len(row['answer']),
                                     random_relative_start=random_span['start'] / len(row['answer']),
                                     reconstruction_checked=True)
                        chosen[span['kind']] += 1
                        reason = None
                        any_spans = ast_all_spans(row['answer'])
                        any_span = random.Random(digest([seed, source_id, 'ast_any'])).choice(any_spans) \
                            if any_spans else None
                        if any_span is None:
                            any_fallbacks['no_span'] += 1
                        else:
                            any_record = infill_record(row['question'], row['answer'], any_span)
                            try:
                                _, any_stats = encode_answer(tokenizer, **any_record, limit=limit)
                                p, _, s = split_source(row['answer'], any_span)
                                current['ast_any'] = dict(**any_record, **common, task='fim', span=any_span,
                                                          prefix=p, suffix=s)
                                stats['ast_any'] = any_stats
                                audit.update(ast_any_span=any_span, ast_any_candidates=len(any_spans),
                                             ast_any_is_semantic_unit=(any_span['start'], any_span['end']) in ast_bounds)
                            except ValueError as exc:
                                if not str(exc).startswith('overlength:'):
                                    raise
                                any_fallbacks['fim_overlength'] += 1
                except ValueError as exc:
                    if not str(exc).startswith('overlength:'):
                        raise
                    reason = 'fim_overlength'
            audit['fallback_reason'] = reason
            if reason:
                failures[reason] += 1
        for name in arms:
            arms[name].append(current[name])
            totals[name].update({k: stats[name][k] for k in ('input_tokens', 'supervised_tokens')})
        audit['encoding'] = stats
        evidence.append(audit)
        if (index + 1) % 500 == 0:
            print(f'Preflight: {index + 1}/{len(train)} rows, {sum(chosen.values())} FIM pairs',
                  file=sys.stderr, flush=True)
    normalized_train.discard(None)
    val_rows, clean_rows = [], []
    overlap = Counter()
    val_stats = Counter()
    for index, row in enumerate(validation):
        _, st = encode_answer(tokenizer, row['question'], row['answer'], limit)
        val_stats.update({k: st[k] for k in ('input_tokens', 'supervised_tokens')})
        record = dict(question=row['question'], answer=row['answer'], split='validation',
                      task='sft', source_row=index + 1)
        val_rows.append(record)
        exact = digest([row['question'], row['answer']]) in qa_train
        norm = normalized_program(row['answer'])
        overlap['exact_qa_rows'] += exact
        overlap['exact_program_rows'] += digest(row['answer']) in raw_programs
        overlap['normalized_program_rows'] += norm is not None and norm in normalized_train
        overlap['unparseable_validation_rows'] += norm is None
        if not exact:
            clean_rows.append(record)
    complete = sum(chosen.values())
    report = dict(train_rows=len(train), validation_rows=len(validation),
                  requested_fim_slots=len(selected), accepted_fim_pairs=complete,
                  unparseable_train_rows=sum(r['parse_error'] is not None for r in evidence),
                  actual_fim_rate=complete / len(train), fallback_counts=dict(failures),
                  train_rows_with_span_type=dict(available), selected_ast_types=dict(chosen),
                  training_tokens={k: dict(v) for k, v in totals.items()},
                  validation_tokens=dict(val_stats), overlap=dict(overlap),
                  validation_exact_qa_nonoverlap_rows=len(clean_rows),
                  random_coincides_with_supported_ast=sum(r.get('random_is_supported_ast_span', False) for r in evidence),
                  random_equals_selected_ast=sum(r.get('random_equals_selected_ast', False) for r in evidence),
                  ast_any_pairs=sum('ast_any_span' in r for r in evidence),
                  ast_any_fallbacks=dict(any_fallbacks),
                  ast_any_is_semantic_unit=sum(r.get('ast_any_is_semantic_unit', False) for r in evidence),
                  scope='syntax/source reconstruction only; no semantic correctness or generation claim',
                  normalization='ordered Clingo AST strings, names retained; not semantic or template decontamination',
                  random_policy='bounded uniform token-boundary span reservoir, exact supervised-token matching including EOS',
                  ready_for_smoke=complete > 0 and bool(clean_rows))
    report['token_distributions'] = {
        name: {field: distribution([r['encoding'][name][field] for r in evidence])
               for field in ('input_tokens', 'supervised_tokens')} for name in arms}
    paired = [r for r in evidence if r.get('reconstruction_checked')]
    report['fim_distributions'] = {
        name: dict(
            supervised_tokens=distribution([r['encoding'][name]['supervised_tokens'] for r in paired]),
            relative_start=distribution([r['ast_relative_start' if name == 'ast_fim' else 'random_relative_start']
                                         for r in paired]),
            prefix_chars=distribution([x['span']['start'] for x in arms[name] if x['task'] == 'fim']),
            suffix_chars=distribution([len(x['suffix']) for x in arms[name] if x['task'] == 'fim']))
        for name in ('ast_fim', 'random_fim')}
    # Preflight completes in memory before writing; manifest is the final completion marker.
    out.mkdir(parents=True, exist_ok=True)
    for name, records in arms.items():
        write_jsonl(out / f'{name}.jsonl', records)
    write_jsonl(out / 'validation.jsonl', val_rows)
    write_jsonl(out / 'validation_exact_nonoverlap.jsonl', clean_rows)
    write_jsonl(out / 'audit.jsonl', evidence)
    write_json(out / 'report.json', report)
    examples = []
    for kind in sorted(chosen):
        ids = [i for i, r in enumerate(evidence) if r.get('ast_span', {}).get('kind') == kind][:3]
        for i in ids:
            examples.append(dict(source_id=evidence[i]['source_id'], kind=kind,
                                 original=train[i], ast=arms['ast_fim'][i], random=arms['random_fim'][i]))
    write_json(out / 'examples.json', examples)
    lines = ['# ASP FIM preflight', '', 'No model training or generation has been run.', '',
             f'- Source rows: {len(train)} train / {len(validation)} validation.',
             f'- FIM slots: {len(selected)} requested / {complete} paired; actual rate {complete / len(train):.2%}.',
             f'- Fallbacks: {dict(failures)}.', f'- AST types: {dict(chosen)}.',
             f'- Validation overlap: {dict(overlap)}.',
             f'- Ready for smoke (basic data checks only): {report["ready_for_smoke"]}.', '',
             'Inspect examples.json and audit.jsonl for spans, lengths, positions and prompt-boundary evidence.',
             'Exact source reconstruction is not proof that the reference ASP satisfies its question.',
             'Random matches can coincide with AST spans; those matches remain in the baseline.',
             'The three arms have equal source slots, not necessarily equal supervised tokens or compute.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    source_files = [Path(__file__), Path(__file__).with_name('fim.py'),
                    Path(__file__).with_name('common.py'), Path(__file__).with_name('asp_ast.py')]
    tok_files = {}
    if tokenizer_path:
        tok_files = {p.name: file_hash(p) for p in Path(tokenizer_path).iterdir()
                     if p.is_file() and (p.name.startswith('tokenizer') or p.name in
                                        ('vocab.json', 'merges.txt', 'special_tokens_map.json', 'added_tokens.json'))}
    write_json(out / 'manifest.json', dict(
        complete=True, schema=1, config=dict(rate=rate, seed=seed, seq_length=limit, attempts=attempts),
        source_files={str(Path(p).resolve()): file_hash(p) for p in (train_path, validation_path)},
        code_hashes={p.name: file_hash(p) for p in source_files}, tokenizer_hashes=tok_files,
        tokenizer_class=type(tokenizer).__name__, eos_token_id=tokenizer.eos_token_id,
        versions={name: importlib.metadata.version(name) for name in ('clingo',)} | {
            name: importlib.metadata.version(name) for name in ('transformers', 'tokenizers')
            if importlib.util.find_spec(name) is not None},
        files={p.name: file_hash(p) for p in out.iterdir() if p.is_file()}))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('train', 'validation', 'output', 'tokenizer'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--rate', type=float, default=.2)
    p.add_argument('--seed', type=int, default=20260924)
    p.add_argument('--seq-length', type=int, default=1024)
    p.add_argument('--attempts', type=int, default=512)
    a = p.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True, use_fast=True)
    result = prepare(a.train, a.validation, a.output, tokenizer, a.rate, a.seed,
                     a.seq_length, a.attempts, a.tokenizer)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
