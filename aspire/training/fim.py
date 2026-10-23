"""Source-preserving ASP infilling records and answer-only encoding (Eq. 1-2). No model imports."""
import random
from collections import defaultdict

from aspire.training.asp_ast import parse


def ast_spans(source):
    """Supported v1 units only. Clingo columns are UTF-8 byte positions."""
    import clingo.ast as ast
    lines = source.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))

    def offset(pos):
        if pos.line == len(lines) + 1 and pos.column == 1:
            return len(source)
        raw = lines[pos.line - 1].encode('utf-8')
        return starts[pos.line - 1] + len(raw[:pos.column - 1].decode('utf-8'))

    found = {}

    def add(node, kind):
        begin, end = offset(node.location.begin), offset(node.location.end)
        if not 0 <= begin < end <= len(source):
            raise ValueError('invalid AST source location')
        if not source[begin:end].strip():
            return
        if not (source[:begin] + source[end:]).strip():
            return  # Do not turn infilling into whole-program generation.
        found[begin, end] = dict(start=begin, end=end, kind=kind)

    for node in parse(source):
        if node.ast_type != ast.ASTType.Rule or not node.body:
            continue
        # No aggregate/optimization-internal masking in v1; a whole rule is OK.
        add(node, 'rule')
        for literal in node.body:
            if literal.ast_type != ast.ASTType.Literal:
                continue
            atom = literal.atom.ast_type
            if atom == ast.ASTType.Comparison:
                add(literal, 'comparison')
            elif atom == ast.ASTType.SymbolicAtom:
                add(literal, 'body_literal')
    return sorted(found.values(), key=lambda x: (x['start'], x['end']))


def ast_all_spans(source):
    """Generic AST-subtree baseline (AST-T5 / AST-FIM style): every located node.

    Rules, head and body literals, atoms, comparisons, terms and variables all
    qualify, without regard to ASP semantic units; duplicates of the same
    character span are kept once. Same non-empty/context constraints as ast_spans.
    """
    import clingo.ast as ast
    lines = source.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))

    def offset(pos):
        if pos.line == len(lines) + 1 and pos.column == 1:
            return len(source)
        raw = lines[pos.line - 1].encode('utf-8')
        return starts[pos.line - 1] + len(raw[:pos.column - 1].decode('utf-8'))

    found = {}

    def visit(node):
        try:
            location = node.location
        except (AttributeError, RuntimeError):
            location = None
        if location is not None:
            begin, end = offset(location.begin), offset(location.end)
            if (0 <= begin < end <= len(source) and source[begin:end].strip()
                    and (source[:begin] + source[end:]).strip()):
                found.setdefault((begin, end), dict(start=begin, end=end, kind='any_node'))
        for key in node.child_keys:
            child = getattr(node, key)
            if isinstance(child, ast.AST):
                visit(child)
            elif child is not None and not isinstance(child, (str, int)):
                try:
                    items = list(child)
                except TypeError:
                    continue
                for item in items:
                    if isinstance(item, ast.AST):
                        visit(item)

    for node in parse(source):
        visit(node)
    return sorted(found.values(), key=lambda x: (x['start'], x['end']))


def choose_ast(spans, rng):
    groups = defaultdict(list)
    for span in spans:
        groups[span['kind']].append(span)
    if not groups:
        return None
    return rng.choice(groups[rng.choice(sorted(groups))])


def split_source(source, span):
    start, end = span['start'], span['end']
    if not 0 <= start < end <= len(source):
        raise ValueError('invalid span')
    parts = source[:start], source[start:end], source[end:]
    if ''.join(parts) != source or not parts[1].strip() or not (parts[0] + parts[2]).strip():
        raise ValueError('invalid reconstruction or empty context')
    return parts


def infill_record(question, source, span):
    prefix, middle, suffix = split_source(source, span)
    # Deterministic delimiters must not occur in user/program text.
    base = 'ASP_FIM'
    while any(f'<{base}_{name}>' in question + source or
              f'</{base}_{name}>' in question + source
              for name in ('TASK', 'PREFIX', 'SUFFIX')):
        base += '_'
    task = ('Complete the missing ASP code. Return only the missing fragment, '
            'without Markdown. Concatenate PREFIX + your answer + SUFFIX to form the program.\n'
            f'<{base}_TASK>{question}</{base}_TASK>\n'
            f'<{base}_PREFIX>{prefix}</{base}_PREFIX>\n'
            f'<{base}_SUFFIX>{suffix}</{base}_SUFFIX>')
    return dict(question=task, answer=middle)


def encode_answer(tokenizer, question, answer, limit=1024):
    if not tokenizer.is_fast or tokenizer.eos_token_id is None:
        raise ValueError('fast tokenizer with EOS required')
    if not answer.strip():
        raise ValueError('empty answer')
    prefix = f'Question: {question}\nAnswer: '
    enc = tokenizer(prefix + answer, add_special_tokens=False, return_offsets_mapping=True)
    ids, labels, crossings = list(enc['input_ids']), [], []
    for i, (start, end) in enumerate(enc['offset_mapping']):
        if end > len(prefix):
            if start < len(prefix):
                if prefix[start:].strip():
                    raise ValueError('answer token crosses non-whitespace prompt text')
                crossings.append(i)
            labels.append(ids[i])
        else:
            labels.append(-100)
    if not any(x != -100 for x in labels) or labels[0] != -100:
        raise ValueError('missing answer or preceding causal context')
    ids.append(tokenizer.eos_token_id)
    labels.append(tokenizer.eos_token_id)
    if len(ids) > limit:
        raise ValueError(f'overlength: {len(ids)} > {limit}; no truncation')
    return dict(input_ids=ids, labels=labels), dict(
        input_tokens=len(ids), supervised_tokens=sum(x != -100 for x in labels),
        boundary_crossing_tokens=crossings)


def match_random(tokenizer, question, source, target_tokens, rng, limit=1024, attempts=512):
    """Uniform shuffled token-boundary spans, conditional on exact target length.

    No AST lookup is used. A bounded search may fail; callers fall back BOTH arms.
    Reservoir sampling avoids quadratic memory for long code.
    """
    offsets = tokenizer(source, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
    boundaries = sorted({0, len(source)} | {x for a, b in offsets if b > a for x in (a, b)})
    pool, seen = [], 0
    for i, start in enumerate(boundaries[:-1]):
        for end in boundaries[i + 1:]:
            if not source[start:end].strip() or not (source[:start] + source[end:]).strip():
                continue
            seen += 1
            item = dict(start=start, end=end, kind='random')
            if len(pool) < attempts:
                pool.append(item)
            else:
                j = rng.randrange(seen)
                if j < attempts:
                    pool[j] = item
    rng.shuffle(pool)
    for index, span in enumerate(pool):
        record = infill_record(question, source, span)
        try:
            _, stats = encode_answer(tokenizer, **record, limit=limit)
        except ValueError as exc:
            if str(exc).startswith('overlength:'):
                continue
            raise
        if stats['supervised_tokens'] == target_tokens:
            return span, stats, index + 1
    return None, None, len(pool)


class AnswerCollator:
    """Preserve labels; real EOS remains supervised even when pad_id == eos_id."""
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, examples):
        import torch
        width = max(len(r['input_ids']) for r in examples)
        inputs, labels, masks = [], [], []
        for row in examples:
            n = len(row['input_ids'])
            if len(row['labels']) != n:
                raise ValueError('label/input length mismatch')
            inputs.append(row['input_ids'] + [self.pad_id] * (width - n))
            labels.append(row['labels'] + [-100] * (width - n))
            masks.append([1] * n + [0] * (width - n))
        return {k: torch.tensor(v, dtype=torch.long) for k, v in
                [('input_ids', inputs), ('labels', labels), ('attention_mask', masks)]}
