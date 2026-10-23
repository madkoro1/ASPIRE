"""Lightweight ASP extraction shared by API-only experiment drivers.

This mirrors the extraction contract in ``model_testing_extension.py`` without
importing torch/transformers or allocating a local model.  In particular it
does not use the reference program or its rule count.
"""

import re


_ANSWER_MARKER_RE = re.compile(r"(?im)^\s*Answer\s*:\s*")
_NEXT_SECTION_RE = re.compile(
    r"(?im)^\s*(?:Question|Explanation|Reasoning|Notes?)\s*:\s*")
_FENCE_RE = re.compile(
    r"```(?:asp|clingo|prolog|logic)?[ \t]*\r?\n?(.*?)(?:```|$)",
    re.IGNORECASE | re.DOTALL,
)
_ASP_START_RE = re.compile(
    r"""(?x)
    ^(?:
        :- | :~ | \# | \{ | -[a-z][A-Za-z0-9_]* |
        [0-9]+\s*\{ |
        [a-z][A-Za-z0-9_]*\s*(?:\(|:-|\.|\{)
    )
    """
)


def _strip_asp_comments(text):
    """Remove ASP line/block comments without touching percent in strings."""
    out = []
    i = 0
    in_string = False
    escaped = False
    in_line_comment = False
    in_block_comment = False
    while i < len(text):
        c = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if in_line_comment:
            if c in "\r\n":
                in_line_comment = False
                out.append(c)
            i += 1
            continue
        if in_block_comment:
            if c == "*" and nxt == "%":
                in_block_comment = False
                i += 2
            else:
                if c in "\r\n":
                    out.append(c)
                i += 1
            continue
        if in_string:
            out.append(c)
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                in_string = False
            i += 1
            continue
        if c == '"':
            in_string = True
            out.append(c)
            i += 1
            continue
        if c == "%" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if c == "%":
            in_line_comment = True
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _looks_like_asp_statement(statement):
    stripped = statement.strip()
    return bool(stripped and _ASP_START_RE.match(stripped))


def candidate_program_text(generated_text):
    """Return only the portion of a decoded response that can contain ASP."""
    marker_matches = list(_ANSWER_MARKER_RE.finditer(generated_text))
    if marker_matches:
        generated_text = generated_text[marker_matches[-1].end():]

    fence_positions = []
    for fenced in _FENCE_RE.finditer(generated_text):
        body = fenced.group(1)
        fence_positions.append(fenced.start())
        if any(_looks_like_asp_statement(line) for line in body.splitlines()):
            return body
    if fence_positions:
        generated_text = generated_text[:fence_positions[0]]

    section = _NEXT_SECTION_RE.search(generated_text)
    if section:
        generated_text = generated_text[:section.start()]

    lines = generated_text.replace("`", "").splitlines()
    for line_index, line in enumerate(lines):
        stripped = line.strip()
        for prefix in ("ASP:", "Program:", "Code:"):
            if stripped.lower().startswith(prefix.lower()):
                stripped = stripped[len(prefix):].strip()
                lines[line_index] = stripped
                break
        if _ASP_START_RE.match(stripped):
            return "\n".join(lines[line_index:])
    return generated_text.replace("`", "")


def split_asp_statements(text):
    """Extract complete ASP statements without using a gold rule count."""
    text = _strip_asp_comments(text)
    statements = []
    start = 0
    i = 0
    in_string = False
    escaped = False
    bracket_depth = 0
    weak_body_ended = False

    def append_statement(end):
        statement = text[start:end].strip()
        if _looks_like_asp_statement(statement):
            statements.append(statement)

    while i < len(text):
        c = text[i]
        prev = text[i - 1] if i else ""
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if in_string:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                in_string = False
            i += 1
            continue
        if c == '"':
            in_string = True
            i += 1
            continue

        current = text[start:i].lstrip()
        is_weak = current.startswith(":~")
        if c == "[":
            bracket_depth += 1
        elif c == "]" and bracket_depth:
            bracket_depth -= 1
            if is_weak and weak_body_ended and bracket_depth == 0:
                end = i + 1
                j = end
                while j < len(text) and text[j] in " \t":
                    j += 1
                if j < len(text) and text[j] == ".":
                    end = j + 1
                    i = j
                append_statement(end)
                start = end
                weak_body_ended = False
        elif c == "." and bracket_depth == 0:
            if prev == "." or nxt == "." or (prev.isdigit() and nxt.isdigit()):
                i += 1
                continue
            if is_weak:
                weak_body_ended = True
            else:
                append_statement(i + 1)
                start = i + 1
        i += 1
    return statements


def parse_answer(generated_text):
    """Return ``(program, statements)`` using the canonical extraction rules."""
    if not isinstance(generated_text, str):
        return "", []
    statements = split_asp_statements(candidate_program_text(generated_text))
    return "\n".join(statements), statements
