"""Clingo AST parsing for span selection and formatting-normalized program hashes."""
import re

import clingo.ast as ast

from aspire.training.common import digest


def parse(text):
    if re.search(r'#\s*(include|script)\b', text, re.I):
        raise ValueError('#include and #script are forbidden')
    nodes = []
    ast.parse_string(text, nodes.append, logger=lambda *args: None)
    return [n for n in nodes if n.ast_type not in (ast.ASTType.Program, ast.ASTType.Comment)]


def normalized_hash(text):
    """Ordered AST strings with names retained; NOT a semantic-equivalence hash."""
    return digest([str(n) for n in parse(text)])
