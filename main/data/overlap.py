"""Train/test overlap of generated data (Sec. 4.1). Clingo + pandas only.

Counts, per test family:
  exact_qa        test (question, program) pair occurs in training
  exact_program   test reference program string occurs in training
  renamed_program equal after alpha-renaming predicates, variables and string
                  constants in order of first occurrence (structural template reuse)
This measures template sharing; it is not a semantic-equivalence check.
"""
import argparse
from collections import Counter, defaultdict
import json
import re

import pandas as pd

from aspire.families import PAPER_FAMILIES

_TOKEN = re.compile(r'"[^"]*"|[A-Za-z_][A-Za-z0-9_]*|\S')
_KEYWORDS = {"not"}


def renamed(program):
    """Alpha-rename predicates/constants (lower-case), variables (upper-case) and strings."""
    maps = defaultdict(dict)
    out = []
    for tok in _TOKEN.findall(program):
        if tok in _KEYWORDS or tok.startswith("#"):
            kind = None
        elif tok.startswith('"'):
            kind = "s"
        elif tok[0].isupper() or tok[0] == "_":
            kind = "V"
        elif tok[0].isalpha():
            kind = "p"
        else:
            kind = None
        if kind is None:
            out.append(tok)
            continue
        table = maps[kind]
        out.append(table.setdefault(tok, f"{kind}{len(table)}"))
    return " ".join(out)


def overlap(train, test):
    qa = {(q, a) for q, a in zip(train["question"], train["answer"])}
    programs = set(train["answer"])
    shapes = {renamed(a) for a in train["answer"]}
    result = defaultdict(Counter)
    for q, a, task in zip(test["prompt"], test["answer"], test["task"]):
        for key in ("all", task):
            r = result[key]
            r["n"] += 1
            r["exact_qa"] += (q, a) in qa
            r["exact_program"] += a in programs
            r["renamed_program"] += renamed(a) in shapes
    return {k: dict(v) for k, v in sorted(result.items())}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args(argv)
    train = pd.read_csv(a.train, keep_default_na=False)
    test = pd.read_csv(a.test, keep_default_na=False)
    test = test[test["task"].isin(PAPER_FAMILIES)]
    report = overlap(train, test)
    with open(a.output, "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report["all"], ensure_ascii=False))


if __name__ == "__main__":
    main()
