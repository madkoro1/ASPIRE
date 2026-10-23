"""Failure taxonomy of generated programs and counterexample hard-coding (CPU, clingo only).

Two analyses over completed runs (protocol_samples.jsonl), no model calls:

1. Error types (Sec. 3.5 (1) and the introduction): every sample that fails the strict
   criterion is assigned ONE outcome category, in this order of precedence:
     no_program     nothing could be extracted
     syntax         parse or grounding error on some evaluated context
     timeout        search did not finish
     interface      no rule head defines a declared output predicate/arity
     unsat_mismatch satisfiability differs on some context
     multi_model    some context has a different number of projected models
     over           only spurious output atoms (too permissive: typically a missing condition)
     under          only missing output atoms (too restrictive / broken derivation chain)
     mixed          both spurious and missing atoms
   Structural indicators are counted in addition (not exclusive), comparing the
   candidate with the reference program:
     fewer_comparisons  fewer comparison literals than the reference
     fewer_negations    fewer default-negated literals than the reference
     undefined_body_pred  a body predicate that no rule, background or fact defines
2. Hard-coding (Sec. 3.5 (3)): for refinement runs, a final program "hard-codes the
   counterexample" if it contains a constant that occurs in a counterexample shown
   during refinement but not in the task text (question, background, F0).

Usage:
  python -m aspire.eval.errors --suite SUITE --output OUT RUN_DIR [RUN_DIR ...]
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from aspire.common import read_jsonl, write_json
from aspire.data.suite import load_suite
from aspire.families import difficulty

CATEGORIES = ("no_program", "syntax", "timeout", "interface", "unsat_mismatch", "multi_model",
              "over", "under", "mixed")
INDICATORS = ("fewer_comparisons", "fewer_negations", "undefined_body_pred")
_ATOM = re.compile(r"\b([a-z][A-Za-z0-9_]*)\s*\(")
_FACT_PRED = re.compile(r"(?:^|[.\s])([a-z][A-Za-z0-9_]*)\s*(?:\(|\.)")


def _contexts(row):
    return [row["instance"]] + list(row["contexts"])


def _phases(row):
    return ["feasible", "optimal"] if row.get("optimization") else ["feasible"]


def program_shape(program):
    """Head signatures, body predicate names, #comparisons, #negated literals (clingo AST)."""
    import clingo.ast as ast
    from aspire.training.asp_ast import parse
    heads, body_preds, comparisons, negations = set(), set(), 0, 0

    def atoms_of(node, out):
        if node.ast_type == ast.ASTType.SymbolicAtom:
            sym = node.symbol
            if sym.ast_type == ast.ASTType.Function:
                out.add((sym.name, len(sym.arguments)))
            elif sym.ast_type == ast.ASTType.SymbolicTerm and sym.symbol.type.name == "Function":
                out.add((sym.symbol.name, len(sym.symbol.arguments)))
        for key in node.child_keys:
            child = getattr(node, key)
            if isinstance(child, ast.AST):
                atoms_of(child, out)
            elif child is not None and not isinstance(child, (str, int)):
                try:
                    for item in child:
                        if isinstance(item, ast.AST):
                            atoms_of(item, out)
                except TypeError:
                    pass

    for node in parse(program):
        if node.ast_type == ast.ASTType.Rule:
            atoms_of(node.head, heads)
            for lit in node.body:
                if lit.ast_type == ast.ASTType.Literal:
                    if lit.atom.ast_type == ast.ASTType.Comparison:
                        comparisons += 1
                    elif lit.sign != ast.Sign.NoSign:
                        negations += 1
                found = set()
                atoms_of(lit, found)
                body_preds |= {n for n, _ in found}
        elif node.ast_type == ast.ASTType.Minimize:
            found = set()
            for lit in node.body:
                atoms_of(lit, found)
            body_preds |= {n for n, _ in found}
    return heads, body_preds, comparisons, negations


def classify(row, public, gold):
    """Outcome category and structural indicators of one failing sample."""
    program = row["candidate_program"]
    if not program.strip():
        return "no_program", set()
    ctx = _contexts(row)
    phases = _phases(row)
    results = [c[f"candidate_{p}"] for c in ctx for p in phases]
    if any(not r["parse_ground"] and not r.get("timed_out") for r in results):
        return "syntax", set()
    if any(r.get("timed_out") or (r["parse_ground"] and not r["terminated"]) for r in results):
        return "timeout", set()
    try:
        heads, body_preds, comparisons, negations = program_shape(program)
        g_heads, _, g_comparisons, g_negations = program_shape(gold)
    except (RuntimeError, ValueError):
        return "syntax", set()
    indicators = set()
    if comparisons < g_comparisons:
        indicators.add("fewer_comparisons")
    if negations < g_negations:
        indicators.add("fewer_negations")
    defined = {n for n, _ in heads}
    defined |= {n for n, _ in program_shape(public["background_asp"])[0]} if public["background_asp"].strip() else set()
    facts = " ".join([public["facts"]] + [c["facts"] for c in ctx])
    defined |= set(_FACT_PRED.findall(facts)) | set(_ATOM.findall(facts))
    if body_preds - defined:
        indicators.add("undefined_body_pred")
    targets = {(n, int(a)) for n, a in public["targets"]}
    if not heads & targets:
        return "interface", indicators
    spurious = missing = False
    for c in ctx:
        if c["correct"]:
            continue
        for p in phases:
            g, k = c[f"gold_{p}"], c[f"candidate_{p}"]
            if g["satisfiable"] != k["satisfiable"]:
                return "unsat_mismatch", indicators
            if len(g["models"]) != len(k["models"]):
                return "multi_model", indicators
            gs, ks = {frozenset(m) for m in g["models"]}, {frozenset(m) for m in k["models"]}
            if gs == ks:
                continue
            g_atoms, k_atoms = set().union(*gs), set().union(*ks)
            spurious |= bool(k_atoms - g_atoms)
            missing |= bool(g_atoms - k_atoms)
            if not (k_atoms - g_atoms or g_atoms - k_atoms):  # same atoms, grouped differently
                return "multi_model", indicators
    return ("mixed" if spurious and missing else "over" if spurious else "under"), indicators


_CONST = re.compile(r"\"[^\"]*\"|-?\b\d+\b|\b[a-z][A-Za-z0-9_]*\b(?!\s*\()")


def program_constants(program):
    """Constants used as terms (numbers, lowercase symbols, strings); predicate names excluded."""
    body = re.sub(r"%.*", "", program)
    out = set()
    for m in _CONST.finditer(body):
        tok = m.group(0)
        if tok in ("not",):
            continue
        out.add(tok)
    return out


def hardcoded(row, public, gold):
    """Constants introduced during refinement that come from a shown counterexample.

    A constant counts if it is in the final program but not in the initial
    candidate, not in the reference program (legitimate constants such as
    weights or labels) and not in the task text (question, background, F0), and
    it occurs in a counterexample shown during refinement.
    """
    shown = [t["diagnosis"]["facts"] for t in row.get("refinement_trace", []) if t.get("diagnosis")]
    if not shown:
        return set()
    task_text = " ".join([public["question"], public["background_asp"], public["facts"]])
    before = program_constants(row["initial_candidate_program"]) | program_constants(gold)
    leaked = set()
    for const in program_constants(row["candidate_program"]) - before:
        pattern = r"(?<![\w\"])" + re.escape(const) + r"(?![\w\"])"
        if re.search(pattern, task_text):
            continue
        if any(re.search(pattern, facts) for facts in shown):
            leaked.add(const)
    return leaked


def analyze(suite, runs, split="iid"):
    _, public_rows, private = load_suite(suite, split)
    public = {r["sample_id"]: r for r in public_rows}
    report = {}
    for run in runs:
        rows = read_jsonl(Path(run) / "protocol_samples.jsonl")
        cats = defaultdict(Counter)
        inds = defaultdict(Counter)
        hard = {"refined_changed": 0, "hardcoded": 0, "of_samples": 0, "of_hardcoded": 0, "examples": []}
        for row in rows:
            sid = row["sample_id"]
            group = difficulty(row["task"])
            if not row["end_to_end_correct"]:
                cat, ind = classify(row, public[sid], private[sid]["gold_program"])
                for g in (group, "overall"):
                    cats[g][cat] += 1
                    inds[g].update(ind)
            if row.get("refinement_trace") and row["candidate_program"] != row["initial_candidate_program"]:
                hard["refined_changed"] += 1
                leaked = hardcoded(row, public[sid], private[sid]["gold_program"])
                hard["hardcoded"] += bool(leaked)
                visible_pass = row["visible_score"] == row["visible_total"]
                if visible_pass and not row["end_to_end_correct"]:
                    hard["of_samples"] += 1
                    hard["of_hardcoded"] += bool(leaked)
                if leaked and len(hard["examples"]) < 5:
                    hard["examples"].append({"sample_id": sid, "task": row["task"], "constants": sorted(leaked),
                                             "program": row["candidate_program"]})
        report[str(run)] = {"failures": {g: dict(c) for g, c in cats.items()},
                            "indicators": {g: dict(c) for g, c in inds.items()},
                            "hardcoding": hard}
    return report


def markdown(report):
    lines = ["# Failure taxonomy", ""]
    head = "| run | group | n_fail | " + " | ".join(CATEGORIES) + " | " + " | ".join(INDICATORS) + " |"
    lines += [head, "|" + "---|" * (3 + len(CATEGORIES) + len(INDICATORS))]
    for run, r in report.items():
        for g in ("easy", "mid", "hard", "overall"):
            c, i = r["failures"].get(g, {}), r["indicators"].get(g, {})
            n = sum(c.values())
            lines.append(f"| {run} | {g} | {n} | "
                         + " | ".join(str(c.get(k, 0)) for k in CATEGORIES) + " | "
                         + " | ".join(str(i.get(k, 0)) for k in INDICATORS) + " |")
    lines += ["", "# Counterexample hard-coding (refinement runs)", "",
              "| run | changed programs | with leaked constants | OF samples | OF with leaked constants |",
              "|---|---|---|---|---|"]
    for run, r in report.items():
        h = r["hardcoding"]
        if h["refined_changed"]:
            lines.append(f"| {run} | {h['refined_changed']} | {h['hardcoded']} | {h['of_samples']} | {h['of_hardcoded']} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", required=True)
    p.add_argument("--split", default="iid")
    p.add_argument("--output", required=True)
    p.add_argument("runs", nargs="+", help="completed run directories")
    a = p.parse_args(argv)
    report = analyze(a.suite, a.runs, a.split)
    out = Path(a.output)
    write_json(out / "errors.json", report)
    (out / "errors.md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))


if __name__ == "__main__":
    main()
