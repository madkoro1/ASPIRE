"""Judge accuracy of the projection verifier (Sec. 3.5 (2), Table 4; the detection curve is supplementary). CPU, clingo only.

From reference programs of the frozen suite, two sets of programs are built:

  positives  equivalence-preserving rewrites (equivalent by construction):
             auxiliary-predicate renaming, rule reordering, body reordering,
             rule splitting through a fresh auxiliary predicate, flipped
             comparisons, and weak constraint -> #minimize
  negatives  mutations: delete a body literal, toggle default negation, relax or
             shift a comparison, rebind a variable, swap the arguments of a
             binary atom, delete a rule, drop weak constraints, hard-code the
             optimum of F0

Ground truth does not come from the judges under test. A variant is labelled
non-equivalent iff the standard projected two-phase comparison differs from the
reference on at least one of `--probes` random probe contexts drawn from the
per-argument value pools of the sample's contexts. Probes are used ONLY for
labels and never by a judge, so labels are independent of every judge (including
the paper's, which reads F0 and the held-out contexts). Positives are equivalent
by construction; positives that differ anywhere (a rewrite bug) and negatives that
differ on no probe are dropped and counted. Mutants that do not parse or
ground on F0 are excluded (they are syntax errors, caught by any executing judge).

Judges: exact text, normalized AST, projection on F0 only, single answer set
(first model; last improving model for optimization), whole answer sets without
projection, optimal phase only, and the paper's judge on F0 ∪ held-out. The
detection curve evaluates the paper's judge on F0 plus the first n extended
contexts.

Usage:
  python -m aspire.eval.verifier_bench --suite SUITE --output OUT [--per-family 10] [--probes 50]
"""
import argparse
from collections import Counter, defaultdict
import random
import re

from aspire.common import digest, write_json
from aspire.data.suite import load_suite
from aspire.training.asp_ast import normalized_hash, parse
from aspire.verify.oracle import Solver
from aspire.verify.solver import _phase_match

FLIP = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "=": "=", "==": "==", "!=": "!="}
RELAX = {"<": "<=", "<=": "<", ">": ">=", ">=": ">"}
_CMP = re.compile(r"^\s*(.+?)\s*(<=|>=|!=|==|<|>|=)\s*(.+?)\s*$")
_VAR = re.compile(r"(?<![\w\"])[A-Z][A-Za-z0-9_]*")
_WEAK = re.compile(r"^:~\s*(.*?)\s*\.\s*\[\s*(.*?)\s*\]\s*$", re.S)


# ----------------------------------------------------------------- AST helpers
def _offsets(source):
    lines = source.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))

    def offset(pos):
        if pos.line == len(lines) + 1 and pos.column == 1:
            return len(source)
        raw = lines[pos.line - 1].encode("utf-8")
        return starts[pos.line - 1] + len(raw[:pos.column - 1].decode("utf-8"))
    return offset


def statements(source):
    """[(kind, text, rule_info)] in order; rule_info has head text and body literal texts."""
    import clingo.ast as ast
    off = _offsets(source)
    out = []
    for node in parse(source):
        a, b = off(node.location.begin), off(node.location.end)
        text = source[a:b].strip()
        if node.ast_type == ast.ASTType.Rule:
            constraint = str(node.head) == "#false"
            head = "" if constraint else source[off(node.head.location.begin):off(node.head.location.end)]
            body, simple = [], True
            for lit in node.body:
                if lit.ast_type != ast.ASTType.Literal:
                    simple = False
                    break
                body.append({"text": source[off(lit.location.begin):off(lit.location.end)],
                             "negated": lit.sign != ast.Sign.NoSign,
                             "comparison": lit.atom.ast_type == ast.ASTType.Comparison})
            out.append(("rule", text, {"head": head, "body": body, "simple": simple, "constraint": constraint}))
        elif node.ast_type == ast.ASTType.Minimize:
            out.append(("weak", text, None))
        else:
            out.append(("other", text, None))
    return out


def render_rule(head, body):
    if not body:
        return head + "."
    return (head + " :- " if head else ":- ") + ", ".join(body) + "."


def assemble(stmts):
    return "\n".join(stmts)


def head_predicates(source):
    """Names of predicates defined in rule heads (regex on head text)."""
    names = set()
    for kind, _, info in statements(source):
        if kind == "rule" and info["head"]:
            names |= set(re.findall(r"\b([a-z][A-Za-z0-9_]*)\s*\(", info["head"]))
            names |= {t for t in re.findall(r"\b([a-z][A-Za-z0-9_]*)\b(?!\s*\()", info["head"])}
    return names


# ------------------------------------------------------------------ variants
def positives(gold, targets):
    st = statements(gold)
    texts = [t for _, t, _ in st]
    out = []
    aux = sorted(head_predicates(gold) - {n for n, _ in targets})
    if aux:
        renamed = gold
        for name in aux:
            renamed = re.sub(r"\b" + re.escape(name) + r"\b(?=\s*\()", name + "_r", renamed)
        out.append(("rename_aux", renamed))
    if len(texts) > 1:
        out.append(("reorder_rules", assemble(texts[::-1])))
    for i, (kind, _, info) in enumerate(st):
        if kind != "rule" or not info["simple"]:
            continue
        lits = [b["text"] for b in info["body"]]
        if len(lits) >= 2:
            new = texts[:i] + [render_rule(info["head"], lits[::-1])] + texts[i + 1:]
            out.append(("reorder_body", assemble(new)))
        if len(lits) >= 2:
            vars_ = sorted(set(_VAR.findall(info["head"]))) if info["head"] else []
            aux_atom = f"aux_split_{i}" + (f"({','.join(vars_)})" if vars_ else "")
            split = [render_rule(aux_atom, lits), render_rule(info["head"], [aux_atom])]
            out.append(("split_rule", assemble(texts[:i] + split + texts[i + 1:])))
        for j, b in enumerate(info["body"]):
            m = _CMP.match(b["text"]) if b["comparison"] and not b["negated"] else None
            if m:
                body = list(lits)
                body[j] = f"{m.group(3)}{FLIP[m.group(2)]}{m.group(1)}"
                out.append(("flip_comparison", assemble(texts[:i] + [render_rule(info["head"], body)] + texts[i + 1:])))
    weak = [i for i, (k, _, _) in enumerate(st) if k == "weak"]
    if weak:
        new = list(texts)
        ok = True
        for i in weak:
            m = _WEAK.match(texts[i])
            if not m:
                ok = False
                break
            new[i] = "#minimize{" + m.group(2) + ": " + m.group(1) + "}."
        if ok:
            out.append(("weak_to_minimize", assemble(new)))
    return out


def negatives(gold, public):
    st = statements(gold)
    texts = [t for _, t, _ in st]
    out = []
    for i, (kind, _, info) in enumerate(st):
        if kind != "rule" or not info["simple"]:
            continue
        lits = [b["text"] for b in info["body"]]
        swap = lambda body: assemble(texts[:i] + [render_rule(info["head"], body)] + texts[i + 1:])
        if len(lits) >= 2:
            for j in range(len(lits)):
                out.append(("delete_literal", swap(lits[:j] + lits[j + 1:])))
        for j, b in enumerate(info["body"]):
            body = list(lits)
            if b["comparison"]:
                m = _CMP.match(b["text"])
                if m and m.group(2) in RELAX:
                    body[j] = f"{m.group(1)}{RELAX[m.group(2)]}{m.group(3)}"
                    out.append(("relax_comparison", swap(body)))
                if m and re.fullmatch(r"-?\d+", m.group(3)):
                    body = list(lits)
                    body[j] = f"{m.group(1)}{m.group(2)}{int(m.group(3)) + 1}"
                    out.append(("shift_threshold", swap(body)))
                continue
            body = list(lits)
            body[j] = re.sub(r"^not\s+", "", b["text"]) if b["negated"] else "not " + b["text"]
            out.append(("toggle_negation", swap(body)))
            rule_vars = sorted(set(_VAR.findall(texts[i])))
            lit_vars = list(dict.fromkeys(_VAR.findall(b["text"])))
            for v in lit_vars:
                for w in rule_vars:
                    if w != v:
                        body = list(lits)
                        body[j] = re.sub(r"(?<![\w\"])" + v + r"(?![\w])", w, b["text"], count=1)
                        out.append(("rebind_variable", swap(body)))
                        break
                break
            args = re.match(r"^(not\s+)?([a-z][A-Za-z0-9_]*)\(\s*([^,()]+?)\s*,\s*([^,()]+?)\s*\)$", b["text"])
            if args and args.group(3) != args.group(4):
                body = list(lits)
                body[j] = f"{args.group(1) or ''}{args.group(2)}({args.group(4)},{args.group(3)})"
                out.append(("swap_arguments", swap(body)))
    if len([k for k, _, _ in st if k == "rule"]) > 1:
        for i, (kind, _, _) in enumerate(st):
            if kind == "rule":
                out.append(("delete_rule", assemble(texts[:i] + texts[i + 1:])))
    if any(k == "weak" for k, _, _ in st):
        out.append(("drop_weak", assemble([t for (k, t, _) in st if k != "weak"])))
        opt = public.get("instance_expected", {}).get("optimal", {}).get("models") or []
        if opt:
            out.append(("hardcode_optimum", " ".join(a + "." for a in opt[0]) or "% empty optimum"))
    return out


# ------------------------------------------------------------------- probing
def ground_facts(facts):
    """Ground atoms of a fact string (ranges and pools expanded by clingo)."""
    from clingo.control import Control
    ctl = Control(["--warn=none"])
    ctl.add("base", [], facts)
    ctl.ground([("base", [])])
    atoms = []
    ctl.solve(on_model=lambda m: atoms.extend(m.symbols(atoms=True)))
    return atoms


def probe_contexts(contexts, n, rng, program=""):
    """Random fact sets over the predicates and per-argument values of the given contexts.

    Numeric argument pools are widened by ±1 around observed values and around the
    numbers of `program` (thresholds), so that boundary cases are probed.
    """
    thresholds = {int(x) for x in re.findall(r"(?<![\w])-?\d+(?![\w])", re.sub(r"%.*", "", program))}
    pools, counts, zero = defaultdict(set), Counter(), set()
    for facts in contexts:
        local = Counter()
        for atom in ground_facts(facts):
            sig = (atom.name, len(atom.arguments))
            if not atom.arguments:
                zero.add(atom.name)
                continue
            local[sig] += 1
            for k, arg in enumerate(atom.arguments):
                pools[sig + (k,)].add(str(arg))
                if arg.type.name == "Number":
                    pools[sig + (k,)] |= {str(arg.number - 1), str(arg.number + 1)}
                    pools[sig + (k,)] |= {str(t + d) for t in thresholds for d in (-1, 0, 1)}
        for sig, c in local.items():
            counts[sig] = max(counts[sig], c)
    probes = []
    for _ in range(n):
        atoms = set()
        for sig, most in sorted(counts.items()):
            for _ in range(rng.randint(0, most + 1)):
                args = [rng.choice(sorted(pools[sig + (k,)])) for k in range(sig[1])]
                atoms.add(f"{sig[0]}({','.join(args)}).")
        atoms |= {f"{z}." for z in zero if rng.random() < .5}
        probes.append(" ".join(sorted(atoms)))
    return probes


class Bench:
    def __init__(self, solver, public, gold):
        self.solver, self.public, self.gold = solver, public, gold
        self.phases = ["feasible", "optimal"] if public["optimization"] else ["feasible"]

    def full(self, program):
        return self.public["background_asp"] + "\n" + program

    def same(self, program, facts, project=True, phases=None):
        """True / False, or None when either side does not finish."""
        targets = self.public["targets"] if project else None
        for phase in phases or self.phases:
            g = self.solver.solve(self.full(self.gold), facts, targets, phase)
            c = self.solver.solve(self.full(program), facts, targets, phase)
            if not g["terminated"] or (c["parse_ground"] and not c["terminated"]) or c.get("timed_out"):
                return None
            if not _phase_match(g, c)[0]:
                return False
        return True

    def single(self, program, facts):
        """Compare one answer set each: the first model, or the last improving model if optimizing."""
        from clingo.control import Control

        def one(prog):
            args = ["--warn=none"] + (["--opt-mode=opt", "0"] if self.public["optimization"] else ["1"])
            ctl = Control(args)
            try:
                ctl.add("base", [], self.full(prog) + "\n" + facts)
                ctl.ground([("base", [])])
            except RuntimeError:
                return "error"
            found = []
            ctl.solve(on_model=lambda m: found.append(sorted(
                str(s) for s in m.symbols(atoms=True)
                if [s.name, len(s.arguments)] in [list(t) for t in self.public["targets"]])))
            return found[-1] if found else "unsat"
        return one(program) == one(self.gold)


def judge_metrics(t):
    """Binary-classification metrics of one judge; positive class = equivalent (correct) program.

    TP: equivalent program accepted   FN: equivalent program rejected (false rejection)
    TN: mutant rejected               FP: mutant accepted (false acceptance)
    sensitivity = recall = 1 - false rejection rate; specificity = 1 - false acceptance rate.
    """
    tp, fn = t["pos_accepted"], t["pos"] - t["pos_accepted"]
    fp, tn = t["neg_accepted"], t["neg"] - t["neg_accepted"]
    pct = lambda num, den: round(100 * num / den, 2) if den else None
    precision, recall = pct(tp, tp + fp), pct(tp, tp + fn)
    f1 = round(2 * precision * recall / (precision + recall), 2) if precision and recall else 0.0
    return {"positives": t["pos"], "negatives": t["neg"], "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "accuracy": pct(tp + tn, tp + tn + fp + fn), "recall": recall, "sensitivity": recall,
            "specificity": pct(tn, tn + fp), "precision": precision, "f1": f1,
            "false_rejection": pct(fn, tp + fn), "false_acceptance": pct(fp, fp + tn)}


def run(a):
    _, rows, private = load_suite(a.suite, a.split)
    by_family = defaultdict(list)
    for r in rows:
        by_family[r["task"]].append(r)
    chosen =[r for fam in sorted(by_family) for r in by_family[fam][:a.per_family]]
    solver = Solver(a.timeout)
    judges = ("exact_text", "normalized_ast", "f0_only", "single_model", "no_projection", "optimal_only", "paper")
    tally = {j: Counter() for j in judges}
    by_op = defaultdict(Counter)
    curve = defaultdict(Counter)
    dropped = Counter()
    try:
        for public in chosen:
            gold = private[public["sample_id"]]["gold_program"]
            bench = Bench(solver, public, gold)
            f0 = public["facts"]
            fb = [c["facts"] for c in public["feedback_contexts"]]
            ho = [c["facts"] for c in private[public["sample_id"]]["scoring_contexts"]]
            probes = probe_contexts([f0] + fb + ho, a.probes, random.Random(digest([a.seed, public["sample_id"]])),
                                    gold)
            # A probe identical to a suite context would let a judge's own facts into the labels.
            suite_keys = {frozenset(map(str, ground_facts(f))) for f in [f0] + fb + ho}
            probes = [p for p in probes if frozenset(map(str, ground_facts(p))) not in suite_keys]
            truth_set = [f0] + fb + ho + probes
            variants = [("pos", op, p) for op, p in positives(gold, public["targets"])]
            variants += [("neg", op, p) for op, p in negatives(gold, public)]
            per_op = Counter()
            for label, op, program in variants:
                if per_op[op] >= a.per_operator:
                    continue
                if label == "neg":
                    r0 = solver.solve(bench.full(program), f0, public["targets"], "feasible")
                    if not r0["parse_ground"]:
                        dropped["neg_syntax"] += 1
                        continue
                # verdicts: suite contexts (used by the judges); probe_verdicts: labels only.
                verdicts = [bench.same(program, f) for f in truth_set[:len(truth_set) - len(probes)]]
                probe_verdicts = [bench.same(program, f) for f in probes]
                if None in verdicts or None in probe_verdicts:
                    dropped[f"{label}_unfinished"] += 1
                    continue
                if label == "pos" and not (all(verdicts) and all(probe_verdicts)):
                    dropped["pos_differs (rewrite bug)"] += 1
                    continue
                # A negative is labelled non-equivalent from the probes alone, so the label
                # never depends on contexts that any judge uses; mutants exposed only by
                # suite contexts are dropped (conservative), not counted as equivalent.
                if label == "neg" and all(probe_verdicts):
                    dropped["neg_equivalent_on_probes"] += 1
                    continue
                per_op[op] += 1
                n_ctx = 1 + len(fb) + len(ho)
                accepts = {
                    "exact_text": program.strip() == gold.strip(),
                    "normalized_ast": normalized_hash(program) == normalized_hash(gold),
                    "f0_only": verdicts[0],
                    "single_model": all(bench.single(program, f) for f in [f0] + ho),
                    "no_projection": all(bench.same(program, f, project=False) for f in [f0] + ho),
                    "optimal_only": all(bench.same(program, f, phases=["optimal"] if public["optimization"] else None)
                                        for f in [f0] + ho),
                    "paper": verdicts[0] and all(verdicts[1 + len(fb):n_ctx]),
                }
                for j in judges:
                    tally[j][label] += 1
                    tally[j][f"{label}_accepted"] += bool(accepts[j])
                by_op[op]["n"] += 1
                by_op[op]["paper_accepted"] += bool(accepts["paper"])
                if label == "neg":
                    extended = verdicts[1:n_ctx]  # feedback then held-out, in suite order
                    for k in sorted({0, 1, 3, 6, len(extended)}):
                        curve[k]["neg"] += 1
                        curve[k]["detected"] += not (verdicts[0] and all(extended[:k]))
            print(f"{public['task']} {public['sample_id']}: {sum(per_op.values())} variants", flush=True)
    finally:
        solver.close()
    result = {"samples": len(chosen), "probes_per_sample": a.probes, "reference_programs": len(chosen), "label_source": "random probe contexts only (never used by a judge)", "dropped": dict(dropped),
              "positive_class": "equivalent program (should be accepted)",
              "judges": {j: judge_metrics(t) for j, t in tally.items()},
              "operators": {op: dict(c) for op, c in by_op.items()},
              "detection_curve": {str(k): round(100 * c["detected"] / c["neg"], 2) for k, c in sorted(curve.items())}}
    write_json(f"{a.output}/verifier_bench.json", result)
    for j, m in result["judges"].items():
        print(f"{j:16s} accuracy={m['accuracy']} precision={m['precision']} recall={m['recall']} specificity={m['specificity']} f1={m['f1']}")
    print(result["detection_curve"], result["dropped"])
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", required=True)
    p.add_argument("--split", default="iid")
    p.add_argument("--output", required=True)
    p.add_argument("--per-family", type=int, default=10, help="samples per family (50 = whole test set)")
    p.add_argument("--per-operator", type=int, default=3, help="variants kept per operator and sample")
    p.add_argument("--probes", type=int, default=50, help="random probe contexts per sample (ground truth only)")
    p.add_argument("--timeout", type=float, default=20)
    p.add_argument("--seed", type=int, default=56)
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
