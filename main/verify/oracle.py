"""Bounded multi-context verification (Sec. 3.3) and the visible-only oracle (Sec. 3.4).

Every comparison projects all stable models of B ∪ P ∪ F onto the declared
output interface O and compares the deduplicated projection sets with the
expected output of that context. Optimization tasks compare both the feasible
phase (weak constraints ignored) and the optimal phase (all optimal models).

Expected outputs are frozen per context by aspire.data.freeze (Sec. 3.1): the
satisfiability and the feasible/optimal projections. Verification therefore
needs only (facts, expected output) pairs, such as contexts labelled by domain
experts; the reference program is not required at run time.

`VisibleOracle` sees the instance F0 plus the public feedback contexts only.
Held-out scoring contexts are reachable exclusively through `evaluate`, which
the runner calls after all inference decisions are final.
"""
import multiprocessing as mp
import re
import time

from aspire.common import digest
from aspire.verify.solver import _atom_delta, _format_projection, _phase_match, solve_projected

FORBIDDEN = re.compile(r"#\s*(?:include|script)\b", re.I)


def _solve(program, facts, targets, phase):
    if FORBIDDEN.search(program + "\n" + facts):
        return {"parse_ground": False, "terminated": False, "satisfiable": False,
                "models": [], "costs": [], "error": "#include/#script are not permitted"}
    return solve_projected(program, facts, targets, phase)


class Solver:
    """Clingo in a spawned worker so that a hard timeout can kill grounding/search."""

    def __init__(self, timeout=20):
        self.timeout = timeout
        self.pool = None
        self.cache = {}
        self.calls = 0
        self.timeouts = 0
        self.seconds = 0.0

    def close(self):
        if self.pool:
            self.pool.terminate()
            self.pool.join()
            self.pool = None

    def solve(self, program, facts, targets, phase="feasible"):
        key = digest([program, facts, targets, phase])
        if key in self.cache:
            return self.cache[key]
        if self.pool is None:
            self.pool = mp.get_context("spawn").Pool(1)
        started = time.perf_counter()
        self.calls += 1
        try:
            value = self.pool.apply_async(_solve, (program, facts, targets, phase)).get(self.timeout)
        except mp.TimeoutError:
            self.timeouts += 1
            self.close()
            value = {"parse_ground": False, "terminated": False, "satisfiable": False,
                     "models": [], "costs": [], "error": "solver timeout", "timed_out": True}
        except Exception:
            self.close()
            raise
        self.seconds += time.perf_counter() - started
        self.cache[key] = value
        return value

    def measure(self):
        return {"solver_calls": self.calls, "solver_timeouts": self.timeouts,
                "solver_seconds": self.seconds}


def _unfinished(result):
    """Search did not complete within budget (distinct from a parse/grounding error)."""
    return bool(result.get("timed_out")) or (result["parse_ground"] and not result["terminated"])


def full_program(public, program):
    return public["background_asp"] + "\n" + program


def phases(public):
    return ["feasible", "optimal"] if public["optimization"] else ["feasible"]


def visible_contexts(public):
    """C_vis = {F0} ∪ C_fb, in traversal order."""
    instance = {"name": "instance", "facts": public["facts"]}
    if "instance_expected" in public:
        instance["expected"] = public["instance_expected"]
    return [instance] + public["feedback_contexts"]


def expected_record(result):
    """The part of a reference solve result that is frozen as the expected output."""
    return {"satisfiable": result["satisfiable"], "models": result["models"], "costs": result["costs"]}


def _expected(solver, public, gold, context, phase):
    """Expected output Π(P*, F) for one phase.

    Frozen suites (schema 2) store it per context, so verification needs no
    reference source; `gold` is only a fallback for legacy suites.
    """
    if "expected" in context:
        e = context["expected"][phase]
        return {"phase": phase, "parse_ground": True, "terminated": True, "error": None,
                "satisfiable": e["satisfiable"], "models": e["models"], "costs": e["costs"]}
    if gold is None:
        raise ValueError(f"context {context['name']} has no expected output and no reference was given")
    g = solver.solve(full_program(public, gold), context["facts"], public["targets"], phase)
    if not g["parse_ground"] or not g["terminated"]:
        raise RuntimeError(f"reference failure in {public['sample_id']} / {context['name']}: {g['error']}")
    return g


def compare_context(solver, public, gold, candidate, context):
    """V(P, F): 1 iff P parses, grounds, terminates and matches on every phase."""
    result = {"name": context["name"], "facts": context["facts"]}
    for phase in phases(public):
        g = _expected(solver, public, gold, context, phase)
        c = solver.solve(full_program(public, candidate), context["facts"], public["targets"], phase)
        matched, cost = _phase_match(g, c, compare_cost=(phase == "optimal"))
        result.update({f"gold_{phase}": g, f"candidate_{phase}": c, f"{phase}_correct": matched})
        if phase == "optimal":
            result["optimal_cost_correct"] = cost
    result["timed_out"] = any(_unfinished(result[f"candidate_{p}"]) for p in phases(public))
    result["correct"] = bool(candidate.strip()) and all(result[f"{p}_correct"] for p in phases(public))
    return result


def evaluate(solver, public, private, candidate):
    """Strict pass R(P) (Eq. 4) over F0 and all held-out contexts, plus breakdowns."""
    gold = private.get("gold_program")
    instance = compare_context(solver, public, gold, candidate, visible_contexts(public)[0])
    held = [compare_context(solver, public, gold, candidate, c)
            for c in private["scoring_contexts"]]
    status = instance["candidate_feasible"]
    return {"instance": instance, "contexts": held,
            "instance_correct": instance["correct"],
            "context_correct": all(c["correct"] for c in held),
            "end_to_end_correct": instance["correct"] and all(c["correct"] for c in held),
            "extraction": bool(candidate.strip()), "optimization": public["optimization"],
            "parse_ground": status["parse_ground"], "terminated": status["terminated"],
            "satisfiable": status["satisfiable"], "feasible_correct": instance["feasible_correct"],
            "optimal_correct": instance.get("optimal_correct"),
            "optimal_cost_correct": instance.get("optimal_cost_correct"),
            "timed_out": instance["timed_out"] or any(c["timed_out"] for c in held)}


class VisibleOracle:
    """Receives PUBLIC contexts with their frozen expected outputs only.

    `gold` may be None (schema-2 suites); it is used only for legacy suites
    whose contexts carry no expected output.
    visible="all" uses C_vis = {F0} ∪ C_fb; visible="instance" uses {F0} only
    (the single-context ablation of Table 5).
    """

    def __init__(self, solver, public, gold=None, visible="all"):
        if visible not in ("all", "instance"):
            raise ValueError(visible)
        self.solver, self.public, self.gold = solver, public, gold
        ctx = visible_contexts(public)
        self.contexts = ctx if visible == "all" else ctx[:1]

    def check(self, candidate):
        return [compare_context(self.solver, self.public, self.gold, candidate, c)
                for c in self.contexts]

    def score(self, candidate):
        """s(P) (Eq. 5): number of visible contexts passed."""
        return sum(c["correct"] for c in self.check(candidate))

    def diagnosis(self, candidate, selection="shortest", rng=None):
        """Diagnose one failing visible context, or return None if all pass.

        selection="shortest" picks the failing context with the fewest fact
        characters, ties broken by traversal order (Sec. 3.4).
        """
        checks = self.check(candidate)
        failed = [c for c in checks if not c["correct"]]
        if not failed:
            return None
        if selection == "random":
            check = rng.choice(failed)
        elif selection == "first":
            check = failed[0]
        elif selection == "shortest":
            check = min(failed, key=lambda c: len(c["facts"]))  # min() keeps the earliest tie
        else:
            raise ValueError(f"unknown selection: {selection}")
        phase = "feasible" if not check["feasible_correct"] or not candidate.strip() else "optimal"
        g, c = check[f"gold_{phase}"], check[f"candidate_{phase}"]
        kind = ("empty_program" if not candidate.strip()
                else "timeout" if _unfinished(c)
                else "parse_error" if not c["parse_ground"]
                else "satisfiability_mismatch" if g["satisfiable"] != c["satisfiable"]
                else "projection_mismatch")
        spurious, missing = _atom_delta(g, c)
        return {"context_name": check["name"], "facts": check["facts"], "phase": phase,
                "optimization": self.public["optimization"], "kind": kind,
                "expected_unsat": not g["satisfiable"], "actual_unsat": c["parse_ground"] and c["terminated"] and not c["satisfiable"],
                "expected": _format_projection(g, limit=len(g["models"]) or 1),
                "actual": _format_projection(c, limit=len(c["models"]) or 1),
                "expected_models": g["models"], "actual_models": c["models"],
                "spurious": spurious, "missing": missing, "error": c.get("error") or "",
                "visible_score": sum(x["correct"] for x in checks), "visible_total": len(checks)}

    def observe(self, candidate):
        """Candidate outputs on visible contexts WITHOUT any reference information."""
        out = []
        for ctx in self.contexts:
            for phase in phases(self.public):
                value = self.solver.solve(full_program(self.public, candidate), ctx["facts"],
                                          self.public["targets"], phase)
                out.append({"context": ctx["name"], "facts": ctx["facts"], "phase": phase, **value})
        return out
