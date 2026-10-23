"""Projected solve-and-check protocol (Sec. 2.4). Clingo only; no model imports.

This module holds the pure Clingo-based scoring logic used by the graded
solve-and-check benchmark.  It is deliberately free of any torch / transformers
imports so that the evaluator can be unit-tested on a CPU-only machine (the
model itself still runs in ``model_testing_extension.py`` on the GPU host).

The protocol, per (sample, context):

  * build the FULL gold and candidate programs (program + context facts) and
    solve both on small instances;
  * project every stable model onto the task's declared ``target_predicates``
    (name + arity) -- never the whole answer set, so a candidate using
    ``rank/2`` is not penalised for a gold that used ``prio/2``;
  * for NON-optimization tasks, compare the set of projected stable models;
  * for OPTIMIZATION tasks, compare BOTH the feasible projections
    (``--opt-mode=ignore``) AND the optimal projections (``--opt-mode=optN``),
    and require BOTH to agree.  Comparing optimal projections alone lets a
    degenerate program such as ``admit(1).`` pass by pinning the single optimal
    decision while never encoding the feasible choice space; the conjunction
    closes that hole.

UNSAT vs SAT-with-empty-projection are kept distinct:
    UNSAT             -> satisfiable=False, models=[]      (no stable model)
    SAT, empty proj   -> satisfiable=True,  models=[[]]    (one model, empty)
    SAT, non-empty    -> satisfiable=True,  models=[[...]]
so an intentionally-UNSAT reference is matched by an UNSAT candidate, while a
SAT-empty reference is never confused with UNSAT.
"""

import json
import re

from clingo.control import Control
from clingo.symbol import parse_term


class Context:
    # get features/words from a string of space separated words
    def gen_feature(self, x):
        ret = []
        for term in str(x.string).split(' '):
            ret.append(parse_term(term))
        return ret


def _is_missing_metadata(value):
    if value is None:
        return True
    try:
        import pandas as pd
        if isinstance(value, float) and bool(pd.isna(value)):
            return True
    except Exception:
        if isinstance(value, float) and value != value:  # NaN
            return True
    return isinstance(value, str) and not value.strip()


def parse_target_predicates(value):
    """Parse a CSV target signature into ``[(name, arity), ...]``.

    Accepted forms include JSON ``[["admit", 1]]``, dictionaries, and compact
    strings such as ``admit/1, cost/2``.  ``None`` means legacy full-model
    comparison.  Name *and* arity are retained to avoid conflating overloaded
    predicates.  Only predicates the task explicitly asks to observe belong
    here; a gold helper predicate must NOT be forced into the comparison.
    """
    if _is_missing_metadata(value):
        return None
    parsed = value
    if isinstance(value, str):
        stripped = value.strip()
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = [part.strip() for part in stripped.split(",") if part.strip()]
    if isinstance(parsed, dict):
        if "name" in parsed:
            parsed = [parsed]
        else:
            parsed = [{"name": name, "arity": arity} for name, arity in parsed.items()]
    if isinstance(parsed, (str, tuple)):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise ValueError(f"invalid target_predicates metadata: {value!r}")

    targets = []
    for item in parsed:
        if isinstance(item, dict):
            name, arity = item.get("name"), item.get("arity")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            name, arity = item
        elif isinstance(item, str):
            match = re.fullmatch(r"\s*([a-z][A-Za-z0-9_]*)\s*[/\:]\s*(\d+)\s*", item)
            if not match:
                raise ValueError(
                    f"target predicate {item!r} must include arity, e.g. admit/1")
            name, arity = match.group(1), match.group(2)
        else:
            raise ValueError(f"invalid target predicate entry: {item!r}")
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][A-Za-z0-9_]*", name):
            raise ValueError(f"invalid target predicate name: {name!r}")
        try:
            arity = int(arity)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid arity for target predicate {name!r}: {arity!r}") from exc
        if arity < 0:
            raise ValueError(f"negative arity for target predicate {name!r}")
        targets.append((name, arity))
    return list(dict.fromkeys(targets))


def parse_optimization_flag(value, gold_program=""):
    """Resolve the has-optimization flag.

    Explicit per-sample metadata is authoritative; ASP syntax detection is only
    a fallback for legacy rows that carry no flag.
    """
    if _is_missing_metadata(value):
        return has_optimization_statement(gold_program)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "optimal", "optimization", "opt", "both"}:
        return True
    if normalized in {"0", "false", "no", "n", "none", "feasible", "ignore"}:
        return False
    raise ValueError(f"invalid optimization metadata: {value!r}")


def parse_context_family(value):
    """Parse optional counterfactual contexts from a CSV cell.

    A context is a fact string or ``{"id"/"name": ..., "facts": ...}``.  Plain
    contexts replace the instance facts; a dict with ``"append": true`` appends
    its facts to the instance instead.  The dataset stores its 10-20 hidden
    contexts as a JSON list of ``{"id", "facts", "tags"}`` objects.
    """
    if _is_missing_metadata(value):
        return []
    parsed = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
    if isinstance(parsed, dict) and "contexts" in parsed:
        parsed = parsed["contexts"]
    elif isinstance(parsed, dict) and "facts" not in parsed:
        parsed = [{"name": name, "facts": facts} for name, facts in parsed.items()]
    if isinstance(parsed, (str, dict)):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise ValueError(f"invalid context metadata: {value!r}")

    contexts = []
    for index, item in enumerate(parsed):
        if isinstance(item, str):
            contexts.append({"name": f"context_{index}", "facts": item, "append": False})
            continue
        if not isinstance(item, dict) or "facts" not in item:
            raise ValueError(f"invalid context entry: {item!r}")
        # The dataset labels contexts with "id"; keep it as the display name so
        # per-context diagnostics stay legible (falls back to a running index).
        name = item.get("name", item.get("id", f"context_{index}"))
        contexts.append({
            "name": str(name),
            "facts": str(item["facts"]),
            "append": bool(item.get("append", False)),
        })
    return contexts


def has_optimization_statement(program):
    return bool(re.search(r":~|#\s*(?:minimize|maximize)\b", program or "", re.IGNORECASE))


def _canonical_projected_models(models):
    unique = {tuple(sorted(model)) for model in models}
    return [list(model) for model in sorted(unique)]


def _project_model(model, target_predicates):
    if target_predicates is None:
        return [str(atom) for atom in model.symbols(atoms=True)]
    targets = set((str(name), int(arity)) for name, arity in target_predicates)
    projected = []
    for atom in model.symbols(atoms=True):
        try:
            signature = (atom.name, len(atom.arguments))
        except RuntimeError:
            continue
        if signature in targets:
            projected.append(str(atom))
    return projected


def solve_projected(program, facts, target_predicates=None, phase="feasible"):
    """Solve one small instance and return projection plus explicit statuses.

    ``phase="feasible"`` ignores all optimization statements and enumerates
    every feasible stable model.  ``phase="optimal"`` uses ``optN`` and retains
    every optimal model; if a candidate omitted all optimization statements,
    all of its feasible models are (correctly) treated as equally optimal.
    """
    if phase not in {"feasible", "optimal"}:
        raise ValueError(f"unknown solve phase: {phase}")
    opt_mode = "ignore" if phase == "feasible" else "optN"
    messages = []
    control = Control(
        ["0", "--warn=none", f"--opt-mode={opt_mode}", "-t", "1"],
        logger=lambda _code, message: messages.append(message),
        message_limit=20,
    )
    result = {
        "phase": phase,
        "parse_ground": False,
        "terminated": False,
        "satisfiable": False,
        "models": [],
        "costs": [],
        "error": None,
    }
    combined = f"{program or ''}\n{facts or ''}"
    try:
        control.add("base", [], combined)
        control.ground([("base", [])], context=Context())
        result["parse_ground"] = True
    except Exception as exc:
        detail = "\n".join(messages).strip()
        result["error"] = detail or str(exc)
        return result

    models = []
    costs = []
    require_proven = phase == "optimal" and has_optimization_statement(program)

    def on_model(model):
        if require_proven and not model.optimality_proven:
            return
        models.append(_project_model(model, target_predicates))
        costs.append(tuple(model.cost))

    try:
        solve_result = control.solve(on_model=on_model)
        result["terminated"] = bool(solve_result.exhausted) and not bool(solve_result.unknown)
        result["satisfiable"] = bool(solve_result.satisfiable)
    except Exception as exc:
        detail = "\n".join(messages).strip()
        result["error"] = detail or str(exc)
        return result

    result["models"] = _canonical_projected_models(models)
    result["costs"] = [list(cost) for cost in sorted(set(costs))]
    return result


def _phase_match(gold_result, candidate_result, compare_cost=False):
    """Return (model_match, cost_match) for one solved phase.

    A comparison is only meaningful when BOTH programs parsed, grounded, and the
    search terminated exhaustively.  Satisfiability is NOT part of readiness: an
    intentionally-UNSAT reference must be matchable by an UNSAT candidate.  The
    match key pairs (satisfiable, models) so that UNSAT (False, []), SAT-empty
    (True, [[]]), and SAT-non-empty are all distinguished, and both-UNSAT is a
    correct match.
    """
    executable = (
        gold_result["parse_ground"] and candidate_result["parse_ground"]
        and gold_result["terminated"] and candidate_result["terminated"]
    )
    gold_key = (gold_result["satisfiable"], gold_result["models"])
    candidate_key = (candidate_result["satisfiable"], candidate_result["models"])
    model_match = bool(executable and gold_key == candidate_key)
    cost_match = None
    if compare_cost:
        gold_cost_key = (gold_result["satisfiable"], gold_result["costs"])
        candidate_cost_key = (candidate_result["satisfiable"], candidate_result["costs"])
        cost_match = bool(executable and gold_cost_key == candidate_cost_key)
    return model_match, cost_match


def split_refinement_contexts(contexts, n_feedback):
    """Split contexts into (feedback, scoring); always keep >=1 for scoring.

    The first n_feedback contexts in generator order become feedback contexts.
    The split is deterministic and not stratified by context tag.
    """
    contexts = list(contexts or [])
    if n_feedback <= 0 or len(contexts) <= 1:
        return [], contexts
    n_feedback = min(n_feedback, len(contexts) - 1)
    return contexts[:n_feedback], contexts[n_feedback:]


def _format_projection(result, limit=4):
    """Render a solve result's projection compactly for prompt feedback."""
    if not result["parse_ground"]:
        return "does not parse"
    if not result["terminated"]:
        return "did not finish"
    if not result["satisfiable"]:
        return "no answer set"
    models = result["models"]
    if models == [[]]:
        return "answer set with none of these atoms"
    shown = ["{" + ", ".join(m) + "}" if m else "{}" for m in models[:limit]]
    more = "" if len(models) <= limit else f" (+{len(models) - limit} more)"
    return ", ".join(shown) + more



def _atom_delta(gold, cand):
    """When both sides have exactly one model, return (spurious, missing) atom
    lists; otherwise (None, None).  This isolates the violating atoms, which is
    the "property-oriented" signal (point at the root cause) as opposed to
    dumping both full projection sets ("simple I/O feedback").
    """
    if len(gold["models"]) != 1 or len(cand["models"]) != 1:
        return None, None
    g, c = set(gold["models"][0]), set(cand["models"][0])
    return sorted(c - g), sorted(g - c)


