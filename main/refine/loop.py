"""Semantic counterexample refinement (Algorithm 1) and the refinement baselines.

All methods share one loop: verify P_k on the visible contexts, keep the best
candidate by visible score s(P) (strictly higher replaces), stop on a full
visible pass, an empty program, an unfinished verification or k = T, and
otherwise produce P_{k+1}. Methods differ ONLY in how P_{k+1} is produced:

  aspire       verified diagnosis -> feedback LLM (fixed rule-level format, checked) -> generator
  template     verified diagnosis rendered by template -> generator
  syntax_only  aspire, but only parse/grounding diagnoses are available
  self_debug   Self-Debugging UT+Expl: the generator explains its program, then fixes it
               given one failing visible context (Chen et al., 2024)
  pgs          PGS Tester-only: feedback LLM states the violated property, generator
               repairs with the PGS repair prompt (He et al., 2026)
  llm_asp      LLM+ASP: generator sees clingo output on F0 (no expected output) and
               answers UPDATE or PASS (Ishay & Lee, 2026). It does not stop on a
               visible pass, which it cannot observe; PASS or the budget ends it.

Held-out contexts never reach this module.
"""
import random
import re

from aspire.common import digest
from aspire.models import prompts as P
from aspire.verify.extraction import parse_answer
from aspire.verify.oracle import phases

METHODS = ("base", "fewshot", "aspire", "template", "syntax_only", "self_debug", "pgs", "llm_asp")
REFINE_METHODS = METHODS[2:]
SOURCES = {
    "aspire": "this paper (Sec. 3.4)",
    "template": "this paper (Sec. 4.7 control)",
    "syntax_only": "this paper (Table 5 ablation)",
    "self_debug": "Chen et al., ICLR 2024, arXiv 2304.05128; UT+Expl, Appendix G.7 prompt",
    "pgs": "He et al., arXiv 2506.18315; Tester feedback + Table 13 repair prompt only (no property checkers/input synthesis)",
    "llm_asp": "Ishay & Lee, Findings ACL 2026, arXiv 2604.27960; UPDATE/PASS prompt of azreasoners/llm-asp-programmer",
}

_GROUND_ATOM = re.compile(r"\b[a-z][A-Za-z0-9_]*\(\s*(?:\"[^\"]*\"|-?\d+|[a-z][A-Za-z0-9_]*)"
                          r"(?:\s*,\s*(?:\"[^\"]*\"|-?\d+|[a-z][A-Za-z0-9_]*))*\s*\)")
_FACT_CONSTANT = re.compile(r"\"[^\"]*\"|\b[a-z][A-Za-z0-9_]*\b(?!\s*\()")


def instance_identifiers(facts, spec):
    """Non-numeric constants of the counterexample that do not occur in the problem text."""
    found = set()
    for atom in re.finditer(r"[a-z][A-Za-z0-9_]*\(([^()]*)\)", facts):
        for const in _FACT_CONSTANT.findall(atom.group(1)):
            if const not in spec:
                found.add(const)
    return found


FEEDBACK_FIELDS = ("Category", "Rule", "Violated condition", "Phase", "Fix direction")
PHASES = ("feasible", "optimal", "not applicable")
_FIELD_LINE = re.compile(r"^\s*(" + "|".join(re.escape(f) for f in FEEDBACK_FIELDS) + r")\s*:\s*(.*?)\s*$",
                         re.I | re.M)


def expected_phase(diagnosis):
    """Phase the feedback must state: the mismatching phase for optimization tasks
    with an output mismatch, otherwise 'not applicable'."""
    if not diagnosis or not diagnosis.get("optimization"):
        return {"not applicable"}
    if diagnosis["kind"] in ("satisfiability_mismatch", "projection_mismatch"):
        return {diagnosis["phase"]}
    return {"not applicable", diagnosis["phase"]}


def parse_feedback_fields(text):
    """Map each of the five fixed fields to its value; a field given twice keeps the first value."""
    fields = {}
    for m in _FIELD_LINE.finditer(text):
        key = next(f for f in FEEDBACK_FIELDS if f.lower() == m.group(1).lower())
        fields.setdefault(key, m.group(2).strip().rstrip("."))
    return fields


def sanitize_feedback(raw, facts, spec, diagnosis=None, max_chars=1000):
    """Return (advice, rejection_reason). Sec. 3.4 invalid-feedback filter.

    Content checks: empty output; ASP code (code fence, ':-', ':~'); ground
    atoms such as recommended(1,2), which restate the failing instance;
    non-numeric constants that occur only in the counterexample. Numeric values
    are allowed because advice may state threshold relations.
    Format checks: all five fields present and non-empty; Category is one of
    the five categories; Phase is consistent with the verifier's diagnosis
    (the mismatching phase for optimization tasks, else 'not applicable').
    """
    from aspire.models.prompts import CATEGORIES
    text = str(raw or "")
    m = re.search(r"<feedback>(.*?)(?:</feedback>|$)", text, flags=re.I | re.S)
    text = (m.group(1) if m else text).replace("</feedback>", "").strip()
    if not text:
        return "", "empty"
    if "```" in text or ":-" in text or ":~" in text:
        return "", "contains_asp_code"
    if _GROUND_ATOM.search(text):
        return "", "contains_instance_atom"
    leaked = [c for c in instance_identifiers(facts, spec) if re.search(r"(?<![\w\"])" + re.escape(c) + r"(?![\w\"])", text)]
    if leaked:
        return "", "contains_instance_constant"
    fields = parse_feedback_fields(text)
    if any(not fields.get(f) for f in FEEDBACK_FIELDS):
        return "", "missing_field"
    if fields["Category"].lower() not in CATEGORIES:
        return "", "invalid_category"
    phase = fields["Phase"].lower()
    if phase not in PHASES or phase not in expected_phase(diagnosis):
        return "", "phase_mismatch"
    advice = " ".join(f"{f}: {fields[f]}." for f in FEEDBACK_FIELDS)
    return re.sub(r"[ \t]+", " ", advice).strip()[:max_chars], None


def _program(raw):
    return parse_answer(raw)[0]


def refine(method, public, initial, backend, oracle, args):
    """Return (selected_program, trace, trajectory)."""
    if method not in REFINE_METHODS:
        raise ValueError(method)
    spec = P.specification(public)
    rng = random.Random(args.seed + int(digest(public["sample_id"])[:8], 16))
    total = len(oracle.contexts)
    trace, trajectory = [], [initial]
    current, best, best_score = initial, initial, -1

    def call(text, role, step, phase, stop=()):
        return backend.generate(text, role, {"sample_id": public["sample_id"], "method": method,
                                             "round": step, "phase": phase}, stop)

    for k in range(args.attempts + 1):
        checks = oracle.check(current)
        score = sum(c["correct"] for c in checks)
        unfinished = any(c["timed_out"] for c in checks)
        record = {"round": k, "candidate_program": current, "visible_score": score,
                  "visible_total": total, "verification_unfinished": unfinished}
        if score > best_score:
            best, best_score = current, score
            record["new_best"] = True
        # LLM+ASP has no expected outputs; as in the paper, only the model's PASS ends it early.
        visible_pass = score == total and method != "llm_asp"
        stop = ("visible_pass" if visible_pass else
                "empty_program" if not current.strip() else
                "verification_unfinished" if unfinished else
                "budget" if k == args.attempts else None)
        if stop:
            record["stop"] = stop
            trace.append(record)
            break
        step = k + 1
        if method == "llm_asp":
            results = [oracle.solver.solve(public["background_asp"] + "\n" + current, public["facts"],
                                           public["targets"], phases(public)[-1])]
            output = P.clingo_output_text(results, current)
            raw, key = call(P.llm_asp_prompt(spec, current, output), args.baseline_role, step, "update_or_pass")
            record.update(clingo_output=output, raw=raw, call_hash=key)
            if re.search(r"OPERATION\s*:\s*PASS", raw):
                record["stop"] = "pass"
                trace.append(record)
                break
            new = _program(raw)
        else:
            diagnosis = oracle.diagnosis(current, args.selection, rng)
            record["diagnosis"] = {k2: v for k2, v in diagnosis.items() if not k2.endswith("_models")}
            if method == "syntax_only" and diagnosis["kind"] != "parse_error":
                record["stop"] = "no_syntax_signal"
                trace.append(record)
                break
            if method in ("aspire", "syntax_only"):
                raw_fb, key_fb = call(P.semantic_feedback_prompt(public, current, diagnosis), "feedback",
                                      step, "feedback", stop=("</feedback>",))
                advice, rejected = sanitize_feedback(raw_fb, diagnosis["facts"], spec, diagnosis)
                record.update(feedback_raw=raw_fb, feedback_call_hash=key_fb, feedback=advice,
                              feedback_rejected=rejected)
                if rejected:  # the round is consumed; no fallback to template feedback
                    trajectory.append(current)
                    trace.append(record)
                    continue
                text = P.revision_request(spec, current, advice)
            elif method == "template":
                text = P.revision_request(spec, current, P.render_diagnosis(diagnosis))
            elif method == "self_debug":
                explanation, key_ex = call(P.self_debug_explain(spec, current), args.baseline_role, step, "explain")
                record.update(explanation=explanation, explain_call_hash=key_ex)
                text = P.self_debug_fix(spec, current, explanation, diagnosis)
            elif method == "pgs":
                prop = ""
                if diagnosis["kind"] not in ("parse_error", "timeout", "empty_program"):
                    raw_prop, key_p = call(P.pgs_tester_prompt(public, current, diagnosis), "feedback", step,
                                           "tester", stop=("\n###", "\n\n"))
                    prop = re.sub(r"\s+", " ", raw_prop.split("###")[0]).strip()[:600]
                    record.update(tester_raw=raw_prop, tester_call_hash=key_p, feedback=prop)
                text = P.pgs_repair_prompt(spec, current, diagnosis, prop)
            role = args.baseline_role if method in ("self_debug", "pgs") else "generator"
            raw, key = call(text, role, step, "revise")
            record.update(revise_prompt=text, raw=raw, call_hash=key)
            new = _program(raw)
        record["format_failure"] = (not new.strip()) or new.strip() == current.strip()
        trace.append(record)
        current = new
        trajectory.append(current)
    selected = best if args.keep == "best" else current
    return selected, trace, trajectory
