"""Prompt templates for the generator, the ASPIRE feedback LLM and the baselines.

Provenance of each template is stated next to it. Templates taken from a paper
are quoted as closely as the ASP setting allows; every substitution is marked.
"""
import json

# --------------------------------------------------------------------------
# Generator (fine-tuned model). Rendered as `Question: {text}\nAnswer:`, the
# frame used in fine-tuning (aspire.training.fim.encode_answer).
# --------------------------------------------------------------------------


def specification(public):
    """Task text q shown to every generator: question, fixed background, F0."""
    text = public["question"]
    if public["background_asp"].strip():
        text += "\nFixed background ASP (already supplied; do not redefine):\n" + public["background_asp"]
    return text + "\nExample input facts (do not copy into your program):\n" + public["facts"]


# One worked repair example (the paper's hospital case, Sec. 3.4). The fine-tuned
# generator was never trained on the revision format, so the format is shown once.
REPAIR_EXAMPLE = {
    "question": ('Build candidate(X,H) from "injured" and "hospital"; derive nearby(X,H) when '
                 '"distance"(X,H,D) is at most 5; then recommend pairs whose H is not listed by '
                 '"blocked". Input predicates: blocked/1, distance/3, hospital/1, injured/1. '
                 'Name the output predicate exactly: recommended/2.'),
    "program": ("candidate(X,H) :- injured(X), hospital(H).\n"
                "nearby(X,H) :- candidate(X,H), distance(X,H,D), D<=5.\n"
                "recommended(X,H) :- nearby(X,H)."),
    "advice": ("Category: missing condition. Rule: the rule defining recommended/2. "
               "Violated condition: hospitals listed by blocked must not be recommended. "
               "Phase: not applicable. "
               "Fix direction: add the default-negation condition not blocked(H) to that rule."),
    "fixed": ("candidate(X,H) :- injured(X), hospital(H).\n"
              "nearby(X,H) :- candidate(X,H), distance(X,H,D), D<=5.\n"
              "recommended(X,H) :- nearby(X,H), not blocked(H)."),
}


def revision_request(spec, program, advice, with_example=True):
    """G(q, P_k, r_k): problem, current program and revision advice; no counterexample facts."""
    head = ""
    if with_example:
        e = REPAIR_EXAMPLE
        head = ("Example of a revision.\nProblem: " + e["question"] + "\nCurrent program:\n" + e["program"] +
                "\nRevision advice: " + e["advice"] + "\nCorrected program:\n" + e["fixed"] + "\n\nNow the task.\n")
    return (head + "Problem: " + spec + "\nCurrent program:\n" + program +
            "\nRevision advice: " + advice +
            "\nWrite the corrected complete ASP program. Do not repeat input facts or fixed background.")


# --------------------------------------------------------------------------
# Verified diagnosis rendering (shared by template feedback and feedback LLMs)
# --------------------------------------------------------------------------


def render_diagnosis(d, style="delta"):
    """Template feedback: atom delta when both sides have one projection, else projection sets."""
    phase = ""
    if d["optimization"]:
        phase = (" in the feasible phase (weak constraints ignored)" if d["phase"] == "feasible"
                 else " in the optimal phase (only optimal answer sets)")
    if d["kind"] == "empty_program":
        return "No ASP program could be extracted from the output."
    if d["kind"] == "parse_error":
        return f"Clingo could not parse or ground the program on input facts:\n{d['facts']}\nError: {d['error']}"
    if d["kind"] == "timeout":
        return f"Solving did not finish within the time limit on input facts:\n{d['facts']}"
    text = f"On input facts:\n{d['facts']}\nthe program's output differs from the expected output{phase}.\n"
    if d["kind"] == "satisfiability_mismatch":
        text += ("Expected: no answer set. " if d["expected_unsat"] else f"Expected answer sets: {d['expected']}. ")
        text += ("Obtained: no answer set." if d["actual_unsat"] else f"Obtained answer sets: {d['actual']}.")
        return text
    if style == "delta" and d["spurious"] is not None:
        return text + f"Wrongly derived atoms: {d['spurious'] or 'none'}\nMissing atoms: {d['missing'] or 'none'}"
    return text + f"Expected projected answer sets: {d['expected']}\nObtained projected answer sets: {d['actual']}"


# --------------------------------------------------------------------------
# ASPIRE semantic feedback (Sec. 3.4). Written for this paper.
# Constraints: (1) compare the failure against the conditions in the problem
# text; (2) fixed Category / Rule / Violated condition / Phase / Fix direction
# format; (3) optimization failures state the feasible or optimal phase.
# (2) and (3) are enforced by aspire.refine.loop.sanitize_feedback.
# The Base model is prompted by completion with two worked examples.
# --------------------------------------------------------------------------

CATEGORIES = ("missing condition", "wrong variable binding", "broken rule chaining",
              "choice or optimization construct", "syntax or grounding")

_SEMANTIC_HEADER = (
    "You review Answer Set Programming (clingo) programs. A verifier ran the candidate program on one "
    "input and found a mismatch with the expected output. Compare the mismatch with the conditions "
    "stated in the problem, find the general rule-level cause, and give revision advice.\n"
    "Rules: do not write ASP code; do not mention the specific constants of the failing input; "
    "describe conditions in terms of predicates and variables.\n"
    "Answer in exactly this format:\n"
    "<feedback>\nCategory: one of " + "; ".join(CATEGORIES) + "\n"
    "Rule: the predicate or rule to change\nViolated condition: the condition of the problem that is violated\n"
    "Phase: feasible, optimal or not applicable\nFix direction: what to add, remove or change\n</feedback>\n")

_SEMANTIC_SHOTS = [
    {"problem": REPAIR_EXAMPLE["question"], "program": REPAIR_EXAMPLE["program"],
     "evidence": ("On input facts:\ninjured(1). hospital(2). distance(1,2,3). blocked(2).\n"
                  "the program's output differs from the expected output.\n"
                  "Wrongly derived atoms: ['recommended(1,2)']\nMissing atoms: none"),
     "answer": ("Category: missing condition\nRule: the rule defining recommended/2\n"
                "Violated condition: a hospital listed by blocked must not be recommended\n"
                "Phase: not applicable\nFix direction: add the negated condition on blocked for H to that rule")},
    {"problem": ('Define "linked" as the transitive closure of "team". Input predicates: team/2. '
                 'Name the output predicate exactly: linked/2.'),
     "program": "linked(X,Y) :- team(X,Y).",
     "evidence": ("On input facts:\nteam(5,6). team(6,7).\nthe program's output differs from the expected output.\n"
                  "Wrongly derived atoms: none\nMissing atoms: ['linked(5,7)']"),
     "answer": ("Category: broken rule chaining\nRule: the rules defining linked/2\n"
                "Violated condition: linked must contain every pair connected by a chain of team edges\n"
                "Phase: not applicable\nFix direction: add a recursive rule that extends linked by one team edge")},
]


def _semantic_block(problem, program, evidence):
    return f"Problem:\n{problem}\nCandidate program:\n{program}\nVerifier evidence:\n{evidence}\n"


def semantic_feedback_prompt(public, program, diagnosis):
    shots = "".join(_semantic_block(s["problem"], s["program"], s["evidence"]) +
                    "<feedback>\n" + s["answer"] + "\n</feedback>\n\n" for s in _SEMANTIC_SHOTS)
    return (_SEMANTIC_HEADER + "\n" + shots +
            _semantic_block(specification(public), program, render_diagnosis(diagnosis)) + "<feedback>\n")


# --------------------------------------------------------------------------
# Self-Debugging (Chen et al., ICLR 2024, arXiv 2304.05128), "UT + Expl" variant,
# 0-shot chat prompt of Appendix G.7 (MBPP). The same model explains and fixes;
# correctness is decided by execution, as in the TransCoder setting.
# Substitutions: "Python code" -> "ASP program"; the failed unit test is one
# failing visible context (facts, expected and obtained projected output).
# --------------------------------------------------------------------------

SELF_DEBUG_EXPLAIN = "Explain the ASP program line by line."


def self_debug_explain(spec, program):
    return f"Complete the following task in ASP:\n{spec}\n{program}\n{SELF_DEBUG_EXPLAIN}"


def self_debug_fix(spec, program, explanation, diagnosis):
    return (f"Complete the following task in ASP:\n{spec}\n{program}\n{SELF_DEBUG_EXPLAIN}\n{explanation}\n"
            "The code above fails the given unit test:\n" + render_diagnosis(diagnosis, "sets") +
            "\nPlease fix the ASP program.")


# --------------------------------------------------------------------------
# PGS (He et al., arXiv 2506.18315), Tester-only adaptation.
# Tester: property statement adapted from the property-generation prompt (Table 11);
# it is asked for the violated property in prose instead of an executable checker.
# Generator: repair prompt of Table 13, verbatim structure.
# --------------------------------------------------------------------------


def pgs_tester_prompt(public, program, diagnosis):
    return ("You are a software testing expert. Your task is to analyze the problem description and state a "
            "specific property or invariant a correct solution must satisfy, which the candidate program "
            "violates on the failing input below. State the property in one or two sentences of prose. "
            "Do not write code.\n"
            f"### Problem Description:\n{specification(public)}\n### Candidate Program:\n{program}\n"
            f"### Failing Execution:\n{render_diagnosis(diagnosis, 'sets')}\n### Violated Property:\n")


def pgs_repair_prompt(spec, program, diagnosis, prop):
    if diagnosis["kind"] in ("parse_error", "timeout", "empty_program"):
        context = (f"Context: The program previously encountered a runtime error. Input: {diagnosis['facts']} "
                   f"Error Details: {diagnosis['error'] or diagnosis['kind']}")
    else:
        context = (f"Context: The program previously produced a wrong answer. Input: {diagnosis['facts']} "
                   f"Generated Output: {diagnosis['actual']} Expected Output: {diagnosis['expected']}")
        if prop:
            context += f" Violated Property: {prop}"
    return (f"### Question: {spec}\n### Buggy Code: ```asp\n{program}\n```\n### Error Context: {context}\n"
            "### Your Task: First, provide a concise explanation of the error. Then, generate the entire corrected program.")


# --------------------------------------------------------------------------
# LLM+ASP (Ishay & Lee, Findings of ACL 2026, arXiv 2604.27960), generator prompt
# from the official repository (azreasoners/llm-asp-programmer, core/prompts.py).
# Substitutions: the candidate-answer-set block shows clingo output on the
# problem instance F0 projected onto the declared outputs; the optional resource
# material is replaced by the short guide below (not the authors' compact.txt).
# --------------------------------------------------------------------------

ASP_GUIDE = """ASP reference:
Rules h(X) :- p(X), not q(X). derive h for p without q.
Every variable must be bound by a positive body atom.
Constants and quoted strings differ. Keep input predicate names and arities.
Recursive relations need a base case.
a(X) | b(X) :- d(X). chooses a or b for each d.
:- a(X), a(Y), X<Y. forbids two different a.
:~ not a(X), d(X). [1@1,X] penalises each d without a.
Return a reusable program for arbitrary valid input facts, not a fixed solution."""


def llm_asp_prompt(spec, program, clingo_output):
    return ("Consider the following resource material describing Answer Set Programming and the language of "
            "Clingo to do the following task.\n<resource-material>\n" + ASP_GUIDE + "\n</resource-material>\n"
            "<instructions>\nYou are tasked with constructing an ASP program to solve the following problem:\n"
            f"<problem-description>\n{spec}\n</problem-description>\n"
            "The ASP program should be made of ASP modules, which each make up a necessary part of the program "
            "to represent the problem.\n\n<current-clingo-modules>\nHere are the current ASP modules:\n"
            f"{program}\n</current-clingo-modules>\n<candidate-answer-set>\n"
            "Here is the candidate answer set to solve the problem, from running the ASP program:\n"
            f"{clingo_output}\n</candidate-answer-set>\n\n"
            "You are tasked with doing the an operation from the following.\n"
            "(UPDATE) - This should be done either to:\n\t(a) write an initial ASP program (still considered an update).\n"
            "\t(b) update the program based on the Clingo output which may help to debug the current output.\n"
            "If updated, make sure to write the complete program.\n"
            "(PASS) - This should be done when the Clingo output is correct. There should be no ambiguity and "
            "consensus on the proposed solution being correct. ONLY use when absolutely sure the output is correct. "
            "If the proposed solution is correct, you may ignore stderr and still pass.\n\n"
            "For UPDATE, format your output exactly like the following, writing the complete program:\n```\n"
            "% MODULES START\n% module <name of first module>\n<ASP code for this module>\n"
            "% module <name of first module> END\n...\n% MODULES END\n```\n"
            "For PASS, format your output exactly like the following (only write the action, since nothing else "
            "is to be done):\n```\nOPERATION: PASS\n```\n\nDo not write anything outside of the three backticks.\n"
            "</instructions>")


def clingo_output_text(results, program):
    """stdout/stderr-like rendering of the feasible-phase result on F0 (first answer set shown)."""
    r = results[0]
    if not r["parse_ground"]:
        lines = []
        source = program.splitlines()
        import re
        for m in re.finditer(r":(\d+):\d+", r.get("error") or ""):
            i = int(m.group(1)) - 1
            if 0 <= i < len(source):
                lines.append(source[i])
        text = "stdout:\nUNKNOWN\n\nstderr:\n" + (r.get("error") or "")
        return text + ("\n\nLines where errors occur:\n" + "\n".join(dict.fromkeys(lines)) if lines else "")
    if not r["terminated"]:
        return "stdout:\nINTERRUPTED (time limit)"
    if not r["satisfiable"]:
        return "stdout:\nUNSATISFIABLE\nModels : 0"
    more = "+" if len(r["models"]) > 1 else ""
    return f"stdout:\nAnswer: 1\n{' '.join(r['models'][0])}\nSATISFIABLE\nModels : 1{more}"


# --------------------------------------------------------------------------
# Few-shot direct generation (no fine-tuning): Question/Answer exemplars.
# --------------------------------------------------------------------------


def fewshot_prompt(exemplars, spec):
    """Question/Answer exemplars; the generator renderer adds the outer `Question: ... Answer:` frame."""
    blocks = [f"{q}\nAnswer: {a}" for q, a in exemplars]
    return "\n\nQuestion: ".join(blocks + [spec])


def dump(value):
    return json.dumps(value, ensure_ascii=False)
