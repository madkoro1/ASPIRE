"""Verifier, oracle, feedback filter, refinement loop and runner tests (CPU, clingo only)."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from aspire.common import digest, file_hash, write_json
from aspire.families import PAPER_FAMILIES, difficulty
from aspire.refine.loop import refine, sanitize_feedback
from aspire.verify.oracle import Solver, VisibleOracle, evaluate, expected_record, full_program, phases

GOLD = ("candidate(X,H) :- transport(X), dispatch(H).\n"
        "nearby(X,H) :- candidate(X,H), road(X,H,D), D<=5.\n"
        "recommended(X,H) :- nearby(X,H), not injured(H).")
NO_BLOCK = "\n".join(GOLD.splitlines()[:2] + ["recommended(X,H) :- nearby(X,H)."])

PUBLIC = {"sample_id": "s1", "sample_index": 0, "task": "track_a_l3_combination_distance_blocked", "level": 3,
          "question": "Recommend pairs whose H is not listed by injured. Output recommended/2.",
          "facts": "transport(1).dispatch(3).road(1,3,2).",
          "targets": [["recommended", 2]], "optimization": False, "background_asp": "",
          "feedback_contexts": [{"name": "context_0", "facts": "transport(1).dispatch(3).dispatch(4).road(1,3,2).road(1,4,1).injured(4)."},
                                {"name": "context_1", "facts": "transport(1).dispatch(4).road(1,4,9)."},
                                {"name": "context_2", "facts": "transport(2).dispatch(5).road(2,5,1).injured(5)."}]}
PRIVATE = {"sample_id": "s1", "gold_program": GOLD,
           "scoring_contexts": [{"name": "context_3", "facts": "transport(7).dispatch(8).road(7,8,3).injured(8)."}]}

OPT_PUBLIC = {**PUBLIC, "sample_id": "s2", "task": "track_a_l3_seen_admission", "optimization": True,
              "targets": [["admit", 1]], "facts": "h(1..2).", "feedback_contexts": [{"name": "context_0", "facts": "h(1..3)."}]}
OPT_GOLD = "admit(X) | skip(X) :- h(X).\n:- admit(X), admit(Y), X<Y.\n:~ skip(X), h(X). [X@1,X]"


class Args(SimpleNamespace):
    def __init__(self, **kw):
        super().__init__(**{"seed": 56, "attempts": 3, "selection": "shortest", "keep": "best",
                            "baseline_role": "generator", **kw})


class ScriptedBackend:
    """Returns pre-set responses per (role, phase) and records every prompt."""

    def __init__(self, responses):
        self.responses, self.calls = responses, []

    def generate(self, text, role, metadata, stop=()):
        self.calls.append((role, metadata["phase"], text))
        queue = self.responses[(role, metadata["phase"])]
        return (queue.pop(0) if len(queue) > 1 else queue[0]), str(len(self.calls))


class VerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solver = Solver(20)

    @classmethod
    def tearDownClass(cls):
        cls.solver.close()

    def test_single_instance_hides_missing_condition(self):
        oracle = VisibleOracle(self.solver, PUBLIC, GOLD)
        checks = oracle.check(NO_BLOCK)
        self.assertTrue(checks[0]["correct"])          # F0 has no injured hospital
        self.assertFalse(checks[1]["correct"])         # context_0 exposes it
        d = oracle.diagnosis(NO_BLOCK)
        self.assertEqual(d["context_name"], "context_2")   # shortest of the two failing contexts
        self.assertEqual(d["spurious"], ["recommended(2,5)"])
        self.assertEqual(oracle.score(GOLD), 4)
        self.assertIsNone(oracle.diagnosis(GOLD))
        self.assertTrue(evaluate(self.solver, PUBLIC, PRIVATE, GOLD)["end_to_end_correct"])
        self.assertFalse(evaluate(self.solver, PUBLIC, PRIVATE, NO_BLOCK)["end_to_end_correct"])

    def test_shortest_failing_context_and_instance_only(self):
        wrong = GOLD.replace("D<=5", "D<=0")
        d = VisibleOracle(self.solver, PUBLIC, GOLD).diagnosis(wrong)
        failing = [c for c in VisibleOracle(self.solver, PUBLIC, GOLD).check(wrong) if not c["correct"]]
        self.assertEqual(len(d["facts"]), min(len(c["facts"]) for c in failing))
        single = VisibleOracle(self.solver, PUBLIC, GOLD, visible="instance")
        self.assertEqual(len(single.contexts), 1)
        self.assertIsNone(single.diagnosis(NO_BLOCK))

    def test_projection_ignores_auxiliary_predicates(self):
        renamed = GOLD.replace("candidate", "pair").replace("nearby", "close")
        self.assertEqual(VisibleOracle(self.solver, PUBLIC, GOLD).score(renamed), 4)

    def test_optimization_checks_feasible_and_optimal(self):
        oracle = VisibleOracle(self.solver, OPT_PUBLIC, OPT_GOLD)
        self.assertEqual(oracle.score(OPT_GOLD), 2)
        pinned = "admit(1)."   # matches the optimum but not the feasible choice space
        d = oracle.diagnosis(pinned)
        self.assertEqual(d["phase"], "feasible")
        no_pref = "admit(X) | skip(X) :- h(X).\n:- admit(X), admit(Y), X<Y."
        self.assertEqual(oracle.diagnosis(no_pref)["phase"], "optimal")

    def test_parse_error_and_forbidden_directives(self):
        oracle = VisibleOracle(self.solver, PUBLIC, GOLD)
        self.assertEqual(oracle.diagnosis("recommended(X,H) :- .")["kind"], "parse_error")
        self.assertEqual(oracle.diagnosis('#include "x.lp".')["kind"], "parse_error")
        self.assertEqual(oracle.diagnosis("")["kind"], "empty_program")

    def test_frozen_expected_outputs_replace_reference(self):
        for pub, gold in ((PUBLIC, GOLD), (OPT_PUBLIC, OPT_GOLD)):
            frozen = freeze_expected(self.solver, pub, gold)
            with_gold, without = VisibleOracle(self.solver, pub, gold), VisibleOracle(self.solver, frozen)
            for cand in (gold, NO_BLOCK, "admit(1).", ""):
                self.assertEqual(with_gold.score(cand), without.score(cand))
                self.assertEqual(with_gold.diagnosis(cand), without.diagnosis(cand))
        private = freeze_expected(self.solver, PUBLIC, GOLD, PRIVATE)
        private.pop("gold_program")
        self.assertTrue(evaluate(self.solver, freeze_expected(self.solver, PUBLIC, GOLD), private, GOLD)["end_to_end_correct"])
        with self.assertRaises(ValueError):
            VisibleOracle(self.solver, PUBLIC).score(GOLD)   # no expected output and no reference


def freeze_expected(solver, public, gold, private=None):
    """What aspire.data.freeze stores: expected outputs per context."""
    def exp(facts):
        return {ph: expected_record(solver.solve(full_program(public, gold), facts, public["targets"], ph))
                for ph in phases(public)}
    if private is not None:
        return {**private, "scoring_contexts": [{**c, "expected": exp(c["facts"])} for c in private["scoring_contexts"]]}
    return {**public, "instance_expected": exp(public["facts"]),
            "feedback_contexts": [{**c, "expected": exp(c["facts"])} for c in public["feedback_contexts"]]}


VALID_FB = ("Category: missing condition\nRule: recommended/2\nViolated condition: injured hospitals are excluded\n"
            "Phase: not applicable\nFix direction: add not injured(H)\n</feedback>")


class FeedbackFilterTests(unittest.TestCase):
    def test_accepts_rule_level_advice_with_numbers(self):
        text, why = sanitize_feedback("<feedback>Category: missing condition\nRule: nearby/2\n"
                                      "Violated condition: distance at most 5\nPhase: not applicable\n"
                                      "Fix direction: add not blocked(H); keep distances at most 5</feedback>",
                                      "road(1,3,2).", "q")
        self.assertIsNone(why)
        self.assertIn("not blocked(H)", text)

    def test_rejections(self):
        facts, spec = 'bed(1,"urgent").bed(2,mild).ward(north).', 'select "urgent" beds'
        cases = {"": "empty", "use ```p :- q.```": "contains_asp_code", "add p :- q": "contains_asp_code",
                 "recommended(1,2) is wrong": "contains_instance_atom", "the ward north is wrong": "contains_instance_constant"}
        for raw, reason in cases.items():
            self.assertEqual(sanitize_feedback(raw, facts, spec)[1], reason, raw)
        ok = VALID_FB.replace("injured hospitals are excluded", 'the label "urgent" must match')
        self.assertIsNone(sanitize_feedback(ok, facts, spec)[1])

    def test_format_is_enforced(self):
        self.assertEqual(sanitize_feedback("Looks good.", "", "q")[1], "missing_field")
        no_phase = "\n".join(l for l in VALID_FB.splitlines() if not l.startswith("Phase"))
        self.assertEqual(sanitize_feedback(no_phase, "", "q")[1], "missing_field")
        bad_cat = VALID_FB.replace("missing condition", "imaginary category")
        self.assertEqual(sanitize_feedback(bad_cat, "", "q")[1], "invalid_category")
        # optimization projection mismatch in the optimal phase must be stated as such
        d = {"optimization": True, "kind": "projection_mismatch", "phase": "optimal"}
        self.assertEqual(sanitize_feedback(VALID_FB, "", "q", d)[1], "phase_mismatch")
        self.assertIsNone(sanitize_feedback(VALID_FB.replace("not applicable", "optimal"), "", "q", d)[1])
        # non-optimization tasks must say not applicable
        plain = {"optimization": False, "kind": "projection_mismatch", "phase": "feasible"}
        self.assertEqual(sanitize_feedback(VALID_FB.replace("not applicable", "feasible"), "", "q", plain)[1],
                         "phase_mismatch")
        advice, _ = sanitize_feedback(VALID_FB, "", "q", plain)
        self.assertTrue(advice.startswith("Category: missing condition."))


class LoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solver = Solver(20)

    @classmethod
    def tearDownClass(cls):
        cls.solver.close()

    def oracle(self):
        return VisibleOracle(self.solver, PUBLIC, GOLD)

    def test_aspire_repairs_without_counterexample_facts_in_generator_prompt(self):
        b = ScriptedBackend({("feedback", "feedback"): [VALID_FB],
                             ("generator", "revise"): [GOLD]})
        selected, trace, trajectory = refine("aspire", PUBLIC, NO_BLOCK, b, self.oracle(), Args())
        self.assertEqual(self.oracle().score(selected), 4)
        self.assertEqual(trace[-1]["stop"], "visible_pass")
        revise = [t for r, p, t in b.calls if p == "revise"][0]
        self.assertNotIn("injured(5)", revise)      # counterexample facts stay out of the generator prompt
        self.assertIn(PUBLIC["facts"], revise)       # F0 is part of the task text

    def test_rejected_feedback_retries_then_switches_context(self):
        # NO_BLOCK fails context_2 (shortest) and context_0; every answer is rejected.
        b = ScriptedBackend({("feedback", "feedback"): ["recommended(2,5) is wrong"],
                             ("feedback", "feedback_retry"): ["recommended(2,5) is wrong"],
                             ("generator", "revise"): ["x(1)."]})
        selected, trace, _ = refine("aspire", PUBLIC, NO_BLOCK, b, self.oracle(), Args())
        self.assertEqual(selected, NO_BLOCK)
        self.assertEqual([t["diagnosis"]["context_name"] for t in trace[:-1]], ["context_2", "context_0"])
        self.assertTrue(all(t["first_feedback_rejected"] == t["feedback_rejected"] == "contains_instance_atom"
                            for t in trace[:-1]))
        self.assertEqual(trace[-1]["stop"], "feedback_exhausted")   # both failing contexts tried
        self.assertFalse(any(p == "revise" for _, p, _ in b.calls))
        retry = [t for r, p, t in b.calls if p == "feedback_retry"]
        self.assertEqual(len(retry), 2)
        self.assertIn("was rejected because it restated a ground atom", retry[0])

    def test_retry_with_reason_can_recover(self):
        b = ScriptedBackend({("feedback", "feedback"): ["recommended(2,5) is wrong"],
                             ("feedback", "feedback_retry"): [VALID_FB],
                             ("generator", "revise"): [GOLD]})
        selected, trace, _ = refine("aspire", PUBLIC, NO_BLOCK, b, self.oracle(), Args())
        self.assertEqual(self.oracle().score(selected), 4)
        self.assertEqual(trace[0]["first_feedback_rejected"], "contains_instance_atom")
        self.assertIsNone(trace[0]["feedback_rejected"])
        self.assertEqual(trace[-1]["stop"], "visible_pass")

    def test_strictly_better_replaces_and_keep_last(self):
        b = ScriptedBackend({("generator", "revise"): ["recommended(X,H) :- dispatch(H), transport(X)."]})
        best, _, traj = refine("template", PUBLIC, NO_BLOCK, b, self.oracle(), Args())
        last, _, _ = refine("template", PUBLIC, NO_BLOCK, b, self.oracle(), Args(keep="last"))
        self.assertEqual(best, NO_BLOCK)
        self.assertNotEqual(last, NO_BLOCK)
        self.assertEqual(len(traj), 4)

    def test_empty_candidate_stops(self):
        b = ScriptedBackend({("generator", "revise"): ["no program here"]})
        _, trace, _ = refine("template", PUBLIC, NO_BLOCK, b, self.oracle(), Args())
        self.assertEqual(trace[-1]["stop"], "empty_program")
        self.assertTrue(trace[0]["format_failure"])

    def test_baseline_prompts(self):
        b = ScriptedBackend({("generator", "explain"): ["line by line"], ("generator", "revise"): [GOLD]})
        refine("self_debug", PUBLIC, NO_BLOCK, b, self.oracle(), Args())
        self.assertIn("Explain the ASP program line by line.", b.calls[0][2])
        self.assertIn("The code above fails the given unit test:", b.calls[1][2])
        b = ScriptedBackend({("feedback", "tester"): ["Injured hospitals are never recommended."],
                             ("generator", "revise"): [GOLD]})
        refine("pgs", PUBLIC, NO_BLOCK, b, self.oracle(), Args())
        self.assertIn("### Error Context: Context: The program previously produced a wrong answer.", b.calls[1][2])
        self.assertIn("Violated Property: Injured hospitals", b.calls[1][2])
        b = ScriptedBackend({("generator", "update_or_pass"): ["```\nOPERATION: PASS\n```"]})
        selected, trace, _ = refine("llm_asp", PUBLIC, NO_BLOCK, b, self.oracle(), Args())
        self.assertEqual(trace[0]["stop"], "pass")
        self.assertNotIn("dispatch(4)", b.calls[0][2])    # LLM+ASP sees no feedback contexts
        self.assertNotIn("Expected", b.calls[0][2])

    def test_syntax_only_stops_without_parse_error(self):
        b = ScriptedBackend({})
        _, trace, _ = refine("syntax_only", PUBLIC, NO_BLOCK, b, self.oracle(), Args())
        self.assertEqual(trace[0]["stop"], "no_syntax_signal")


class SuiteAndRunnerTests(unittest.TestCase):
    def test_families(self):
        self.assertEqual(len(PAPER_FAMILIES), 13)
        self.assertEqual(difficulty("track_a_l1_choice"), "excluded")
        self.assertEqual(difficulty("track_a_l3_seen_admission"), "hard")

    def test_mock_end_to_end_run_and_report(self):
        from aspire.eval.report import main as report_main
        from aspire.eval.run import main as run_main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = root / "suite"
            suite.mkdir()
            excluded = {**PUBLIC, "sample_id": "s3", "task": "track_a_l1_choice"}
            for part in ("test",):
                (suite / f"{part}_iid.public.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in (PUBLIC, excluded)), encoding="utf-8")
                (suite / f"{part}_iid.private.jsonl").write_text(
                    "".join(json.dumps({**PRIVATE, "sample_id": r["sample_id"]}) + "\n" for r in (PUBLIC, excluded)),
                    encoding="utf-8")
            manifest = {"schema_version": 1, "files": {p.name: file_hash(p) for p in suite.glob("*.jsonl")}}
            manifest["suite_id"] = digest(manifest)
            write_json(suite / "manifest.json", manifest)
            common = ["--suite", str(suite), "--backend", "mock", "--model", "mock", "--clingo-timeout", "20"]
            run_main(common + ["--method", "base", "--output", str(root / "runs/base")])
            run_main(common + ["--method", "aspire", "--output", str(root / "runs/aspire"),
                               "--initial", str(root / "runs/base")])
            rows = [json.loads(x) for x in (root / "runs/aspire/protocol_samples.jsonl").read_text().splitlines()]
            self.assertEqual([r["sample_id"] for r in rows], ["s1"])   # excluded family filtered out
            report_main(["--root", str(root / "runs"), "--output", str(root / "report"), "--reference", "base"])
            self.assertTrue((root / "report/accuracy.csv").exists())
            self.assertIn("aspire", (root / "report/REPORT.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
