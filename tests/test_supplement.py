"""Supplementary experiments: generic AST baseline, verifier ablations, aspire_facts,
failure taxonomy, hard-coding detection and the verifier benchmark (CPU, clingo only)."""
import random
import unittest

from aspire.eval.errors import classify, hardcoded, program_constants
from aspire.eval.verifier_bench import Bench, judge_metrics, negatives, positives, probe_contexts
from aspire.models import prompts as P
from aspire.refine.loop import refine
from aspire.training.fim import ast_all_spans, ast_spans, split_source
from aspire.verify.oracle import Solver, VisibleOracle, evaluate

from tests.test_pipeline import (GOLD, NO_BLOCK, OPT_GOLD, OPT_PUBLIC, PRIVATE, PUBLIC, VALID_FB, Args,
                                 ScriptedBackend)


class GenericAstTests(unittest.TestCase):
    def test_all_nodes_cover_semantic_units_and_finer_nodes(self):
        source = "nearby(X,H) :- candidate(X,H), distance(X,H,D), D<=5.\nrecommended(X,H) :- nearby(X,H), not blocked(H)."
        texts = {source[s["start"]:s["end"]] for s in ast_all_spans(source)}
        semantic = {source[s["start"]:s["end"]] for s in ast_spans(source)}
        self.assertTrue(semantic <= texts)            # every semantic unit is also an AST node
        self.assertIn("H", texts)                     # finer nodes: variables, terms, atoms
        self.assertIn("5", texts)
        for span in ast_all_spans(source):
            self.assertEqual("".join(split_source(source, span)), source)


class VerifierAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solver = Solver(20)

    @classmethod
    def tearDownClass(cls):
        cls.solver.close()

    def test_no_projection_rejects_renamed_auxiliaries(self):
        renamed = GOLD.replace("nearby", "close")
        standard = VisibleOracle(self.solver, PUBLIC, GOLD)
        whole = VisibleOracle(self.solver, PUBLIC, GOLD, project=False)
        self.assertEqual(standard.score(renamed), 4)
        self.assertLess(whole.score(renamed), 4)
        with self.assertRaises(ValueError):
            VisibleOracle(self.solver, PUBLIC, None, project=False)

    def test_optimal_only_accepts_pinned_optimum(self):
        pinned = "admit(2)."   # the optimum on h(1..2): skipping 1 costs least
        both = VisibleOracle(self.solver, OPT_PUBLIC, OPT_GOLD)
        optimal = VisibleOracle(self.solver, OPT_PUBLIC, OPT_GOLD, phases_mode="optimal")
        self.assertFalse(both.check(pinned)[0]["correct"])
        self.assertTrue(optimal.check(pinned)[0]["correct"])   # the hole the feasible phase closes
        self.assertEqual(optimal.diagnosis("")["phase"], "optimal")

    def test_aspire_facts_shows_counterexample_to_generator(self):
        b = ScriptedBackend({("feedback", "feedback"): [VALID_FB], ("generator", "revise"): [GOLD]})
        refine("aspire_facts", PUBLIC, NO_BLOCK, b, VisibleOracle(self.solver, PUBLIC, GOLD), Args())
        revise = [t for _, p, t in b.calls if p == "revise"][0]
        self.assertIn("Failing input facts:\ntransport(2).dispatch(5).road(2,5,1).injured(5).", revise)
        self.assertEqual(P.revision_request("q", "p.", "a"), P.revision_request("q", "p.", "a", facts=None))


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solver = Solver(20)

    @classmethod
    def tearDownClass(cls):
        cls.solver.close()

    def row(self, program, initial=None, trace=()):
        r = evaluate(self.solver, PUBLIC, PRIVATE, program)
        r.update(candidate_program=program, initial_candidate_program=initial or program,
                 refinement_trace=list(trace), task=PUBLIC["task"])
        return r

    def test_missing_negation_is_over_generation(self):
        cat, ind = classify(self.row(NO_BLOCK), PUBLIC, GOLD)
        self.assertEqual(cat, "over")
        self.assertIn("fewer_negations", ind)

    def test_syntax_interface_and_undefined_predicate(self):
        self.assertEqual(classify(self.row("recommended(X,H) :- ."), PUBLIC, GOLD)[0], "syntax")
        self.assertEqual(classify(self.row("rec(X,H) :- transport(X), dispatch(H)."), PUBLIC, GOLD)[0], "interface")
        broken = "recommended(X,H) :- near(X,H), not injured(H)."
        cat, ind = classify(self.row(broken), PUBLIC, GOLD)
        self.assertEqual(cat, "under")
        self.assertIn("undefined_body_pred", ind)

    def test_hardcoded_counterexample_constant(self):
        trace = [{"diagnosis": {"facts": "transport(7).dispatch(9).road(7,9,1).injured(9)."}}]
        patched = NO_BLOCK.replace("recommended(X,H) :- nearby(X,H).", "recommended(X,H) :- nearby(X,H), H!=9.")
        self.assertEqual(hardcoded(self.row(patched, NO_BLOCK, trace), PUBLIC, GOLD), {"9"})
        self.assertEqual(hardcoded(self.row(GOLD, NO_BLOCK, trace), PUBLIC, GOLD), set())
        self.assertNotIn("not", program_constants(GOLD))


class JudgeMetricTests(unittest.TestCase):
    def test_confusion_matrix_metrics(self):
        # 10 equivalent programs, 8 accepted; 20 mutants, 2 accepted
        m = judge_metrics({"pos": 10, "pos_accepted": 8, "neg": 20, "neg_accepted": 2})
        self.assertEqual((m["tp"], m["fn"], m["fp"], m["tn"]), (8, 2, 2, 18))
        self.assertEqual(m["accuracy"], 86.67)
        self.assertEqual(m["sensitivity"], 80.0)      # = 100 - false rejection
        self.assertEqual(m["specificity"], 90.0)      # = 100 - false acceptance
        self.assertEqual(m["precision"], 80.0)
        self.assertEqual(m["f1"], 80.0)
        reject_all = judge_metrics({"pos": 5, "pos_accepted": 0, "neg": 5, "neg_accepted": 0})
        self.assertEqual((reject_all["sensitivity"], reject_all["specificity"], reject_all["f1"]), (0.0, 100.0, 0.0))


class VerifierBenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solver = Solver(20)

    @classmethod
    def tearDownClass(cls):
        cls.solver.close()

    def contexts(self, public, private):
        return [public["facts"]] + [c["facts"] for c in public["feedback_contexts"]] + \
               [c["facts"] for c in private["scoring_contexts"]]

    def test_rewrites_are_equivalent_and_mutants_mostly_differ(self):
        ctx = self.contexts(PUBLIC, PRIVATE)
        probes = probe_contexts(ctx, 20, random.Random(1), GOLD)
        self.assertTrue(any("6)" in p for p in probes))   # threshold 5 widened to 4..6
        bench = Bench(self.solver, PUBLIC, GOLD)
        ops = set()
        for op, program in positives(GOLD, PUBLIC["targets"]):
            ops.add(op)
            self.assertTrue(all(bench.same(program, f) for f in ctx + probes), (op, program))
        self.assertTrue({"rename_aux", "reorder_rules", "reorder_body", "split_rule", "flip_comparison"} <= ops)
        mutants = dict(negatives(GOLD, PUBLIC))
        self.assertFalse(all(bench.same(mutants["toggle_negation"], f) for f in ctx))
        boundary = "transport(1).dispatch(3).road(1,3,6)."
        self.assertFalse(bench.same(mutants["shift_threshold"], boundary))
        self.assertTrue(bench.same(mutants["shift_threshold"], PUBLIC["facts"]))   # invisible on F0

    def test_weak_constraint_rewrite_and_hardcoded_optimum(self):
        public = {**OPT_PUBLIC, "instance_expected": {"optimal": {"models": [["admit(2)"]]}}}
        bench = Bench(self.solver, public, OPT_GOLD)
        rewrite = dict(positives(OPT_GOLD, public["targets"]))["weak_to_minimize"]
        self.assertIn("#minimize{", rewrite)
        for facts in ("h(1..2).", "h(1..3).", "h(2;5)."):
            self.assertTrue(bench.same(rewrite, facts))
        hard = dict(negatives(OPT_GOLD, public))["hardcode_optimum"]
        self.assertEqual(hard, "admit(2).")
        self.assertTrue(bench.same(hard, "h(1..2).", phases=["optimal"]))   # optimal-only judge is fooled
        self.assertFalse(bench.same(hard, "h(1..2)."))                     # two-phase judge is not


if __name__ == "__main__":
    unittest.main()
