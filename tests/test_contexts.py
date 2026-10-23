"""The five formerly id-shifted families must expose typical wrong programs (Sec. 2.4)."""
import json
import random
import re
import unittest

from aspire.data.freeze import convert
from aspire.data.generate import TEST_VOCAB, TRACK_A_SPECS, test_rows
from aspire.verify.oracle import Solver, VisibleOracle, evaluate

FAMILIES = ("track_a_l1_negative_filtering", "track_a_l1_join", "track_a_l2_join_negative_filter",
            "track_a_l2_combination_negative_filter", "track_a_l3_join_status_blocked")


def mutations(gold):
    """Wrong variants: drop each body literal, flip negated variables, parity filters, ignore join keys."""
    rules = gold.splitlines()
    out = set()
    for i, rule in enumerate(rules):
        head, body = rule.rstrip(".").split(":-")
        literals = re.findall(r'(?:not )?[a-z_]+\([^()]*\)', body)
        for lit in literals:
            if len(literals) > 1:
                rest = [l for l in literals if l != lit]
                out.add("\n".join(rules[:i] + [f"{head}:-{','.join(rest)}."] + rules[i + 1:]))
            m = re.fullmatch(r"not ([a-z_]+)\(([A-Z])\)", lit)
            if m:
                for var in "XH":
                    if var != m.group(2) and var in head:
                        out.add(gold.replace(lit, f"not {m.group(1)}({var})"))
                for parity in ("1", "0"):
                    out.add(gold.replace(lit, f"{m.group(2)}\\2={parity}"))
            m = re.fullmatch(r"([a-z_]+)\(H,Z\)", lit)
            if m:
                out.add(gold.replace(lit, f"{m.group(1)}(H,W)"))     # key ignored
                out.add(gold.replace(lit, f"{m.group(1)}(Z,H)"))     # arguments swapped
            m = re.fullmatch(r'([a-z_]+)\(X,"[^"]*"\)', lit)
            if m:
                out.add(gold.replace(lit, f"{m.group(1)}(X,_)"))     # any label accepted
    out.discard(gold)
    return sorted(out)


class ContextDiscriminationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solver = Solver(20)
        specs = [s for s in TRACK_A_SPECS if s.name in FAMILIES]
        cls.rows = test_rows(specs, 3, 20260719, TEST_VOCAB, "track_a", "iid")

    @classmethod
    def tearDownClass(cls):
        cls.solver.close()

    def test_mutants_fail_on_feedback_and_held_out(self):
        self.assertEqual({r["task"] for r in self.rows}, set(FAMILIES))
        for index, row in enumerate(self.rows):
            self.assertNotIn("id_renaming", row["hidden_contexts"])
            public, private = convert(row, index, "iid", "test", 3)
            gold = private["gold_program"]
            oracle = VisibleOracle(self.solver, public, gold)
            self.assertEqual(oracle.score(gold), len(oracle.contexts))
            for wrong in mutations(gold):
                checks = oracle.check(wrong)
                held = evaluate(self.solver, public, private, wrong)["contexts"]
                with self.subTest(task=row["task"], wrong=wrong):
                    self.assertFalse(all(c["correct"] for c in checks[1:]), "no feedback context fails")
                    self.assertFalse(all(c["correct"] for c in held), "no held-out context fails")

    def test_parity_filter_counterexample_is_caught(self):
        row = next(r for r in self.rows if r["task"] == "track_a_l1_negative_filtering")
        public, private = convert(row, 0, "iid", "test", 3)
        entity = re.match(r"available\(X\):-([a-z_]+)\(X\)", private["gold_program"]).group(1)
        wrong = f"available(X):-{entity}(X),X\\2=1."
        self.assertFalse(evaluate(self.solver, public, private, wrong)["end_to_end_correct"])


if __name__ == "__main__":
    unittest.main()
