"""Freeze generated test CSVs into a public/private evaluation suite.

Public rows carry the question, F0, interface, background and feedback contexts.
Private rows carry the held-out scoring contexts and, for audit only, the
reference program. Every context (F0 included) stores its expected output,
computed offline from the reference program; the runner verifies against these
expected outputs and never executes the reference.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import pandas as pd

from aspire.common import append_jsonl, digest, file_hash, write_json
from aspire.families import PAPER_FAMILIES
from aspire.verify.extraction import split_asp_statements
from aspire.verify.oracle import Solver, expected_record, full_program, phases, visible_contexts
from aspire.verify.solver import (parse_context_family, parse_optimization_flag,
                                  parse_target_predicates, split_refinement_contexts)


def _missing(value):
    return value is None or (isinstance(value, float) and pd.isna(value)) or (
        isinstance(value, str) and not value.strip())


def _metadata_for_row(row):
    targets = parse_target_predicates(row.get("target_predicates"))
    optimization = parse_optimization_flag(row.get("optimization"), str(row["answer"]))
    raw = row.get("hidden_contexts")
    all_contexts = [] if _missing(raw) else parse_context_family(raw)
    background = "" if _missing(row.get("background_asp")) else str(row["background_asp"])
    return targets, optimization, all_contexts, background


def canonical_facts(facts):
    # Ground fact order is immaterial. Preserve quoted strings and ranges.
    return digest(sorted(s.strip() for s in split_asp_statements(facts)))


def convert(row, index, split, partition, n_feedback):
    targets, optimization, all_contexts, background = _metadata_for_row(row)
    if not targets or not all_contexts:
        raise ValueError(f"row {index}: declared targets and contexts are required")
    facts = str(row["fact"])
    # Materialize append semantics once, then remove duplicate facts, including
    # copies of the visible instance. Preserve first-occurrence order.
    unique, seen = [], {canonical_facts(facts)}
    for c in all_contexts:
        materialized = (facts + "\n" + c["facts"]) if c.get("append") else c["facts"]
        key = canonical_facts(materialized)
        if key not in seen:
            unique.append({"name": f"context_{len(unique)}", "facts": materialized})
            seen.add(key)
    if len(unique) <= n_feedback:
        raise ValueError(f"row {index}: need >{n_feedback} distinct contexts, got {len(unique)}")
    feedback, scoring = split_refinement_contexts(unique, n_feedback)
    task = str(row["task"])
    sid = digest([partition, split, index, task, str(row["prompt"]), facts])[:24]
    public = {"sample_id": sid, "sample_index": index, "task": task,
              "question": str(row["prompt"]), "facts": facts, "targets": targets,
              "optimization": optimization, "background_asp": background,
              "feedback_contexts": feedback, "partition": partition, "split": split,
              "level": int(row.get("level", 0)),
              "control_type": str(row.get("control_type", "")),
              "composition_split": str(row.get("composition_split", "unknown")),
              "context_count_original": len(all_contexts),
              "context_duplicates_removed": len(all_contexts) - len(unique)}
    private = {"sample_id": sid, "gold_program": str(row["answer"]), "scoring_contexts": scoring}
    return public, private


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iid", required=True)
    p.add_argument("--ood", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--feedback-contexts", type=int, default=3)
    p.add_argument("--clingo-timeout", type=float, default=20)
    p.add_argument("--families", choices=("paper", "all"), default="paper",
                   help="paper: the 13 easy/mid/hard families of Table 1")
    a = p.parse_args(argv)
    if a.feedback_contexts < 1:
        p.error("--feedback-contexts must be positive")
    out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        p.error("output must be empty; frozen suites are never overwritten")
    frames = {s: pd.read_csv(getattr(a, s), keep_default_na=False) for s in ("iid", "ood")}
    if a.families == "paper":
        # Keep the source row index so sample IDs do not depend on the filter.
        frames = {s: f[f["task"].isin(PAPER_FAMILIES)] for s, f in frames.items()}
    out.mkdir(parents=True, exist_ok=True)
    solver, counts, duplicates = Solver(a.clingo_timeout), {}, 0
    try:
        for partition, family in (("test", frames),):
            for split, frame in family.items():
                prefix = f"{partition}_{split}"
                counts[prefix] = dict(Counter(frame["task"]))
                for i, row in frame.iterrows():
                    public, private = convert(row, int(i), split, partition, a.feedback_contexts)
                    duplicates += public["context_duplicates_removed"]
                    # Expected outputs are computed offline here and frozen with each
                    # context; verification at run time compares against them only.
                    for c in visible_contexts(public) + private["scoring_contexts"]:
                        expected = {}
                        for phase in phases(public):
                            result = solver.solve(full_program(public, private["gold_program"]),
                                                  c["facts"], public["targets"], phase)
                            if not result["parse_ground"] or not result["terminated"]:
                                raise ValueError(f"reference failed: {prefix}/{i}/{c['name']}: {result['error']}")
                            expected[phase] = expected_record(result)
                        if c["name"] == "instance":
                            public["instance_expected"] = expected
                        else:
                            c["expected"] = expected
                    append_jsonl(out / f"{prefix}.public.jsonl", public)
                    append_jsonl(out / f"{prefix}.private.jsonl", private)
                    solver.cache.clear()
                    if (i + 1) % 100 == 0:
                        print(f"[freeze] {prefix} {i + 1}/{len(frame)}", flush=True)
        files = {p.name: file_hash(p) for p in sorted(out.glob("*.jsonl"))}
        manifest = {"schema_version": 2, "files": files, "counts": counts,
                    "source_csv_sha256": {s: file_hash(getattr(a, s)) for s in frames},
                    "feedback_contexts": a.feedback_contexts, "families": a.families,
                    "duplicates_removed": duplicates,
                    "expected_outputs": "frozen per context (satisfiable, feasible/optimal projections, costs)",
                    "protocol": "bounded context projected equivalence; feasible AND optimal; costs auxiliary",
                    "compatibility": "schema 2: discriminative contexts for 5 formerly id-shifted families; "
                                     "do not mix percentages with schema-1 suites"}
        manifest["suite_id"] = digest(manifest)
        write_json(out / "manifest.json", manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    finally:
        solver.close()


if __name__ == "__main__":
    main()
