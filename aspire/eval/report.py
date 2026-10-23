"""Aggregate completed runs into the paper's tables (Tables 2-6).

Per run and per group (easy / mid / hard / overall):
  strict      R(P)=1: F0 and every held-out context pass (Eq. 4)
  held_out    every held-out context passes
  visible     s(P) = |C_vis|
  OF          among visible-pass samples, share with R(P)=0 (Eq. 7), with numerator/denominator
  fixed / regressed relative to the run's own initial candidates (strict criterion)
Costs: LLM calls and tokens per role, invalid-feedback counts by reason,
revision format failures, solver timeouts. Paired comparisons between runs on
identical sample sets use exact McNemar and a bootstrap 95% CI, overall and per
difficulty group; --reference may be repeated (e.g. ast_fim/base and ast_fim/aspire).
"""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import random

from aspire.common import digest, file_hash, read_jsonl, write_json
from aspire.families import GROUPS, difficulty


def mcnemar_exact_p(fixed, regressed):
    n = fixed + regressed
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(fixed, regressed) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(reference, contender, seed=56, draws=2000):
    """Paired strict-pass comparison of two runs over identical sample IDs."""
    if set(reference) != set(contender):
        raise ValueError("paired runs must contain identical sample IDs")
    ids = sorted(reference)
    deltas = [int(contender[i]) - int(reference[i]) for i in ids]
    fixed, regressed = deltas.count(1), deltas.count(-1)
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(deltas, k=len(ids))) / len(ids) for _ in range(draws))
    lo, hi = means[int(0.025 * draws)], means[int(0.975 * draws) - 1]
    return {"n": len(ids), "fixed": fixed, "regressed": regressed, "delta": (fixed - regressed) / len(ids),
            "ci95_low": lo, "ci95_high": hi, "mcnemar_p": mcnemar_exact_p(fixed, regressed)}


def group_metrics(rows):
    n = len(rows)
    strict = sum(r["end_to_end_correct"] for r in rows)
    vis = [r for r in rows if r["visible_score"] == r["visible_total"]]
    of_num = sum(not r["end_to_end_correct"] for r in vis)
    fixed = sum(r["end_to_end_correct"] and not r["initial_result"]["end_to_end_correct"] for r in rows)
    regressed = sum(not r["end_to_end_correct"] and r["initial_result"]["end_to_end_correct"] for r in rows)
    pct = lambda x: round(100 * x / n, 2) if n else None
    return {"n": n, "strict": pct(strict), "initial_strict": pct(sum(r["initial_result"]["end_to_end_correct"] for r in rows)),
            "held_out": pct(sum(r["context_correct"] for r in rows)), "visible": pct(len(vis)),
            "OF": round(100 * of_num / len(vis), 2) if vis else None, "OF_num": of_num, "OF_den": len(vis),
            "fixed": fixed, "regressed": regressed,
            "timeouts": sum(bool(r.get("timed_out")) for r in rows)}


def trace_costs(rows, usage):
    reasons, rounds, fmt = Counter(), 0, 0
    feedback_calls, retries, rejected_rounds = 0, 0, 0
    for r in rows:
        for t in r["refinement_trace"]:
            if "feedback_raw" in t or "tester_raw" in t:
                feedback_calls += 1
            if "feedback_retry_raw" in t:  # retry after a rejected first answer
                feedback_calls += 1
                retries += 1
                reasons[t["first_feedback_rejected"]] += 1
            if t.get("feedback_rejected"):
                reasons[t["feedback_rejected"]] += 1
                rejected_rounds += 1
            if "raw" in t:
                rounds += 1
                fmt += bool(t.get("format_failure"))
    n = len(rows) or 1
    out = {"samples": len(rows),
           "generator_calls_per_sample": round(usage["generator"]["calls"] / n, 3),
           "feedback_calls_per_sample": round(usage["feedback"]["calls"] / n, 3),
           "tokens_per_sample": round(sum(usage[k]["prompt_tokens"] + usage[k]["completion_tokens"]
                                          for k in ("generator", "feedback")) / n, 1),
           "llm_seconds_per_sample": round(sum(usage[k]["seconds"] for k in ("generator", "feedback")) / n, 3),
           "solver_seconds_per_sample": round(sum(r["inference_solver"]["solver_seconds"] for r in rows) / n, 3),
           "invalid_feedback": sum(reasons.values()),
           "invalid_feedback_rate": round(sum(reasons.values()) / feedback_calls, 4) if feedback_calls else None,
           "feedback_retries": retries,
           "rounds_without_valid_feedback": rejected_rounds,
           "stop_feedback_exhausted": sum(any(t.get("stop") == "feedback_exhausted" for t in r["refinement_trace"])
                                          for r in rows),
           "revisions": rounds, "format_failures": fmt,
           "format_failure_rate": round(fmt / rounds, 4) if rounds else None}
    out.update({f"invalid_{k}": v for k, v in reasons.items()})
    return out


def load_runs(root):
    runs, incomplete = [], []
    for path in sorted(Path(root).glob("**/run_config.json")):
        folder = path.parent
        if not (folder / "COMPLETE.json").exists():
            incomplete.append(str(folder))
            continue
        config = json.loads(path.read_text(encoding="utf-8"))
        complete = json.loads((folder / "COMPLETE.json").read_text(encoding="utf-8"))
        if complete["results_sha256"] != file_hash(folder / "protocol_samples.jsonl") or complete["config_sha256"] != digest(config):
            raise ValueError(f"changed completed run: {folder}")
        rows = read_jsonl(folder / "protocol_samples.jsonl")
        if sorted(r["sample_id"] for r in rows) != sorted(config["samples"]):
            raise ValueError(f"missing/duplicate samples: {folder}")
        runs.append({"label": str(folder.relative_to(root)).replace("\\", "/"), "config": config, "rows": rows,
                     "usage": json.loads((folder / "usage.json").read_text(encoding="utf-8"))})
    return runs, incomplete


def overlap_report(runs, a, out):
    """Metrics without test samples whose exact reference program occurs in training (Sec. 3.1)."""
    import pandas as pd
    from aspire.data.suite import load_suite
    if not a.suite:
        raise ValueError("--overlap-train needs --suite")
    train_programs = set(pd.read_csv(a.overlap_train, keep_default_na=False)["answer"])
    _, _, private = load_suite(a.suite, a.split)
    excluded = {sid for sid, p in private.items() if p["gold_program"] in train_programs}
    per_task = Counter()
    rows_out = []
    for run in runs:
        kept = [r for r in run["rows"] if r["sample_id"] not in excluded]
        if run is runs[0]:
            per_task.update(r["task"] for r in run["rows"] if r["sample_id"] in excluded)
        for group in GROUPS + ("overall",):
            subset = [r for r in kept if group == "overall" or difficulty(r["task"]) == group]
            rows_out.append({"run": run["label"], "group": group, **group_metrics(subset)})
    save_csv(out / "accuracy_no_overlap.csv", rows_out)
    write_json(out / "overlap_excluded.json", {"excluded": len(excluded), "per_task": dict(per_task),
                                               "sample_ids": sorted(excluded)})
    print(f"overlap: {len(excluded)} test samples excluded {dict(per_task)}")


def save_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, help="directory containing completed run directories")
    p.add_argument("--output", required=True)
    p.add_argument("--reference", action="append", default=[],
                   help="label of a run every other run is paired against (repeatable)")
    p.add_argument("--overlap-train", default="",
                   help="training CSV; also report metrics without test samples whose reference "
                        "program text occurs in training (needs --suite)")
    p.add_argument("--suite", default="", help="frozen suite (for --overlap-train)")
    p.add_argument("--split", default="iid")
    a = p.parse_args(argv)
    runs, incomplete = load_runs(a.root)
    if not runs:
        raise ValueError("no completed runs found")
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    if a.overlap_train:
        overlap_report(runs, a, out)
    table, costs, pairs = [], [], []
    for run in runs:
        c = run["config"]
        shared = {"run": run["label"], "method": c["method"], "split": c["split"],
                  "seed": c["seed"], "selection": c["selection"], "visible_set": c["visible"], "keep": c["keep"],
                  "verify_projection": c.get("verify_projection", "on"), "verify_phases": c.get("verify_phases", "both"),
                  "grammar": c.get("grammar", "none")}
        for group in GROUPS + ("overall",):
            subset = [r for r in run["rows"] if group == "overall" or difficulty(r["task"]) == group]
            table.append({**shared, "group": group, **group_metrics(subset)})
        costs.append({**shared, **trace_costs(run["rows"], run["usage"])})
    by_label = {r["label"]: r for r in runs}

    def group_pairs(reference, contender, ref_label, con_label):
        """Paired comparison overall and per difficulty group on shared sample IDs."""
        tasks = {r["sample_id"]: r["task"] for r in contender["rows"]}
        for group in ("overall",) + GROUPS:
            keep = {i for i in tasks if group == "overall" or difficulty(tasks[i]) == group}
            ref = {i: v for i, v in reference.items() if i in keep}
            con = {i: v for i, v in contender_strict(contender).items() if i in keep}
            if ref and set(ref) == set(con):
                pairs.append({"reference": ref_label, "contender": con_label, "group": group, **paired(ref, con)})

    def contender_strict(run):
        return {r["sample_id"]: r["end_to_end_correct"] for r in run["rows"]}

    for run in runs:
        group_pairs({r["sample_id"]: r["initial_result"]["end_to_end_correct"] for r in run["rows"]},
                    run, "own_initial", run["label"])
    for label in a.reference:
        ref = by_label[label]
        ref_strict = contender_strict(ref)
        for run in runs:
            if run is not ref and set(contender_strict(run)) == set(ref_strict):
                group_pairs(ref_strict, run, label, run["label"])
    save_csv(out / "accuracy.csv", table)
    save_csv(out / "costs.csv", costs)
    save_csv(out / "paired.csv", pairs)
    write_json(out / "incomplete_runs.json", incomplete)
    lines = ["# ASPIRE results (strict pass rate, %)", "",
             "| Run | Easy | Mid | Hard | Overall | Held-out | Visible | OF (num/den) |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for run in runs:
        g = {r["group"]: r for r in table if r["run"] == run["label"]}
        o = g["overall"]
        lines.append(f"| {run['label']} | {g['easy']['strict']} | {g['mid']['strict']} | {g['hard']['strict']} | "
                     f"{o['strict']} | {o['held_out']} | {o['visible']} | {o['OF']} ({o['OF_num']}/{o['OF_den']}) |")
    lines += ["", f"Incomplete runs excluded: {len(incomplete)}."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
