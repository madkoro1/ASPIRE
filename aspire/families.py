"""Task families evaluated in the paper (Table 1) and their fixed difficulty groups.

Difficulty is assigned per family, not per sample: L1 basic components are easy,
L2 two-component dependent compositions are medium, L3 compositions are hard.
The three single-construct L1 decision controls exist in the generator but are
excluded from the paper's evaluation.
"""
EASY = (
    "track_a_l1_filtering",
    "track_a_l1_negative_filtering",
    "track_a_l1_join",
    "track_a_l1_closure",
)
MID = (
    "track_a_l2_join_negative_filter",
    "track_a_l2_closure_triage",
    "track_a_l2_combination_negative_filter",
    "track_a_l2_numeric_pipeline",
)
HARD = (
    "track_a_l3_closure_triage_blocked",
    "track_a_l3_join_status_blocked",
    "track_a_l3_combination_distance_blocked",
    "track_a_l3_seen_admission",
    "track_a_l3_length_matched_control",
)
EXCLUDED = (
    "track_a_l1_choice",
    "track_a_l1_capacity_constraint",
    "track_a_l1_fixed_weak_priority",
)
PAPER_FAMILIES = EASY + MID + HARD
GROUPS = ("easy", "mid", "hard")


def difficulty(task):
    task = task.lower()
    if task in EASY:
        return "easy"
    if task in MID:
        return "mid"
    if task in HARD:
        return "hard"
    return "excluded" if task in EXCLUDED else "other"


def select(rows, families="paper"):
    """Keep paper families ('paper') or all rows ('all')."""
    if families == "all":
        return list(rows)
    if families != "paper":
        raise ValueError(f"unknown family selection: {families}")
    return [r for r in rows if r["task"] in PAPER_FAMILIES]
