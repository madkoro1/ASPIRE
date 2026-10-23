"""Domain vocabulary for the disaster medical-rescue scenario.

This module does NOT change the LLASP method (the per-pattern generator
functions in dataset_generation_extension.py are reused as-is). It only supplies
the vocabulary that gets plugged into those generators, drawn from the
earthquake medical-rescue scenario (see data/问题整理.md).

Design decisions:
  * TRAIN and TEST vocabularies are DISJOINT (original LLASP protocol): the
    model is evaluated on words it never saw in fine-tuning, so accuracy
    reflects learning the ASP *pattern*, not memorizing words.
  * PREDICATE-NAME words and LABEL (value) words are kept in SEPARATE, disjoint
    roles. A word is either an entity/set used as a predicate, OR an attribute
    value used as a quoted/constant label -- never both. This removes the
    confound seen in the first scenario run, where value-like nouns
    (speed/capacity/central) sat in the predicate pool and the model, having
    seen them as `speed(X,C)` etc., emitted preference labels as predicates
    with arguments (e.g. `:~assign(4,speed(1,1))` instead of
    `:~assign(4,"speed").[1@1]`).

All names are valid clingo identifiers (lowercase ASCII, single token, not ASP
keywords).
"""

# --- PREDICATE-NAME pools (entities / sets you quantify over) ---------------
TRAIN_PREDICATES = [
    "injured", "triage", "township", "hospital", "ambulance", "helicopter",
    "team", "bed", "road", "transport", "treatment", "rescue", "stretcher",
    "crew", "dispatch", "depot",
]
TEST_PREDICATES = [
    "casualty", "clinic", "ward", "truck", "chopper", "convoy", "litter",
    "carrier", "shelter", "staff", "cot", "outpost", "station", "camp",
    "medic", "driver",
]

# --- LABEL pools (attribute VALUES: severity / posture / scope / state) ------
# Deliberately value-like words that never appear as predicate names.
TRAIN_LABELS = [
    "critical", "serious", "mild", "lying", "sitting", "urgent",
    "daytime", "nighttime", "stable", "delayed",
]
TEST_LABELS = [
    "severe", "moderate", "minor", "supine", "seated", "priority",
    "morning", "evening", "worsening", "immediate",
]

# --- attribute names (join field markers) -----------------------------------
TRAIN_ATTRIBUTES = ["id", "time", "location", "level", "window", "day"]
TEST_ATTRIBUTES = ["code", "moment", "place", "grade", "span", "date"]

# --- transitive-closure names (road-network reachability theme) -------------
TRAIN_CLOSURES = ["reach", "connected", "reachable", "linked"]
TEST_CLOSURES = ["route", "path", "network", "access"]


def _all_pools():
    return {
        "TRAIN_PREDICATES": TRAIN_PREDICATES, "TEST_PREDICATES": TEST_PREDICATES,
        "TRAIN_LABELS": TRAIN_LABELS, "TEST_LABELS": TEST_LABELS,
        "TRAIN_ATTRIBUTES": TRAIN_ATTRIBUTES, "TEST_ATTRIBUTES": TEST_ATTRIBUTES,
        "TRAIN_CLOSURES": TRAIN_CLOSURES, "TEST_CLOSURES": TEST_CLOSURES,
    }


def assert_clean():
    """Guarantee the key invariants: train/test disjoint, and (crucially) no
    word is ever both a predicate name and a label."""
    preds = set(TRAIN_PREDICATES) | set(TEST_PREDICATES)
    labs = set(TRAIN_LABELS) | set(TEST_LABELS)

    assert not (set(TRAIN_PREDICATES) & set(TEST_PREDICATES)), "train/test predicate overlap"
    assert not (set(TRAIN_LABELS) & set(TEST_LABELS)), "train/test label overlap"
    assert not (preds & labs), f"predicate/label overlap: {preds & labs}"

    # every pool must contain only valid clingo identifiers, and have no dups
    import re
    ident = re.compile(r"[a-z][a-zA-Z0-9_]*$")
    for name, pool in _all_pools().items():
        assert len(pool) == len(set(pool)), f"duplicates in {name}"
        for w in pool:
            assert ident.match(w), f"invalid identifier {w!r} in {name}"


def train_pools():
    """(predicates, labels, attributes, closures) for TRAIN generation."""
    return TRAIN_PREDICATES, TRAIN_LABELS, TRAIN_ATTRIBUTES, TRAIN_CLOSURES


def test_pools():
    """(predicates, labels, attributes, closures) for TEST generation."""
    return TEST_PREDICATES, TEST_LABELS, TEST_ATTRIBUTES, TEST_CLOSURES


if __name__ == "__main__":
    assert_clean()
    print("vocab OK")
    print(f"train predicates={len(TRAIN_PREDICATES)} labels={len(TRAIN_LABELS)}")
    print(f"test  predicates={len(TEST_PREDICATES)} labels={len(TEST_LABELS)}")
