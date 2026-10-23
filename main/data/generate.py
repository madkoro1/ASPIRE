"""Generate structure-controlled compositional ASP datasets.

The original LLASP core data teaches mostly isolated templates.  This module
keeps every atomic construct visible during training, adds a selected set of
coherent two-component programs, and holds out different pair/triple
compositions for testing.  The split is therefore compositional: a held-out
test program contains only familiar local constructs, but their *combination*
was never used as a training target.

Four test sets form a 2x2 diagnostic design:

    composition: seen / held out
    vocabulary:  training (IID) / disjoint test (OOD)

Every row also records its components, rule/dependency-graph features,
dependency edges, and expected rule count.  The existing training script
ignores these extra columns; model_testing_extension.py consumes ``task`` and
``rule_count`` when they are present.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import pandas as pd

from aspire.data import vocab as sv


FEATURES = {
    "positive_dependency",
    "negative_dependency",
    "shared_variable",
    "recursive_scc",
    "base_recursive",
    "choice_rule",
    "weak_constraint",
    "aggregate",
    "comparison",
}


@dataclass(frozen=True)
class Vocabulary:
    predicates: tuple[str, ...]
    labels: tuple[str, ...]
    attributes: tuple[str, ...]
    closures: tuple[str, ...]


@dataclass(frozen=True)
class Example:
    question: str
    answer: str
    facts: str
    target_predicates: tuple[tuple[str, int], ...] = ()
    hidden_contexts: tuple[dict[str, Any], ...] = ()
    # Fixed program prepended to BOTH the reference and the candidate before
    # solving.  Isolated-construct controls use it to supply the choice space so
    # the model only has to write the construct under test: a weak constraint on
    # its own cannot create alternatives, so without a background choice the
    # control would be vacuous (deleting the weak constraint would not change
    # the answer).  Empty when the model must produce the whole program.
    background: str = ""


Builder = Callable[[random.Random, Vocabulary], Example]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    components: tuple[str, ...]
    features: tuple[str, ...]
    dependency_edges: tuple[str, ...]
    builder: Builder
    kind: str  # atomic, seen_composite, heldout_composite
    track: str = "auto"
    level: int = 0
    control_type: str = "auto"
    target_predicates: tuple[tuple[str, int], ...] = ()
    optimization: bool | None = None
    context_family: str = ""
    pattern: str = ""

    @property
    def rule_count(self) -> int:
        # ASP weak constraints are rules too; all builders emit one rule/line.
        return len(self.builder(random.Random(0), TRAIN_VOCAB).answer.splitlines())


def _vocab(train: bool) -> Vocabulary:
    pools = sv.train_pools() if train else sv.test_pools()
    return Vocabulary(*(tuple(pool) for pool in pools))


TRAIN_VOCAB = _vocab(True)
TEST_VOCAB = _vocab(False)


def _pick(rng: random.Random, values: Sequence[str], n: int = 1) -> list[str]:
    return rng.sample(list(values), n)


def _preamble(text: str) -> str:
    return f"Write an ASP program for the following medical-rescue problem. {text}"


def _label_facts(labels: Iterable[str]) -> str:
    return "".join(f'triage_label("{label}").' for label in labels)


def _facts(predicate: str, tuples: Iterable[Sequence[Any]]) -> str:
    """Render ground facts while keeping quoted labels explicit.

    Strings already surrounded by double quotes are preserved; other strings
    are treated as ASP constants.  Builders use this only for controlled
    numeric/string test contexts, not arbitrary user input.
    """
    rendered = []
    for values in tuples:
        terms = ",".join(str(value) for value in values)
        rendered.append(f"{predicate}({terms}).")
    return "".join(rendered)


def _context(context_id: str, facts: str, *tags: str) -> dict[str, Any]:
    return {"id": context_id, "facts": facts, "tags": list(tags)}


def _contexts(items: Iterable[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    result = tuple(items)
    if not 10 <= len(result) <= 20:
        raise ValueError(f"hidden context suites must contain 10-20 contexts, got {len(result)}")
    return result


def _shift_numeric_constants(facts: str, offset: int) -> str:
    """Produce a semantics-preserving-ish renaming fallback for legacy rows.

    Track-specific builders below provide stronger counterfactual suites.
    This fallback makes every legacy evaluation row expose explicit contexts
    without changing its builder interface.  Uniform positive shifts preserve
    ranges and relative numeric order.
    """
    if offset == 0:
        return facts

    def replace(match: re.Match[str]) -> str:
        value = int(match.group(0))
        return str(value + offset) if value >= 0 else str(value - offset)

    return re.sub(r"(?<![A-Za-z0-9_\"])-?\d+(?![A-Za-z0-9_\"])", replace, facts)


def _default_hidden_contexts(facts: str) -> tuple[dict[str, Any], ...]:
    return _contexts(
        _context(
            f"id_renaming_{index:02d}",
            _shift_numeric_constants(facts, offset),
            "id_renaming",
            "legacy_context",
        )
        for index, offset in enumerate((0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110))
    )


# Discriminative suites for the five Track A families that previously used
# ``_default_hidden_contexts``.  Uniform id shifts keep the blocked set and the
# join pattern of F0, so e.g. a parity filter X\2=1 passes every shifted copy
# of crew(1..4).treatment(2).treatment(4).  These suites vary which ids are
# blocked (none, all, odd, even, ids outside the entity set, ids on the wrong
# argument), the join pattern (no match, one-to-many, many-to-many, keys that
# collide with entity ids) and the status labels, so each listed mutation of
# the reference fails on at least one feedback AND one held-out context
# (tests/test_contexts.py).


def _negative_filter_contexts(entity: str, blocked: str) -> tuple[dict[str, Any], ...]:
    suites = (
        ((1, 2, 3, 4), (1,)),            # an odd id blocked: defeats parity filters
        ((1, 2, 3), ()),                 # empty blocked set
        ((2, 4, 6), (2, 4, 6)),          # every entity blocked
        ((1, 2, 3, 4, 5), (1, 3, 5)),
        ((1, 2), (7,)),                  # blocked id outside the entity set
        ((5,), ()),
        ((3, 4, 5, 6), (4, 5)),
        ((1, 3, 5), (3,)),
        ((2, 7, 9), (2, 9)),
        ((10, 11, 12), (11, 13)),
        ((1, 2, 3, 4), (1, 2, 3, 4)),
        ((6, 8), (5,)),
    )
    return _contexts(
        _context(f"negative_filter_{i:02d}",
                 _facts(entity, ((e,) for e in ents)) + _facts(blocked, ((b,) for b in blk)),
                 "negative_filter", "blocked_set_variation",
                 *(["empty_blocked"] if not blk else []),
                 *(["all_blocked"] if set(ents) <= set(blk) else []))
        for i, (ents, blk) in enumerate(suites)
    )


_JOIN_SUITES = (
    # (left(X,Z), right(H,Z)); keys collide with ids so swapped arguments fail
    (((1, 10), (2, 10)), ((3, 10),)),                 # two X share one key
    (((1, 10),), ((3, 20),)),                         # no matching key
    (((1, 10),), ((3, 10), (4, 10))),                 # one-to-many
    (((1, 2), (2, 3)), ((3, 2), (4, 3))),             # keys equal to entity ids
    (((1, 10), (2, 20), (3, 30)), ((4, 20), (5, 30), (6, 40))),
    (((5, 5),), ((5, 5),)),                           # X = H = Z
    (((1, 10), (1, 20)), ((2, 10), (3, 20))),         # one X, two keys
    (((2, 7), (3, 8)), ((4, 8), (5, 7))),             # crossed keys
    (((1, 3),), ((3, 1),)),                           # swapped arguments do not match
    (((1, 10), (2, 10)), ((3, 10), (4, 10))),         # many-to-many
    (((6, 11),), ((7, 11), (8, 12))),
    (((9, 4), (4, 9)), ((4, 4), (9, 9))),
)


def _join_contexts(left: str, right: str) -> tuple[dict[str, Any], ...]:
    return _contexts(
        _context(f"join_{i:02d}", _facts(left, l) + _facts(right, r), "join", "join_pattern")
        for i, (l, r) in enumerate(_JOIN_SUITES)
    )


def _join_negative_contexts(left: str, right: str, blocked: str) -> tuple[dict[str, Any], ...]:
    # Blocked ids sometimes match X (left side) instead of H, to catch blocked(X).
    blocked_sets = ((3,), (), (4,), (1,), (5,), (5,), (2,), (4, 5), (1, 3), (3, 4), (8,), (9, 4))
    return _contexts(
        _context(f"join_negative_{i:02d}",
                 _facts(left, l) + _facts(right, r) + _facts(blocked, ((b,) for b in blk)),
                 "join", "negative_filter", *(["empty_blocked"] if not blk else []))
        for i, ((l, r), blk) in enumerate(zip(_JOIN_SUITES, blocked_sets))
    )


def _combination_negative_contexts(left: str, right: str, blocked: str) -> tuple[dict[str, Any], ...]:
    suites = (
        ((1, 2), (3, 4), (3,)),
        ((1, 2), (3, 4), ()),             # nothing blocked
        ((1,), (2, 3), (2, 3)),           # every H blocked
        ((1, 2), (3, 4), (1,)),           # blocked id is an X, not an H
        ((1, 2, 3), (1, 2, 3), (2,)),     # X and H ranges overlap
        ((5,), (6,), (7,)),               # blocked id unused
        ((1, 3), (5, 7, 9), (5, 9)),
        ((2,), (4, 6, 8), (6,)),
        ((1, 2), (2, 4), (2,)),           # one id is both an X and a blocked H
        ((7, 8), (9,), ()),
        ((1, 2, 3, 4), (5, 6), (6,)),
        ((4,), (4, 5), (4,)),
    )
    return _contexts(
        _context(f"combination_negative_{i:02d}",
                 _facts(left, ((x,) for x in xs)) + _facts(right, ((h,) for h in hs))
                 + _facts(blocked, ((b,) for b in blk)),
                 "combination", "negative_filter", *(["empty_blocked"] if not blk else []))
        for i, (xs, hs, blk) in enumerate(suites)
    )


def _join_status_blocked_contexts(left: str, right: str, status: str, blocked: str,
                                  label: str, other: str) -> tuple[dict[str, Any], ...]:
    # (left, right, status entries as (X, has_target_label), blocked H ids).
    # The first three (feedback) and the remaining nine (held-out) each expose:
    # any label accepted, blocked dropped, blocked(X), H parity, key ignored, swapped key.
    suites = (
        (((1, 10), (2, 10)), ((3, 10),), ((1, True), (2, False)), ()),
        (((1, 10),), ((3, 10), (4, 10)), ((1, True),), (3,)),
        (((1, 10), (2, 20)), ((3, 10), (4, 30)), ((1, True), (2, True)), (4,)),
        (((1, 10), (2, 10)), ((3, 10), (4, 10)), ((1, True), (2, False)), (4,)),
        (((1, 10), (2, 20)), ((3, 20), (4, 10)), ((1, True), (2, True)), (3,)),
        (((5, 7),), ((6, 7),), ((5, False),), ()),
        (((1, 3),), ((3, 1),), ((1, True), (3, True)), ()),
        (((2, 8), (3, 8)), ((4, 8),), ((2, True), (3, False)), (2,)),   # blocked id is an X
        (((1, 10),), ((2, 10),), ((1, True),), (2,)),
        (((4, 9), (6, 9)), ((5, 9), (7, 9)), ((4, True), (6, True)), (5,)),
        (((1, 10),), ((3, 20),), ((1, True),), ()),
        (((9, 4), (4, 9)), ((4, 4), (9, 9)), ((9, True), (4, False)), (4,)),
    )
    result = []
    for i, (l, r, st, blk) in enumerate(suites):
        facts = _facts(left, l) + _facts(right, r)
        facts += "".join(f'{status}({x},"{label if hit else other}").' for x, hit in st)
        facts += _facts(blocked, ((b,) for b in blk))
        result.append(_context(f"join_status_blocked_{i:02d}", facts, "join", "status_label", "negative_filter"))
    return _contexts(result)


def _network_contexts(
    edge: str,
    triage: str | None = None,
    blocked: str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Small graph contexts covering chains, branches, cycles, and unreachable sites."""
    topologies = (
        (((1, 2), (2, 3)), (2, 3, 9), ()),
        (((3, 1), (1, 2)), (2, 8), ()),
        (((7, 4), (4, 9), (9, 2)), (2, 6), (9,)),
        (((1, 2), (1, 3), (3, 4)), (2, 4, 8), (2,)),
        (((5, 6), (6, 5), (6, 7)), (5, 7, 9), ()),
        (((10, 20),), (20, 30), ()),
        (((2, 4), (8, 9)), (4, 9, 11), (9,)),
        (((9, 3), (3, 6), (9, 8), (8, 6)), (6, 12), (8,)),
        (((4, 1), (1, 7), (7, 11), (11, 13)), (7, 13, 20), (11,)),
        (((21, 5), (5, 14), (30, 31)), (14, 31, 40), (31,)),
        (((6, 2), (2, 9), (9, 12), (12, 6)), (9, 12, 18), (12,)),
        (((15, 3), (15, 8), (8, 10), (20, 21)), (3, 10, 21), (3,)),
    )
    result = []
    for index, (edges, triage_nodes, blocked_nodes) in enumerate(topologies):
        facts = _facts(edge, edges)
        if triage:
            facts += _facts(triage, ((node,) for node in triage_nodes))
        if blocked:
            facts += _facts(blocked, ((node,) for node in blocked_nodes))
        tags = ["closure_graph", "id_permutation"]
        if triage and any(node not in {v for pair in edges for v in pair} for node in triage_nodes):
            tags.append("unreachable_triage")
        if len(edges) >= 2:
            tags.append("multi_hop_chain")
        result.append(_context(f"network_{index:02d}", facts, *tags))
    return _contexts(result)


def _threshold_distance_contexts(
    entity: str,
    destination: str,
    distance: str,
    blocked: str | None = None,
    threshold: int = 5,
) -> tuple[dict[str, Any], ...]:
    """Contexts whose measured values STRADDLE the comparison threshold.

    ``_default_hidden_contexts`` shifts every numeric constant uniformly, which
    pushes all measured values past a fixed threshold baked into the programs:
    gold and any wrong-comparator candidate then agree on the empty/full
    selection and the context stops discriminating (ATTRIBUTION_AND_ROADMAP.md
    §2.5).  These suites keep ids varied while pinning distance values below,
    AT (to separate <= from <), and above the threshold.
    """
    below, at, above = threshold - 3, threshold, threshold + 3
    just_below, just_above = threshold - 1, threshold + 1
    suites = (
        ((1, 2), (3, 4), ((1, 3, below), (1, 4, above), (2, 3, at), (2, 4, just_above)), (4,)),
        ((1, 2), (3, 4), ((1, 3, at), (1, 4, just_below), (2, 3, just_above), (2, 4, above)), (3,)),
        ((5, 6), (7, 8), ((5, 7, just_below), (5, 8, at), (6, 7, above), (6, 8, below)), (8,)),
        ((1, 3), (2, 9), ((1, 2, above), (1, 9, above), (3, 2, above), (3, 9, above)), (9,)),
        ((1, 3), (2, 9), ((1, 2, below), (1, 9, below), (3, 2, below), (3, 9, below)), (2,)),
        ((4,), (6, 7), ((4, 6, at), (4, 7, at)), (7,)),
        ((10, 20), (30,), ((10, 30, just_above), (20, 30, just_below)), ()),
        ((2, 4, 6), (8,), ((2, 8, below), (4, 8, at), (6, 8, above)), (8,)),
        ((1,), (2, 3, 4), ((1, 2, just_below), (1, 3, just_above), (1, 4, at)), (3,)),
        ((9, 11), (12, 14), ((9, 12, at), (9, 14, below), (11, 12, above), (11, 14, just_below)), (12,)),
        ((1, 2), (3,), ((1, 3, below),), ()),
        ((7, 8), (5, 6), ((7, 5, above), (7, 6, just_above), (8, 5, just_below), (8, 6, below)), (5, 6)),
    )
    result = []
    for index, (entities, destinations, distances, blocked_nodes) in enumerate(suites):
        facts = _facts(entity, ((e,) for e in entities))
        facts += _facts(destination, ((d,) for d in destinations))
        facts += _facts(distance, distances)
        if blocked:
            facts += _facts(blocked, ((node,) for node in blocked_nodes))
        tags = ["threshold_straddle", "id_permutation"]
        if any(d[2] == threshold for d in distances):
            tags.append("boundary_equality")
        result.append(_context(f"threshold_{index:02d}", facts, *tags))
    return _contexts(result)


def _length_control_contexts(
    entity: str, status: str, label: str, left: str, right: str,
    blocked: str, measure: str, threshold: int = 5,
) -> tuple[dict[str, Any], ...]:
    """Threshold-aware contexts for the four independent length-control rules.

    Across the SUITE (not per context), every per-rule mutation — label drop,
    key-ignoring join, negation drop, and comparator off-by-ones — is caught by
    at least one context: label placement varies, join keys match or miss,
    blocked hits differ, and measure values straddle the threshold instead of
    being uniformly shifted past it.  Individual contexts need not
    discriminate every rule.
    """
    suites = (
        # (entities, labelled, join_left, join_right, blocked_ids, measures)
        ((1, 2, 3), (1,), ((1, 10),), ((3, 10),), (2,), ((1, 2), (2, 8))),
        ((1, 2, 3), (2,), ((1, 10),), ((3, 20),), (3,), ((1, threshold), (2, threshold + 1))),
        ((4, 5, 6), (4, 5), ((4, 7), (5, 8)), ((6, 7),), (5,), ((4, threshold - 1), (5, threshold + 4), (6, threshold))),
        ((1, 2), (), ((1, 9),), ((2, 9),), (1, 2), ((1, 8), (2, 9))),
        ((1, 2), (1, 2), ((1, 9), (2, 9)), ((1, 9), (2, 8)), (), ((1, threshold), (2, threshold + 1))),
        ((7, 8, 9), (9,), ((7, 30),), ((8, 30), (9, 31)), (7,), ((7, threshold), (8, threshold), (9, threshold))),
        ((10, 20), (10,), ((10, 5),), ((20, 6),), (20,), ((10, 2),)),
        ((1, 3, 5), (3,), ((1, 2), (3, 2)), ((5, 2),), (3,), ((1, 6), (3, 4), (5, 5))),
        ((2, 4), (2, 4), ((2, 11),), ((4, 11),), (4,), ((2, 7), (4, 3))),
        ((6,), (6,), ((6, 12),), ((6, 12),), (6,), ((6, threshold + 1),)),
        ((1, 2, 3, 4), (1, 4), ((2, 15),), ((3, 15),), (2, 3), ((1, 4), (2, 5), (3, 6), (4, 7))),
        ((5, 7), (), ((5, 20),), ((7, 21),), (), ((5, 3), (7, 9))),
    )
    result = []
    for index, (entities, labelled, join_left, join_right, blocked_ids, measures) in enumerate(suites):
        facts = _facts(entity, ((e,) for e in entities))
        facts += "".join(f'{status}({e},"{label}").' for e in labelled)
        facts += _facts(left, join_left)
        facts += _facts(right, join_right)
        facts += _facts(blocked, ((b,) for b in blocked_ids))
        facts += _facts(measure, measures)
        tags = ["length_control", "threshold_straddle"]
        if any(m[1] == threshold for m in measures):
            tags.append("boundary_equality")
        result.append(_context(f"length_control_{index:02d}", facts, *tags))
    return _contexts(result)


def _selection_contexts(
    entity: str,
    status: str,
    label: str,
    other_label: str,
) -> tuple[dict[str, Any], ...]:
    suites = (
        ((1, 2, 3), (1, 3), (2,)),
        ((3, 1, 2), (2,), (1, 3)),
        ((11, 4, 27), (4, 27), (11,)),
        ((8, 2, 5, 1), (8, 1), (2, 5)),
        ((9, 7), (7,), (9,)),
        ((20, 3, 14), (20,), (3, 14)),
        ((6, 12, 4), (6, 12, 4), ()),
        ((5, 15, 25), (), (5, 15, 25)),
        ((13, 2, 31, 7), (31, 7), (13, 2)),
        ((4, 10, 1), (10,), (1, 4)),
        ((22, 5, 17), (17, 22), (5,)),
        ((30, 9, 2, 18), (9, 18), (30, 2)),
    )
    result = []
    for index, (ids, selected, distractors) in enumerate(suites):
        facts = _facts(entity, ((value,) for value in ids))
        facts += _facts(status, ((value, f'"{label}"') for value in selected))
        facts += _facts(status, ((value, f'"{other_label}"') for value in distractors))
        result.append(_context(
            f"selection_{index:02d}", facts, "id_permutation", "label_distractor"
        ))
    return _contexts(result)


def _admission_contexts(
    entity: str,
    severity: str,
    capacity: str,
    priority: str,
    labels: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    high, medium, low = labels
    suites = (
        (((1, high), (2, medium), (3, low)), 1, ()),
        (((3, low), (1, medium), (2, high)), 1, ("id_permutation",)),
        (((11, medium), (4, low), (27, high)), 1, ("nonconsecutive_ids",)),
        (((8, high), (2, medium), (5, low)), 2, ("capacity_two",)),
        (((9, high), (7, medium), (5, low)), 0, ("capacity_zero",)),
        (((1, high), (3, high), (2, low)), 1, ("top_tie",)),
        (((4, high), (10, medium), (1, medium)), 2, ("boundary_tie",)),
        (((20, low), (3, high), (14, medium), (8, low)), 2, ("four_entities",)),
        (((6, low), (12, medium), (4, high)), 1, ("reversed_fact_order",)),
        (((5, medium), (15, medium), (25, medium)), 1, ("all_tie",)),
        (((13, low), (2, high)), 1, ("two_entities",)),
        (((30, high), (9, low), (2, high), (18, medium)), 2, ("top_tie_capacity_two",)),
    )
    result = []
    for index, (assignments, cap, extra_tags) in enumerate(suites):
        facts = _facts(entity, ((identifier,) for identifier, _ in assignments))
        facts += _facts(
            severity,
            ((identifier, f'"{severity_label}"') for identifier, severity_label in assignments),
        )
        facts += f"{capacity}({cap})."
        facts += (
            f'{priority}("{high}",3).'
            f'{priority}("{medium}",2).'
            f'{priority}("{low}",1).'
        )
        result.append(_context(
            f"admission_{index:02d}",
            facts,
            "id_permutation",
            "capacity_variation",
            *extra_tags,
        ))
    return _contexts(result)


def _dispatch_contexts(
    entity: str,
    resource: str,
    unavailable: str,
    travel: str,
) -> tuple[dict[str, Any], ...]:
    suites = (
        ((1, 2), (1, 2, 3), (2,), {(1, 1): 2, (3, 1): 4, (1, 2): 3, (3, 2): 1}),
        ((3, 1), (3, 1, 2), (1,), {(3, 3): 1, (2, 3): 4, (3, 1): 5, (2, 1): 2}),
        ((11, 4), (7, 2, 9), (2,), {(7, 11): 3, (9, 11): 1, (7, 4): 1, (9, 4): 4}),
        ((8, 2, 5), (1, 3, 6), (6,), {(1, 8): 1, (3, 8): 3, (1, 2): 2,
                                       (3, 2): 2, (1, 5): 4, (3, 5): 1}),
        ((9, 7), (4, 5, 6), (), {(4, 9): 1, (5, 9): 1, (6, 9): 3,
                                  (4, 7): 4, (5, 7): 2, (6, 7): 2}),
        ((20, 3), (10, 11, 12), (11,), {(10, 20): 6, (12, 20): 2,
                                        (10, 3): 1, (12, 3): 5}),
        ((6, 12), (2, 4, 8), (4,), {(2, 6): 2, (8, 6): 2, (2, 12): 3, (8, 12): 1}),
        ((5, 15, 25), (7, 9, 13), (13,), {(7, 5): 1, (9, 5): 4,
                                           (7, 15): 3, (9, 15): 2,
                                           (7, 25): 5, (9, 25): 1}),
        ((13, 2), (21, 22, 23), (22,), {(21, 13): 7, (23, 13): 2,
                                        (21, 2): 1, (23, 2): 6}),
        ((4, 10), (30, 20, 40), (40,), {(30, 4): 2, (20, 4): 4,
                                        (30, 10): 3, (20, 10): 1}),
        ((22, 5), (14, 18, 19), (18,), {(14, 22): 1, (19, 22): 1,
                                        (14, 5): 2, (19, 5): 3}),
        ((30, 9, 2), (5, 15, 25), (15,), {(5, 30): 4, (25, 30): 1,
                                           (5, 9): 2, (25, 9): 3,
                                           (5, 2): 5, (25, 2): 2}),
    )
    result = []
    for index, (entities, resources, unavailable_resources, costs) in enumerate(suites):
        facts = _facts(entity, ((value,) for value in entities))
        facts += _facts(resource, ((value,) for value in resources))
        facts += _facts(unavailable, ((value,) for value in unavailable_resources))
        facts += _facts(travel, ((a, x, cost) for (a, x), cost in costs.items()))
        tags = ["id_permutation", "multiple_feasible_resources", "variable_objective"]
        values = list(costs.values())
        if len(values) != len(set(values)):
            tags.append("objective_tie")
        result.append(_context(f"dispatch_{index:02d}", facts, *tags))
    return _contexts(result)


def _capacity_contexts(entity: str, capacity: str) -> tuple[dict[str, Any], ...]:
    suites = (
        ((1, 2, 3), 1),
        ((3, 1, 2), 2),
        ((11, 4, 27), 0),
        ((8, 2, 5, 1), 3),
        ((9, 7), 1),
        ((20, 3, 14), 2),
        ((6, 12, 4), 3),
        ((5, 15, 25), 0),
        ((13, 2, 31, 7), 2),
        ((4, 10, 1), 1),
        ((22, 5, 17), 4),
        ((30, 9, 2, 18), 1),
    )
    return _contexts(
        _context(
            f"capacity_{index:02d}",
            _facts(entity, ((value,) for value in ids)) + f"{capacity}({cap}).",
            "id_permutation",
            "capacity_variation",
        )
        for index, (ids, cap) in enumerate(suites)
    )


def _single_choice_cost_contexts(entity: str, cost: str) -> tuple[dict[str, Any], ...]:
    suites = (
        ((1, 2, 3), (2, 3, 1)),
        ((3, 1, 2), (4, 1, 3)),
        ((11, 4, 27), (2, 2, 5)),
        ((8, 2, 5, 1), (4, 2, 1, 3)),
        ((9, 7), (1, 5)),
        ((20, 3, 14), (6, 2, 4)),
        ((6, 12, 4), (3, 3, 3)),
        ((5, 15, 25), (5, 1, 2)),
        ((13, 2, 31, 7), (2, 4, 2, 6)),
        ((4, 10, 1), (7, 1, 5)),
        ((22, 5, 17), (1, 1, 3)),
        ((30, 9, 2, 18), (4, 2, 5, 1)),
    )
    result = []
    for index, (ids, costs) in enumerate(suites):
        facts = _facts(entity, ((value,) for value in ids))
        facts += _facts(cost, zip(ids, costs))
        tags = ["id_permutation", "variable_objective"]
        if len(costs) != len(set(costs)):
            tags.append("objective_tie")
        result.append(_context(f"choice_cost_{index:02d}", facts, *tags))
    return _contexts(result)


def _allocation_contexts(
    entity: str,
    destination: str,
    capacity: str,
    travel: str,
) -> tuple[dict[str, Any], ...]:
    result = []
    for index in range(12):
        entities = (1 + index * 10, 2 + index * 10, 3 + index * 10)
        destinations = (4 + index * 10, 5 + index * 10, 6 + index * 10)
        capacities = (1, 1, 1) if index % 3 == 0 else (2, 1, 1)
        if index == 4:
            capacities = (3, 0, 0)
        facts = _facts(entity, ((value,) for value in entities))
        facts += _facts(destination, ((value,) for value in destinations))
        facts += _facts(capacity, zip(destinations, capacities))
        costs = []
        for entity_index, x in enumerate(entities):
            for destination_index, h in enumerate(destinations):
                value = 1 + ((entity_index + destination_index + index) % 5)
                if index in (2, 7) and destination_index < 2:
                    value = 2
                costs.append((h, x, value))
        facts += _facts(travel, costs)
        tags = ["id_permutation", "capacity_variation", "multiple_feasible_resources",
                "variable_objective"]
        if index in (2, 7):
            tags.append("objective_tie")
        result.append(_context(f"allocation_{index:02d}", facts, *tags))
    return _contexts(result)


def _seen_admission_contexts(
    entity: str,
    severity: str | None,
    labels: Sequence[str] | None,
) -> tuple[dict[str, Any], ...]:
    """Counterfactuals for the Track A seen-syntax decision task.

    Capacity is FIXED at one (encoded by the model as a pairwise-exclusion
    constraint ``:- admit(X), admit(Y), X < Y``), so these contexts vary only
    ids, the severity map, fact order, entity count, and tie structure -- never
    the capacity or the priority mapping (which is baked into the fixed weak
    weights).  This respects the constraint that Track A cannot claim hidden
    capacity variation: capacity variation belongs to Track B.
    """
    # The isolated choice/capacity controls carry no severity facts, so they
    # reuse the same id/order/count/tie perturbations with the severity map
    # omitted.
    high, medium, low = labels if labels else (None, None, None)
    suites = (
        (((1, high), (2, medium), (3, low)), ("baseline",)),
        (((3, low), (1, medium), (2, high)), ("id_permutation",)),
        (((11, medium), (4, low), (27, high)), ("nonconsecutive_ids",)),
        (((8, high), (2, medium), (5, low)), ("id_permutation",)),
        (((9, high), (7, low)), ("two_entities",)),
        (((20, high), (3, high), (14, low)), ("top_tie",)),
        (((6, low), (12, medium), (4, high)), ("reversed_severity_order",)),
        (((5, medium), (15, medium), (25, medium)), ("all_tie",)),
        (((13, low), (2, high), (31, medium), (7, low)), ("four_entities",)),
        (((4, high), (10, medium), (1, low)), ("id_permutation",)),
        (((22, low), (5, high), (17, medium)), ("id_permutation",)),
        (((30, high), (9, low), (2, high), (18, medium)), ("four_entities", "top_tie")),
    )
    result = []
    for index, (assignments, extra_tags) in enumerate(suites):
        facts = _facts(entity, ((identifier,) for identifier, _ in assignments))
        tags = ["id_permutation", "fixed_capacity_one", *extra_tags]
        if severity is not None:
            facts += _facts(
                severity,
                ((identifier, f'"{label}"') for identifier, label in assignments),
            )
            tags.insert(1, "severity_variation")
        result.append(_context(f"seen_admission_{index:02d}", facts, *tags))
    return _contexts(result)


def atomic_assignment(rng: random.Random, v: Vocabulary) -> Example:
    entity = _pick(rng, v.predicates)[0]
    labels = _pick(rng, v.labels, 3)
    question = _preamble(
        f'For every element represented by predicate "{entity}", choose exactly one '
        f'triage label from "{labels[0]}", "{labels[1]}", and "{labels[2]}".'
    )
    answer = f'1 {{ assign(X,L) : triage_label(L) }} 1 :- {entity}(X).'
    facts = f"{entity}(1..3)." + _label_facts(labels)
    return Example(question, answer, facts)


def atomic_constraint(rng: random.Random, v: Vocabulary) -> Example:
    resource, unavailable = _pick(rng, v.predicates, 2)
    question = _preamble(
        f'Forbid every dispatch(X,A) whose resource A belongs to predicate "{unavailable}". '
        f'Predicate "{resource}" describes the available resource domain.'
    )
    answer = f":-dispatch(X,A),{unavailable}(A)."
    # The illustrative instance must satisfy its own constraint: the previous
    # facts asserted dispatch(2,2) together with unavailable(2), which this very
    # constraint forbids, making the reference program UNSAT on its own facts.
    facts = f"{resource}(1..2).dispatch(1,1).{unavailable}(2)."
    return Example(question, answer, facts)


def atomic_preference(rng: random.Random, v: Vocabulary) -> Example:
    resource, time_pred = _pick(rng, v.predicates, 2)
    question = _preamble(
        f'Minimize the travel time of dispatches. Predicate "{time_pred}" gives '
        f'{time_pred}(A,X,T), and "{resource}" identifies resources.'
    )
    answer = f":~dispatch(X,A),{time_pred}(A,X,T).[T@1,X,A]"
    facts = (
        f"{resource}(1..2).dispatch(1,1).dispatch(2,2)."
        f"{time_pred}(1,1,2).{time_pred}(2,2,4)."
    )
    return Example(question, answer, facts, (("dispatch", 2),))


def atomic_combination(rng: random.Random, v: Vocabulary) -> Example:
    left, right = _pick(rng, v.predicates, 2)
    question = _preamble(
        f'Generate every candidate pair between the sets represented by "{left}" and "{right}".'
    )
    answer = f"candidate(X,Y):-{left}(X),{right}(Y)."
    facts = f"{left}(1..2).{right}(1..3)."
    return Example(question, answer, facts)


def atomic_filtering(rng: random.Random, v: Vocabulary) -> Example:
    entity, status = _pick(rng, v.predicates, 2)
    label = _pick(rng, v.labels)[0]
    question = _preamble(
        f'Select every "{entity}" element whose "{status}" label is the string "{label}".'
    )
    answer = f'selected(X):-{entity}(X),{status}(X,"{label}").'
    facts = f'{entity}(1..3).{status}(1,"{label}").{status}(2,"{v.labels[0]}").'
    return Example(question, answer, facts)


def atomic_join(rng: random.Random, v: Vocabulary) -> Example:
    left, right = _pick(rng, v.predicates, 2)
    question = _preamble(
        f'Join binary predicates "{left}" and "{right}" on their shared second/second '
        "identifier and derive matched(X,Y)."
    )
    answer = f"matched(X,Y):-{left}(X,Z),{right}(Y,Z)."
    facts = f"{left}(1,10).{left}(2,20).{right}(3,10).{right}(4,30)."
    return Example(question, answer, facts)


def atomic_negative_filtering(rng: random.Random, v: Vocabulary) -> Example:
    entity, blocked = _pick(rng, v.predicates, 2)
    question = _preamble(
        f'Derive available(X) for every "{entity}" element that is not listed by "{blocked}".'
    )
    answer = f"available(X):-{entity}(X),not {blocked}(X)."
    facts = f"{entity}(1..3).{blocked}(2)."
    return Example(question, answer, facts)


def atomic_closure(rng: random.Random, v: Vocabulary) -> Example:
    edge = _pick(rng, v.predicates)[0]
    closure = _pick(rng, v.closures)[0]
    question = _preamble(
        f'Define "{closure}" as the transitive closure of binary predicate "{edge}".'
    )
    answer = (
        f"{closure}(X,Y):-{edge}(X,Y).\n"
        f"{closure}(X,Y):-{edge}(X,Z),{closure}(Z,Y)."
    )
    facts = f"{edge}(1,2).{edge}(2,3).{edge}(3,4)."
    return Example(question, answer, facts)


def atomic_filtering_two_labels(rng: random.Random, v: Vocabulary) -> Example:
    """Training-only length variant: the SAME filtering family, TWO rules.

    Attribution (§2.2/§2.6) showed the dominant L1 failure is output
    discipline: one fixed rule count per family taught the model nothing about
    when to stop.  Pairing one-rule and two-rule variants of the same family
    forces rule count to be read from the task description.
    """
    entity, status = _pick(rng, v.predicates, 2)
    label_a, label_b, other = _pick(rng, v.labels, 3)
    question = _preamble(
        f'Select every "{entity}" X whose "{status}" value is the string '
        f'"{label_a}" or the string "{label_b}", with one rule per accepted label.'
    )
    answer = (
        f'selected(X):-{entity}(X),{status}(X,"{label_a}").\n'
        f'selected(X):-{entity}(X),{status}(X,"{label_b}").'
    )
    facts = (
        f'{entity}(1..4).{status}(1,"{label_a}").'
        f'{status}(2,"{label_b}").{status}(3,"{other}").'
    )
    return Example(question, answer, facts)


def atomic_closure_direct_only(rng: random.Random, v: Vocabulary) -> Example:
    """Training-only closure-family length variant: ONE rule (no recursion).

    Attribution (§2.7) found closure regressed to 0% purely through output
    discipline: the prefix was byte-identical to gold (oracle ceiling 100%) and
    the model then appended self-contradicting constraints.  The closure family
    had a single fixed rule count in training, so nothing taught it to stop.
    This 1-rule / the 2-rule / the 3-rule variants below make the rule count a
    property of the question.
    """
    edge = _pick(rng, v.predicates)[0]
    question = _preamble(
        f'Derive direct(X,Y) for pairs linked by ONE step of binary predicate '
        f'"{edge}". Do not follow chains of more than one step.'
    )
    answer = f"direct(X,Y):-{edge}(X,Y)."
    facts = f"{edge}(1,2).{edge}(2,3).{edge}(3,4)."
    return Example(question, answer, facts, (("direct", 2),))


def atomic_closure_two_step(rng: random.Random, v: Vocabulary) -> Example:
    """Training-only variant: TWO rules but NON-recursive (exactly two steps).

    Same length as the recursive closure, so rule count alone cannot decide
    whether to recurse: the recursive literal has to come from the question.
    """
    edge = _pick(rng, v.predicates)[0]
    question = _preamble(
        f'Derive within_two(X,Y) for pairs reachable in exactly one or exactly '
        f'two steps of "{edge}". Do not define it recursively.'
    )
    answer = (
        f"within_two(X,Y):-{edge}(X,Y).\n"
        f"within_two(X,Y):-{edge}(X,Z),{edge}(Z,Y)."
    )
    facts = f"{edge}(1,2).{edge}(2,3).{edge}(3,4)."
    return Example(question, answer, facts, (("within_two", 2),))


def atomic_closure_reflexive(rng: random.Random, v: Vocabulary) -> Example:
    """Training-only closure-family variant: THREE rules (reflexive closure)."""
    edge, node = _pick(rng, v.predicates, 2)
    closure = _pick(rng, v.closures)[0]
    question = _preamble(
        f'Define "{closure}" as the REFLEXIVE transitive closure of "{edge}": '
        f'every "{node}" reaches itself, and chains of "{edge}" compose.'
    )
    answer = (
        f"{closure}(X,X):-{node}(X).\n"
        f"{closure}(X,Y):-{edge}(X,Y).\n"
        f"{closure}(X,Y):-{edge}(X,Z),{closure}(Z,Y)."
    )
    facts = f"{node}(1..3).{edge}(1,2).{edge}(2,3)."
    return Example(question, answer, facts, ((closure, 2),))


def atomic_choice_free_brace(rng: random.Random, v: Vocabulary) -> Example:
    """Choice-cardinality curriculum 1/4: FREE choice per element (2^n plans).

    Attribution (§2.7) showed the decision-task break is the choice construct,
    and specifically its CARDINALITY semantics: the model emits
    ``1 { admit(X) : e(X) } 1`` (exactly one) where the task asks for a free
    admit/skip choice.  Prefix-oracle ceiling was 0%, so this is a semantic gap,
    not an output-discipline gap -- these four variants differ ONLY in
    cardinality so the semantics must be read from the question.
    """
    entity = _pick(rng, v.predicates)[0]
    question = _preamble(
        f'Each "{entity}" X may independently be admitted or not, with no limit '
        f'on how many are admitted. Encode every possible admission plan.'
    )
    answer = f"{{ admit(X) }} :- {entity}(X)."
    facts = f"{entity}(1..3)."
    return Example(question, answer, facts, (("admit", 1),))


def atomic_choice_free_disjunction(rng: random.Random, v: Vocabulary) -> Example:
    """Choice-cardinality curriculum 2/4: same free choice, disjunctive syntax."""
    entity = _pick(rng, v.predicates)[0]
    question = _preamble(
        f'Each "{entity}" X is either admitted or skipped, with no limit on how '
        f'many are admitted. Use one disjunctive rule and encode every plan.'
    )
    answer = f"admit(X) | skip(X) :- {entity}(X)."
    facts = f"{entity}(1..3)."
    return Example(question, answer, facts, (("admit", 1), ("skip", 1)))


def atomic_choice_exactly_one(rng: random.Random, v: Vocabulary) -> Example:
    """Choice-cardinality curriculum 3/4: EXACTLY one admitted."""
    entity = _pick(rng, v.predicates)[0]
    question = _preamble(
        f'Admit EXACTLY ONE "{entity}" X -- no more and no fewer.'
    )
    answer = f"1 {{ admit(X) : {entity}(X) }} 1."
    facts = f"{entity}(1..3)."
    return Example(question, answer, facts, (("admit", 1),))


def atomic_choice_at_most_one(rng: random.Random, v: Vocabulary) -> Example:
    """Choice-cardinality curriculum 4/4: AT MOST one admitted (may admit none)."""
    entity = _pick(rng, v.predicates)[0]
    question = _preamble(
        f'Admit AT MOST ONE "{entity}" X; admitting none is also allowed.'
    )
    answer = f"{{ admit(X) : {entity}(X) }} 1."
    facts = f"{entity}(1..3)."
    return Example(question, answer, facts, (("admit", 1),))


def atomic_negative_filtering_double(rng: random.Random, v: Vocabulary) -> Example:
    """Training-only body-length variant: one rule, TWO negated exclusions."""
    entity, blocked_a, blocked_b = _pick(rng, v.predicates, 3)
    question = _preamble(
        f'Derive available(X) for every "{entity}" element listed by neither '
        f'"{blocked_a}" nor "{blocked_b}".'
    )
    answer = f"available(X):-{entity}(X),not {blocked_a}(X),not {blocked_b}(X)."
    facts = f"{entity}(1..4).{blocked_a}(2).{blocked_b}(3)."
    return Example(question, answer, facts)


def atomic_filtering_pair_outputs(rng: random.Random, v: Vocabulary) -> Example:
    """Training-only two-output variant: two DIFFERENT declared head names.

    Also breaks the single-head naming reflex: the interface declares two
    output predicates and each rule must use the right one.
    """
    entity, status = _pick(rng, v.predicates, 2)
    label_a, label_b = _pick(rng, v.labels, 2)
    question = _preamble(
        f'Derive urgent(X) for every "{entity}" X whose "{status}" is the string '
        f'"{label_a}", and safe(X) for those whose "{status}" is the string "{label_b}".'
    )
    answer = (
        f'urgent(X):-{entity}(X),{status}(X,"{label_a}").\n'
        f'safe(X):-{entity}(X),{status}(X,"{label_b}").'
    )
    facts = (
        f'{entity}(1..4).{status}(1,"{label_a}").{status}(2,"{label_b}").'
    )
    return Example(question, answer, facts, (("urgent", 1), ("safe", 1)))


def atomic_aggregate_comparison(rng: random.Random, v: Vocabulary) -> Example:
    destination, allocation = _pick(rng, v.predicates, 2)
    threshold = rng.choice((1, 2))
    question = _preamble(
        f'Derive overloaded(H) for each "{destination}" H when the number of X with '
        f'"{allocation}"(X,H) is greater than {threshold}.'
    )
    answer = (
        f"overloaded(H):-{destination}(H),"
        f"#count{{X:{allocation}(X,H)}}>{threshold}."
    )
    facts = f"{destination}(1..2).{allocation}(1,1).{allocation}(2,1).{allocation}(3,1)."
    return Example(question, answer, facts)


def assignment_constraint(rng: random.Random, v: Vocabulary) -> Example:
    entity, priority = _pick(rng, v.predicates, 2)
    labels = _pick(rng, v.labels, 3)
    forbidden = labels[0]
    question = _preamble(
        f'Assign exactly one triage label to each "{entity}" element from '
        f'"{labels[0]}", "{labels[1]}", and "{labels[2]}". Label "{forbidden}" '
        f'is forbidden unless predicate "{priority}" holds for that element.'
    )
    answer = (
        f"1 {{ assign(X,L) : triage_label(L) }} 1 :- {entity}(X).\n"
        f':-assign(X,"{forbidden}"),not {priority}(X).'
    )
    facts = f"{entity}(1..3).{priority}(1)." + _label_facts(labels)
    return Example(question, answer, facts)


def combination_filtering(rng: random.Random, v: Vocabulary) -> Example:
    left, right, capacity = _pick(rng, v.predicates, 3)
    question = _preamble(
        f'Build all candidate(X,H) pairs between "{left}" and "{right}", then derive '
        f'selected(X,H) only when "{capacity}"(H,C) reports C greater than zero.'
    )
    answer = (
        f"candidate(X,H):-{left}(X),{right}(H).\n"
        f"selected(X,H):-candidate(X,H),{capacity}(H,C),C>0."
    )
    facts = f"{left}(1..2).{right}(1..2).{capacity}(1,2).{capacity}(2,0)."
    return Example(question, answer, facts)


def join_negative_filtering(rng: random.Random, v: Vocabulary) -> Example:
    left, right, blocked = _pick(rng, v.predicates, 3)
    question = _preamble(
        f'Join "{left}"(X,Z) with "{right}"(H,Z) to obtain matched(X,H). From those '
        f'matches derive eligible(X,H) only when H is not listed by "{blocked}".'
    )
    answer = (
        f"matched(X,H):-{left}(X,Z),{right}(H,Z).\n"
        f"eligible(X,H):-matched(X,H),not {blocked}(H)."
    )
    facts = f"{left}(1,10).{left}(2,20).{right}(3,10).{right}(4,20).{blocked}(4)."
    return Example(question, answer, facts)


def closure_constraint(rng: random.Random, v: Vocabulary) -> Example:
    edge, closed = _pick(rng, v.predicates, 2)
    closure = _pick(rng, v.closures)[0]
    question = _preamble(
        f'Define "{closure}" as the transitive closure of "{edge}" and reject any '
        f'solution in which a reachable destination belongs to "{closed}".'
    )
    answer = (
        f"{closure}(X,Y):-{edge}(X,Y).\n"
        f"{closure}(X,Y):-{edge}(X,Z),{closure}(Z,Y).\n"
        f":-{closure}(X,Y),{closed}(Y)."
    )
    # Keep the instance satisfiable while still grounding the constraint.
    facts = f"{edge}(1,2).{edge}(2,3).{closed}(4)."
    return Example(question, answer, facts)


def filtering_preference(rng: random.Random, v: Vocabulary) -> Example:
    entity, status, delay = _pick(rng, v.predicates, 3)
    label = _pick(rng, v.labels)[0]
    question = _preamble(
        f'Select each "{entity}" X having "{status}"(X,"{label}"). Prefer selected '
        f'elements with lower values D supplied by "{delay}"(X,D).'
    )
    answer = (
        f'selected(X):-{entity}(X),{status}(X,"{label}").\n'
        f":~selected(X),{delay}(X,D).[D@1,X]"
    )
    facts = (
        f'{entity}(1..3).{status}(1,"{label}").{status}(2,"{label}").'
        f"{delay}(1,2).{delay}(2,5)."
    )
    return Example(question, answer, facts)


def assignment_aggregate_comparison(rng: random.Random, v: Vocabulary) -> Example:
    entity, destination = _pick(rng, v.predicates, 2)
    question = _preamble(
        f'Assign each "{entity}" X to exactly one "{destination}" H. Derive overloaded(H) '
        "when more than two elements are allocated to H, and forbid overloaded destinations."
    )
    answer = (
        f"1 {{ allocate(X,H) : {destination}(H) }} 1 :- {entity}(X).\n"
        f"overloaded(H):-{destination}(H),#count{{X:allocate(X,H)}}>2.\n"
        ":-overloaded(H)."
    )
    facts = f"{entity}(1..3).{destination}(1..2)."
    return Example(question, answer, facts)


def assignment_preference(rng: random.Random, v: Vocabulary) -> Example:
    entity, delay = _pick(rng, v.predicates, 2)
    labels = _pick(rng, v.labels, 3)
    costly = labels[0]
    question = _preamble(
        f'Assign exactly one of "{labels[0]}", "{labels[1]}", and "{labels[2]}" to '
        f'each "{entity}" element. Assigning "{costly}" incurs the element-specific '
        f'cost D given by "{delay}"(X,D).'
    )
    answer = (
        f"1 {{ assign(X,L) : triage_label(L) }} 1 :- {entity}(X).\n"
        f':~assign(X,"{costly}"),{delay}(X,D).[D@1,X]'
    )
    facts = f"{entity}(1..3)." + _label_facts(labels) + f"{delay}(1,3).{delay}(2,1).{delay}(3,2)."
    return Example(question, answer, facts)


def closure_filtering(rng: random.Random, v: Vocabulary) -> Example:
    edge, closed = _pick(rng, v.predicates, 2)
    closure = _pick(rng, v.closures)[0]
    question = _preamble(
        f'Compute the transitive closure "{closure}" of "{edge}". Then derive '
        f'safe_reach(X,Y) for reachable Y that are not listed by "{closed}".'
    )
    answer = (
        f"{closure}(X,Y):-{edge}(X,Y).\n"
        f"{closure}(X,Y):-{edge}(X,Z),{closure}(Z,Y).\n"
        f"safe_reach(X,Y):-{closure}(X,Y),not {closed}(Y)."
    )
    facts = f"{edge}(1,2).{edge}(2,3).{edge}(3,4).{closed}(3)."
    return Example(question, answer, facts)


def combination_aggregate_comparison(rng: random.Random, v: Vocabulary) -> Example:
    left, right = _pick(rng, v.predicates, 2)
    threshold = rng.choice((1, 2))
    question = _preamble(
        f'Build candidate(X,H) from every pair in "{left}" and "{right}". Derive '
        f'crowded(H) when the number of candidates X for H is greater than {threshold}.'
    )
    answer = (
        f"candidate(X,H):-{left}(X),{right}(H).\n"
        f"crowded(H):-{right}(H),#count{{X:candidate(X,H)}}>{threshold}."
    )
    facts = f"{left}(1..3).{right}(1..2)."
    return Example(question, answer, facts)


def guessing_constraint_optimization(rng: random.Random, v: Vocabulary) -> Example:
    entity, resource, unavailable, travel = _pick(rng, v.predicates, 4)
    question = _preamble(
        f'For each "{entity}" X choose exactly one "{resource}" A for dispatch(X,A). '
        f'Resources listed by "{unavailable}" cannot be used. Among admissible dispatches, '
        f'minimize T supplied by "{travel}"(A,X,T).'
    )
    answer = (
        f"1 {{ dispatch(X,A) : {resource}(A) }} 1 :- {entity}(X).\n"
        f":-dispatch(X,A),{unavailable}(A).\n"
        f":~dispatch(X,A),{travel}(A,X,T).[T@1,X,A]"
    )
    # Resource 2 is unavailable, while resources 1 and 3 remain feasible.
    # Their costs cross between the two entities, so the weak constraint
    # changes the optimum instead of being observationally inert.
    facts = (
        f"{entity}(1..2).{resource}(1..3).{unavailable}(2)."
        f"{travel}(1,1,2).{travel}(3,1,4)."
        f"{travel}(1,2,3).{travel}(3,2,1)."
        f"{travel}(2,1,1).{travel}(2,2,1)."
    )
    return Example(
        question,
        answer,
        facts,
        (("dispatch", 2),),
        _dispatch_contexts(entity, resource, unavailable, travel),
    )


# ---------------------------------------------------------------------------
# Explicit evaluation tracks
#
# Track A deliberately stays inside the syntax already exercised by the
# scenario-up model: normal/disjunctive rules, joins, default negation,
# comparisons, constraints, and recursive closure.  Track B isolates richer
# choice/cardinality, #count, and variable-cost weak-constraint constructs.
# Each track has four families at each of three levels; level-3 includes a
# length-matched independent control so rule count is not mistaken for
# dependency composition.
# ---------------------------------------------------------------------------


def track_a_l1_filtering(rng: random.Random, v: Vocabulary) -> Example:
    entity, status = _pick(rng, v.predicates, 2)
    label, other = _pick(rng, v.labels, 2)
    question = _preamble(
        f'Select every "{entity}" X whose "{status}" value is the string "{label}".'
    )
    answer = f'selected(X):-{entity}(X),{status}(X,"{label}").'
    facts = f'{entity}(1..3).{status}(1,"{label}").{status}(2,"{other}").'
    return Example(
        question, answer, facts, (("selected", 1),),
        _selection_contexts(entity, status, label, other),
    )


def track_a_l1_choice(rng: random.Random, v: Vocabulary) -> Example:
    """Isolated control: the disjunctive choice space alone (seen syntax).

    One of the three constructs the L3 decision task composes.  Without this
    control a decision-task failure cannot be attributed to composition rather
    than to the construct itself.
    """
    entity = _pick(rng, v.predicates)[0]
    question = _preamble(
        f'Each "{entity}" X is either admitted or skipped, and nothing else '
        f'constrains the outcome. Encode every possible admit/skip plan.'
    )
    answer = f"admit(X) | skip(X) :- {entity}(X)."
    facts = f"{entity}(1..3)."
    return Example(
        question, answer, facts, (("admit", 1),),
        _seen_admission_contexts(entity, None, None),
    )


def track_a_l1_capacity_constraint(rng: random.Random, v: Vocabulary) -> Example:
    """Isolated control: the capacity constraint alone, over a given choice space."""
    entity = _pick(rng, v.predicates)[0]
    background = f"admit(X) | skip(X) :- {entity}(X)."
    question = _preamble(
        f'The rule "{background}" is already given, so each "{entity}" X is '
        f'either admitted or skipped. Write ONLY the integrity constraint that '
        f'forbids admitting more than one "{entity}". Do not restate the given '
        f'rule and do not define admit yourself.'
    )
    answer = ":- admit(X), admit(Y), X < Y."
    facts = f"{entity}(1..3)."
    return Example(
        question, answer, facts, (("admit", 1),),
        _seen_admission_contexts(entity, None, None),
        background,
    )


def track_a_l1_fixed_weak_priority(rng: random.Random, v: Vocabulary) -> Example:
    """Isolated control: fixed-weight weak constraints alone.

    The background supplies the choice space AND the capacity constraint, so the
    feasible space is identical for every candidate and the score depends purely
    on whether the optimisation objective is written correctly.
    """
    entity, severity = _pick(rng, v.predicates, 2)
    high, medium, low = _pick(rng, v.labels, 3)
    background = (
        f"admit(X) | skip(X) :- {entity}(X).\n"
        f":- admit(X), admit(Y), X < Y."
    )
    question = _preamble(
        f'The rules "admit(X) | skip(X) :- {entity}(X)." and '
        f'":- admit(X), admit(Y), X < Y." are already given, so exactly one '
        f'"{entity}" may be admitted. Write ONLY the weak constraints that '
        f'penalise leaving an "{entity}" X unadmitted by its severity '
        f'"{severity}"(X,S): "{high}" costs 3, "{medium}" costs 2, "{low}" costs 1. '
        f'Do not restate the given rules and do not define admit yourself.'
    )
    answer = (
        f':~ {entity}(X), not admit(X), {severity}(X,"{high}"). [3@1,X]\n'
        f':~ {entity}(X), not admit(X), {severity}(X,"{medium}"). [2@1,X]\n'
        f':~ {entity}(X), not admit(X), {severity}(X,"{low}"). [1@1,X]'
    )
    facts = (
        f"{entity}(1..3)."
        f'{severity}(1,"{high}").{severity}(2,"{medium}").{severity}(3,"{low}").'
    )
    return Example(
        question, answer, facts, (("admit", 1),),
        _seen_admission_contexts(entity, severity, (high, medium, low)),
        background,
    )


def track_a_l1_negative_filtering(rng: random.Random, v: Vocabulary) -> Example:
    entity, blocked = _pick(rng, v.predicates, 2)
    question = _preamble(
        f'Derive available(X) for each "{entity}" X not listed by "{blocked}".'
    )
    answer = f"available(X):-{entity}(X),not {blocked}(X)."
    facts = f"{entity}(1..4).{blocked}(2).{blocked}(4)."
    return Example(
        question, answer, facts, (("available", 1),), _negative_filter_contexts(entity, blocked)
    )


def track_a_l1_join(rng: random.Random, v: Vocabulary) -> Example:
    left, right = _pick(rng, v.predicates, 2)
    question = _preamble(
        f'Join "{left}"(X,Z) and "{right}"(H,Z) on Z to derive matched(X,H).'
    )
    answer = f"matched(X,H):-{left}(X,Z),{right}(H,Z)."
    facts = f"{left}(1,10).{left}(2,20).{right}(3,10).{right}(4,30)."
    return Example(question, answer, facts, (("matched", 2),), _join_contexts(left, right))


def track_a_l1_closure(rng: random.Random, v: Vocabulary) -> Example:
    edge = _pick(rng, v.predicates)[0]
    closure = _pick(rng, v.closures)[0]
    question = _preamble(f'Define "{closure}" as the transitive closure of "{edge}".')
    answer = (
        f"{closure}(X,Y):-{edge}(X,Y).\n"
        f"{closure}(X,Y):-{edge}(X,Z),{closure}(Z,Y)."
    )
    facts = f"{edge}(1,2).{edge}(2,3)."
    return Example(
        question, answer, facts, ((closure, 2),), _network_contexts(edge)
    )


def track_a_l2_join_negative_filter(rng: random.Random, v: Vocabulary) -> Example:
    left, right, blocked = _pick(rng, v.predicates, 3)
    question = _preamble(
        f'Join "{left}"(X,Z) with "{right}"(H,Z) as matched(X,H), then derive '
        f'eligible(X,H) only for H not listed by "{blocked}".'
    )
    answer = (
        f"matched(X,H):-{left}(X,Z),{right}(H,Z).\n"
        f"eligible(X,H):-matched(X,H),not {blocked}(H)."
    )
    facts = (
        f"{left}(1,10).{left}(2,20).{right}(3,10).{right}(4,20)."
        f"{blocked}(4)."
    )
    # Observe only the final deliverable; `matched` is a named intermediate the
    # model may legitimately inline, so it must not be forced into the check.
    return Example(
        question, answer, facts, (("eligible", 2),),
        _join_negative_contexts(left, right, blocked),
    )


def track_a_l2_closure_triage(rng: random.Random, v: Vocabulary) -> Example:
    edge, triage = _pick(rng, v.predicates, 2)
    closure = _pick(rng, v.closures)[0]
    question = _preamble(
        f'Compute transitive closure "{closure}" of "{edge}", then derive '
        f'servable(X,Y) for reachable Y listed by "{triage}".'
    )
    answer = (
        f"{closure}(X,Y):-{edge}(X,Y).\n"
        f"{closure}(X,Y):-{edge}(X,Z),{closure}(Z,Y).\n"
        f"servable(X,Y):-{closure}(X,Y),{triage}(Y)."
    )
    facts = f"{edge}(1,2).{edge}(2,3).{triage}(3).{triage}(9)."
    return Example(
        question, answer, facts, (("servable", 2),),
        _network_contexts(edge, triage=triage),
    )


def track_a_l2_combination_negative_filter(rng: random.Random, v: Vocabulary) -> Example:
    left, right, blocked = _pick(rng, v.predicates, 3)
    question = _preamble(
        f'Build candidate(X,H) from "{left}" and "{right}", then derive '
        f'eligible(X,H) when H is not in "{blocked}".'
    )
    answer = (
        f"candidate(X,H):-{left}(X),{right}(H).\n"
        f"eligible(X,H):-candidate(X,H),not {blocked}(H)."
    )
    facts = f"{left}(1..3).{right}(4..6).{blocked}(5)."
    return Example(
        question, answer, facts, (("eligible", 2),),
        _combination_negative_contexts(left, right, blocked),
    )


def track_a_l2_numeric_pipeline(rng: random.Random, v: Vocabulary) -> Example:
    entity, destination, distance = _pick(rng, v.predicates, 3)
    question = _preamble(
        f'Build candidate(X,H) from "{entity}" and "{destination}", then derive '
        f'nearby(X,H) only when "{distance}"(X,H,D) has D at most 5.'
    )
    answer = (
        f"candidate(X,H):-{entity}(X),{destination}(H).\n"
        f"nearby(X,H):-candidate(X,H),{distance}(X,H,D),D<=5."
    )
    facts = (
        f"{entity}(1..2).{destination}(3..4)."
        f"{distance}(1,3,2).{distance}(1,4,8)."
        f"{distance}(2,3,5).{distance}(2,4,7)."
    )
    return Example(
        question, answer, facts, (("nearby", 2),),
        _threshold_distance_contexts(entity, destination, distance),
    )


def track_a_l3_closure_triage_blocked(rng: random.Random, v: Vocabulary) -> Example:
    edge, triage, blocked = _pick(rng, v.predicates, 3)
    closure = _pick(rng, v.closures)[0]
    question = _preamble(
        f'Compute closure "{closure}" of "{edge}"; derive reachable_triage(X,Y) for '
        f'reachable "{triage}" nodes; then derive servable(X,Y) unless Y is in "{blocked}".'
    )
    answer = (
        f"{closure}(X,Y):-{edge}(X,Y).\n"
        f"{closure}(X,Y):-{edge}(X,Z),{closure}(Z,Y).\n"
        f"reachable_triage(X,Y):-{closure}(X,Y),{triage}(Y).\n"
        f"servable(X,Y):-reachable_triage(X,Y),not {blocked}(Y)."
    )
    facts = f"{edge}(1,2).{edge}(2,3).{triage}(2).{triage}(3).{blocked}(2)."
    return Example(
        question,
        answer,
        facts,
        (("servable", 2),),
        _network_contexts(edge, triage=triage, blocked=blocked),
    )


def track_a_l3_join_status_blocked(rng: random.Random, v: Vocabulary) -> Example:
    left, right, status, blocked = _pick(rng, v.predicates, 4)
    label = _pick(rng, v.labels)[0]
    question = _preamble(
        f'Join "{left}"(X,Z) and "{right}"(H,Z) as matched(X,H); retain urgent_match '
        f'for X having "{status}"(X,"{label}"); then derive eligible unless H is "{blocked}".'
    )
    answer = (
        f"matched(X,H):-{left}(X,Z),{right}(H,Z).\n"
        f'urgent_match(X,H):-matched(X,H),{status}(X,"{label}").\n'
        f"eligible(X,H):-urgent_match(X,H),not {blocked}(H)."
    )
    facts = (
        f"{left}(1,10).{left}(2,20).{right}(3,10).{right}(4,20)."
        f'{status}(1,"{label}").{status}(2,"{v.labels[0]}").{blocked}(4).'
    )
    return Example(
        question, answer, facts,
        (("eligible", 2),),
        _join_status_blocked_contexts(left, right, status, blocked, label,
                                      next(x for x in v.labels if x != label)),
    )


def track_a_l3_combination_distance_blocked(rng: random.Random, v: Vocabulary) -> Example:
    entity, destination, distance, blocked = _pick(rng, v.predicates, 4)
    question = _preamble(
        f'Build candidate(X,H) from "{entity}" and "{destination}"; derive nearby(X,H) '
        f'when "{distance}"(X,H,D) is at most 5; then recommend pairs whose H is not '
        f'listed by "{blocked}".'
    )
    answer = (
        f"candidate(X,H):-{entity}(X),{destination}(H).\n"
        f"nearby(X,H):-candidate(X,H),{distance}(X,H,D),D<=5.\n"
        f"recommended(X,H):-nearby(X,H),not {blocked}(H)."
    )
    facts = (
        f"{entity}(1..2).{destination}(3..4).{blocked}(4)."
        f"{distance}(1,3,2).{distance}(1,4,3)."
        f"{distance}(2,3,8).{distance}(2,4,4)."
    )
    return Example(
        question, answer, facts,
        (("recommended", 2),),
        _threshold_distance_contexts(entity, destination, distance, blocked=blocked),
    )


def track_a_l3_length_matched_control(rng: random.Random, v: Vocabulary) -> Example:
    entity, status, left, right, blocked, measure = _pick(rng, v.predicates, 6)
    label = _pick(rng, v.labels)[0]
    question = _preamble(
        "Write four independent medical rules: a string-label filter, a join, a "
        "negative filter, and a numeric filter. No derived predicate feeds another rule. "
        f'Use "{entity}", "{status}", "{left}", "{right}", "{blocked}", and "{measure}".'
    )
    answer = (
        f'selected(X):-{entity}(X),{status}(X,"{label}").\n'
        f"matched(X,H):-{left}(X,Z),{right}(H,Z).\n"
        f"available(X):-{entity}(X),not {blocked}(X).\n"
        f"nearby(X):-{entity}(X),{measure}(X,D),D<=5."
    )
    facts = (
        f'{entity}(1..3).{status}(1,"{label}").{blocked}(2).'
        f"{left}(1,10).{right}(3,10).{measure}(1,2).{measure}(2,8)."
    )
    targets = (("selected", 1), ("matched", 2), ("available", 1), ("nearby", 1))
    return Example(
        question, answer, facts, targets,
        _length_control_contexts(entity, status, label, left, right, blocked, measure),
    )


def track_a_l3_seen_admission(rng: random.Random, v: Vocabulary) -> Example:
    """Track A L3 decision task in scenario-up's SEEN syntax.

    Composes three constructs the model saw in isolation -- disjunctive choice
    (``a|b:-``), a pairwise-exclusion capacity-1 constraint, and fixed-weight
    weak constraints with ``not`` -- into a triage-admission decision.  The
    model must ENCODE THE WHOLE PROGRAM (no background is injected), so the
    feasible-space check is non-trivial and ``admit(1).`` cannot pass it.
    Capacity is fixed at one and priorities are baked into the weak weights,
    so the hidden contexts vary only ids/severity/order/ties.
    """
    entity, severity = _pick(rng, v.predicates, 2)
    high, medium, low = _pick(rng, v.labels, 3)
    question = _preamble(
        f'Each "{entity}" X is either admitted or skipped, and AT MOST ONE may be '
        f'admitted. Leaving X unadmitted costs its severity priority via '
        f'"{severity}"(X,S): "{high}" costs 3, "{medium}" costs 2, "{low}" costs 1. '
        f'Encode all feasible admit/skip plans and minimise the total priority of '
        f'the patients left unadmitted.'
    )
    answer = (
        f"admit(X) | skip(X) :- {entity}(X).\n"
        f":- admit(X), admit(Y), X < Y.\n"
        f':~ {entity}(X), not admit(X), {severity}(X,"{high}"). [3@1,X]\n'
        f':~ {entity}(X), not admit(X), {severity}(X,"{medium}"). [2@1,X]\n'
        f':~ {entity}(X), not admit(X), {severity}(X,"{low}"). [1@1,X]'
    )
    facts = (
        f"{entity}(1..3)."
        f'{severity}(1,"{high}").{severity}(2,"{medium}").{severity}(3,"{low}").'
    )
    return Example(
        question, answer, facts, (("admit", 1),),
        _seen_admission_contexts(entity, severity, (high, medium, low)),
    )


def track_b_l1_choice(rng: random.Random, v: Vocabulary) -> Example:
    entity = _pick(rng, v.predicates)[0]
    labels = _pick(rng, v.labels, 3)
    question = _preamble(
        f'For each "{entity}" X choose exactly one triage_label L as assign(X,L).'
    )
    answer = f"1 {{ assign(X,L) : triage_label(L) }} 1 :- {entity}(X)."
    facts = f"{entity}(1..3)." + _label_facts(labels)
    return Example(question, answer, facts, (("assign", 2),), _default_hidden_contexts(facts))


def track_b_l1_count(rng: random.Random, v: Vocabulary) -> Example:
    destination, allocation = _pick(rng, v.predicates, 2)
    question = _preamble(
        f'Derive overloaded(H) for "{destination}" H with more than two '
        f'"{allocation}"(X,H) facts.'
    )
    answer = (
        f"overloaded(H):-{destination}(H),"
        f"#count{{X:{allocation}(X,H)}}>2."
    )
    facts = (
        f"{destination}(1..2).{allocation}(1,1).{allocation}(2,1)."
        f"{allocation}(3,1).{allocation}(4,2)."
    )
    return Example(
        question, answer, facts, (("overloaded", 1),), _default_hidden_contexts(facts)
    )


def track_b_l1_variable_weak(rng: random.Random, v: Vocabulary) -> Example:
    travel = _pick(rng, v.predicates)[0]
    question = _preamble(
        f'For supplied dispatch(X,A) facts, minimize variable cost T from "{travel}"(A,X,T).'
    )
    answer = f":~dispatch(X,A),{travel}(A,X,T).[T@1,X,A]"
    facts = (
        f"dispatch(1,4).dispatch(2,5).{travel}(4,1,2).{travel}(5,2,7)."
    )
    return Example(question, answer, facts, (("dispatch", 2),), _default_hidden_contexts(facts))


def track_b_l1_count_constraint(rng: random.Random, v: Vocabulary) -> Example:
    capacity = _pick(rng, v.predicates)[0]
    question = _preamble(
        f'Reject supplied admit(X) facts when their #count exceeds K from "{capacity}"(K).'
    )
    answer = f":-{capacity}(K),#count{{X:admit(X)}}>K."
    facts = f"admit(1).admit(2).{capacity}(2)."
    return Example(question, answer, facts, (("admit", 1),), _default_hidden_contexts(facts))


def track_b_l2_choice_capacity(rng: random.Random, v: Vocabulary) -> Example:
    entity, capacity = _pick(rng, v.predicates, 2)
    question = _preamble(
        f'Choose any admit(X) from "{entity}" elements, but reject choices whose count '
        f'exceeds K supplied by "{capacity}"(K).'
    )
    answer = (
        f"{{ admit(X) }}:-{entity}(X).\n"
        f":-{capacity}(K),#count{{X:admit(X)}}>K."
    )
    facts = f"{entity}(1..3).{capacity}(1)."
    return Example(
        question, answer, facts, (("admit", 1),), _capacity_contexts(entity, capacity)
    )


def track_b_l2_choice_optimization(rng: random.Random, v: Vocabulary) -> Example:
    entity, cost = _pick(rng, v.predicates, 2)
    question = _preamble(
        f'Choose exactly one admit(X) from "{entity}", minimizing C from "{cost}"(X,C).'
    )
    answer = (
        f"1 {{ admit(X) : {entity}(X) }} 1.\n"
        f":~admit(X),{cost}(X,C).[C@1,X]"
    )
    facts = f"{entity}(1..3).{cost}(1,3).{cost}(2,1).{cost}(3,2)."
    return Example(
        question, answer, facts, (("admit", 1),),
        _single_choice_cost_contexts(entity, cost),
    )


def track_b_l2_choice_unavailable(rng: random.Random, v: Vocabulary) -> Example:
    entity, resource, unavailable, unused_travel = _pick(rng, v.predicates, 4)
    question = _preamble(
        f'Choose exactly one "{resource}" A for each "{entity}" X as dispatch(X,A), '
        f'forbidding A listed by "{unavailable}".'
    )
    answer = (
        f"1 {{ dispatch(X,A) : {resource}(A) }} 1 :- {entity}(X).\n"
        f":-dispatch(X,A),{unavailable}(A)."
    )
    facts = f"{entity}(1..2).{resource}(1..3).{unavailable}(2)."
    return Example(
        question, answer, facts, (("dispatch", 2),),
        _dispatch_contexts(entity, resource, unavailable, unused_travel),
    )


def track_b_l2_candidate_count(rng: random.Random, v: Vocabulary) -> Example:
    left, right = _pick(rng, v.predicates, 2)
    question = _preamble(
        f'Build candidate(X,H) from "{left}" and "{right}", then derive crowded(H) '
        "when more than two candidates depend on H."
    )
    answer = (
        f"candidate(X,H):-{left}(X),{right}(H).\n"
        f"crowded(H):-{right}(H),#count{{X:candidate(X,H)}}>2."
    )
    facts = f"{left}(1..3).{right}(4..5)."
    targets = (("candidate", 2), ("crowded", 1))
    return Example(question, answer, facts, targets, _default_hidden_contexts(facts))


def track_b_l3_admission(rng: random.Random, v: Vocabulary) -> Example:
    entity, severity, capacity, priority = _pick(rng, v.predicates, 4)
    labels = _pick(rng, v.labels, 3)
    high, medium, low = labels
    question = _preamble(
        f'Encode all feasible admit(X) decisions for "{entity}" elements. At most K '
        f'may be admitted, where "{capacity}"(K). Severity is "{severity}"(X,S), and '
        f'"{priority}"(S,P) gives the penalty for leaving X unadmitted; minimize it.'
    )
    answer = (
        f"{{ admit(X) }}:-{entity}(X).\n"
        f":-{capacity}(K),#count{{X:admit(X)}}>K.\n"
        f":~{entity}(X),not admit(X),{severity}(X,S),{priority}(S,P).[P@1,X]"
    )
    facts = (
        f"{entity}(1..3)."
        f'{severity}(1,"{high}").{severity}(2,"{medium}").{severity}(3,"{low}").'
        f"{capacity}(1)."
        f'{priority}("{high}",3).{priority}("{medium}",2).{priority}("{low}",1).'
    )
    return Example(
        question, answer, facts, (("admit", 1),),
        _admission_contexts(entity, severity, capacity, priority, labels),
    )


def track_b_l3_dispatch(rng: random.Random, v: Vocabulary) -> Example:
    return guessing_constraint_optimization(rng, v)


def track_b_l3_allocation(rng: random.Random, v: Vocabulary) -> Example:
    entity, destination, capacity, travel = _pick(rng, v.predicates, 4)
    question = _preamble(
        f'Assign each "{entity}" X to exactly one "{destination}" H as allocate(X,H). '
        f'Forbid assignments exceeding "{capacity}"(H,K) according to #count, and '
        f'minimize variable "{travel}"(H,X,T).'
    )
    answer = (
        f"1 {{ allocate(X,H) : {destination}(H) }} 1 :- {entity}(X).\n"
        f":-{destination}(H),{capacity}(H,K),#count{{X:allocate(X,H)}}>K.\n"
        f":~allocate(X,H),{travel}(H,X,T).[T@1,X,H]"
    )
    facts = (
        f"{entity}(1..3).{destination}(4..6)."
        f"{capacity}(4,1).{capacity}(5,1).{capacity}(6,1)."
        f"{travel}(4,1,1).{travel}(5,1,3).{travel}(6,1,4)."
        f"{travel}(4,2,4).{travel}(5,2,1).{travel}(6,2,3)."
        f"{travel}(4,3,3).{travel}(5,3,4).{travel}(6,3,1)."
    )
    return Example(
        question, answer, facts, (("allocate", 2),),
        _allocation_contexts(entity, destination, capacity, travel),
    )


def track_b_l3_length_matched_control(rng: random.Random, v: Vocabulary) -> Example:
    entity, destination, allocation, unavailable, delay = _pick(rng, v.predicates, 5)
    question = _preamble(
        "Write four independent rich ASP rules: an admit choice, an overload count, "
        "an unavailable-dispatch constraint, and a variable-delay weak constraint. "
        "No predicate derived by one rule is consumed by another."
    )
    answer = (
        f"{{ admit(X) }}:-{entity}(X).\n"
        f"overloaded(H):-{destination}(H),#count{{X:{allocation}(X,H)}}>1.\n"
        f":-dispatch(X,A),{unavailable}(A).\n"
        f":~treated(X),{delay}(X,D).[D@1,X]"
    )
    facts = (
        f"{entity}(1..3).{destination}(4..5)."
        f"{allocation}(1,4).{allocation}(2,4)."
        f"dispatch(1,6).{unavailable}(7).treated(2).{delay}(2,3)."
    )
    targets = (("admit", 1), ("overloaded", 1), ("dispatch", 2), ("treated", 1))
    return Example(question, answer, facts, targets, _default_hidden_contexts(facts))


ATOMIC_SPECS = (
    TaskSpec("atomic_assignment", ("assignment",),
             ("choice_rule", "positive_dependency", "shared_variable"),
             ("entity -positive-> assign",), atomic_assignment, "atomic"),
    TaskSpec("atomic_constraint", ("constraint",),
             ("positive_dependency", "shared_variable"),
             ("dispatch -positive-> constraint",), atomic_constraint, "atomic"),
    TaskSpec("atomic_preference", ("preference",),
             ("positive_dependency", "shared_variable", "weak_constraint"),
             ("dispatch -positive-> weak_constraint",), atomic_preference, "atomic"),
    TaskSpec("atomic_combination", ("combination",),
             ("positive_dependency", "shared_variable"),
             ("left -positive-> candidate", "right -positive-> candidate"),
             atomic_combination, "atomic"),
    TaskSpec("atomic_filtering", ("filtering",),
             ("positive_dependency", "shared_variable"),
             ("entity -positive-> selected", "status -positive-> selected"),
             atomic_filtering, "atomic"),
    TaskSpec("atomic_join", ("join",),
             ("positive_dependency", "shared_variable"),
             ("left -positive-> matched", "right -positive-> matched"), atomic_join, "atomic"),
    TaskSpec("atomic_negative_filtering", ("negative_filtering",),
             ("negative_dependency", "positive_dependency", "shared_variable"),
             ("entity -positive-> available", "blocked -negative-> available"),
             atomic_negative_filtering, "atomic"),
    TaskSpec("atomic_closure", ("closure",),
             ("base_recursive", "positive_dependency", "recursive_scc", "shared_variable"),
             ("edge -positive-> closure", "closure -positive-> closure"), atomic_closure, "atomic"),
    TaskSpec("atomic_aggregate_comparison", ("aggregate_comparison",),
             ("aggregate", "comparison", "positive_dependency", "shared_variable"),
             ("allocation -aggregate-> overloaded",), atomic_aggregate_comparison, "atomic"),
    # Training-only length/naming variants (ATOMIC_SPECS never feeds test CSVs):
    # same families with different rule counts and head-name demands, so output
    # length and head names must be read from the question, not memorized.
    TaskSpec("atomic_filtering_two_labels", ("filtering",),
             ("positive_dependency", "shared_variable"),
             ("entity -positive-> selected", "status -positive-> selected"),
             atomic_filtering_two_labels, "atomic"),
    TaskSpec("atomic_negative_filtering_double", ("negative_filtering",),
             ("negative_dependency", "positive_dependency", "shared_variable"),
             ("entity -positive-> available", "blocked -negative-> available"),
             atomic_negative_filtering_double, "atomic"),
    TaskSpec("atomic_filtering_pair_outputs", ("filtering",),
             ("positive_dependency", "shared_variable"),
             ("entity -positive-> urgent", "status -positive-> safe"),
             atomic_filtering_pair_outputs, "atomic"),
    # P1a (§2.7): closure-family rule-count variants. closure regressed to 0%
    # with a 100% prefix-oracle ceiling, i.e. pure output-discipline failure,
    # because the family had one fixed rule count in training. 1 / 2 / 2-recursive
    # / 3 rule variants make the count and the recursion a property of the task.
    TaskSpec("atomic_closure_direct_only", ("closure",),
             ("positive_dependency", "shared_variable"),
             ("edge -positive-> direct",), atomic_closure_direct_only, "atomic"),
    TaskSpec("atomic_closure_two_step", ("closure",),
             ("positive_dependency", "shared_variable"),
             ("edge -positive-> within_two",), atomic_closure_two_step, "atomic"),
    TaskSpec("atomic_closure_reflexive", ("closure",),
             ("base_recursive", "positive_dependency", "recursive_scc", "shared_variable"),
             ("edge -positive-> closure", "closure -positive-> closure"),
             atomic_closure_reflexive, "atomic"),
    # P2 (§2.7): choice-cardinality curriculum. The decision task breaks at the
    # choice construct with a 0% prefix-oracle ceiling (semantic gap, not
    # discipline): the model writes "exactly one" for a free admit/skip choice.
    # These four differ ONLY in cardinality, so it must be read from the task.
    TaskSpec("atomic_choice_free_brace", ("assignment",),
             ("choice_rule", "positive_dependency", "shared_variable"),
             ("entity -positive-> admit",), atomic_choice_free_brace, "atomic"),
    TaskSpec("atomic_choice_free_disjunction", ("assignment",),
             ("positive_dependency", "shared_variable"),
             ("entity -positive-> admit",), atomic_choice_free_disjunction, "atomic"),
    # Cardinality-bounded choice ("1 { ... } 1" / "{ ... } 1") is a choice rule
    # with bounds, not a #count aggregate, so only choice_rule is declared.
    TaskSpec("atomic_choice_exactly_one", ("assignment",),
             ("choice_rule",),
             ("entity -positive-> admit",), atomic_choice_exactly_one, "atomic"),
    TaskSpec("atomic_choice_at_most_one", ("assignment",),
             ("choice_rule",),
             ("entity -positive-> admit",), atomic_choice_at_most_one, "atomic"),
)


SEEN_COMPOSITE_SPECS = (
    TaskSpec("assignment_constraint", ("assignment", "constraint"),
             ("choice_rule", "negative_dependency", "positive_dependency", "shared_variable"),
             ("assign -positive-> constraint", "priority -negative-> constraint"),
             assignment_constraint, "seen_composite"),
    TaskSpec("combination_filtering", ("combination", "filtering"),
             ("comparison", "positive_dependency", "shared_variable"),
             ("candidate -positive-> selected",), combination_filtering, "seen_composite"),
    TaskSpec("join_negative_filtering", ("join", "negative_filtering"),
             ("negative_dependency", "positive_dependency", "shared_variable"),
             ("matched -positive-> eligible", "blocked -negative-> eligible"),
             join_negative_filtering, "seen_composite"),
    TaskSpec("closure_constraint", ("closure", "constraint"),
             ("base_recursive", "positive_dependency", "recursive_scc", "shared_variable"),
             ("edge -positive-> closure", "closure -positive-> closure",
              "closure -positive-> constraint"), closure_constraint, "seen_composite"),
    TaskSpec("filtering_preference", ("filtering", "preference"),
             ("positive_dependency", "shared_variable", "weak_constraint"),
             ("selected -positive-> weak_constraint",), filtering_preference, "seen_composite"),
    TaskSpec("assignment_aggregate_comparison", ("assignment", "aggregate_comparison"),
             ("aggregate", "choice_rule", "comparison", "positive_dependency", "shared_variable"),
             ("allocate -aggregate-> overloaded", "overloaded -positive-> constraint"),
             assignment_aggregate_comparison, "seen_composite"),
)


HELDOUT_COMPOSITE_SPECS = (
    TaskSpec("assignment_preference", ("assignment", "preference"),
             ("choice_rule", "positive_dependency", "shared_variable", "weak_constraint"),
             ("assign -positive-> weak_constraint",), assignment_preference, "heldout_composite"),
    TaskSpec("closure_filtering", ("closure", "filtering"),
             ("base_recursive", "negative_dependency", "positive_dependency", "recursive_scc",
              "shared_variable"),
             ("edge -positive-> closure", "closure -positive-> closure",
              "closure -positive-> safe_reach", "closed -negative-> safe_reach"),
             closure_filtering, "heldout_composite"),
    TaskSpec("combination_aggregate_comparison", ("combination", "aggregate_comparison"),
             ("aggregate", "comparison", "positive_dependency", "shared_variable"),
             ("candidate -aggregate-> crowded",), combination_aggregate_comparison,
             "heldout_composite"),
    TaskSpec("guessing_constraint_optimization", ("assignment", "constraint", "preference"),
             ("choice_rule", "positive_dependency", "shared_variable", "weak_constraint"),
             ("dispatch -positive-> constraint", "dispatch -positive-> weak_constraint"),
             guessing_constraint_optimization, "heldout_composite"),
)


TRACK_A_SPECS = (
    # Isolated controls for the three constructs the L3 decision task composes.
    # Without them a decision-task failure cannot be attributed to composition
    # rather than to an individual construct.
    TaskSpec(
        "track_a_l1_choice", ("assignment",),
        ("positive_dependency", "shared_variable"),
        ("entity -positive-> admit",),
        track_a_l1_choice, "track_evaluation",
        track="A", level=1, control_type="atomic_control",
        optimization=False, context_family="seen_admission_ties",
        pattern="disjunctive_choice",
    ),
    TaskSpec(
        "track_a_l1_capacity_constraint", ("constraint",),
        ("comparison", "positive_dependency", "shared_variable"),
        ("admit -positive-> constraint",),
        track_a_l1_capacity_constraint, "track_evaluation",
        track="A", level=1, control_type="isolated_control",
        optimization=False, context_family="seen_admission_ties",
        pattern="capacity_constraint",
    ),
    TaskSpec(
        "track_a_l1_fixed_weak_priority", ("preference",),
        ("negative_dependency", "positive_dependency", "shared_variable",
         "weak_constraint"),
        ("admit -negative-> weak_constraint",),
        track_a_l1_fixed_weak_priority, "track_evaluation",
        track="A", level=1, control_type="isolated_control",
        optimization=True, context_family="seen_admission_ties",
        pattern="fixed_weight_preference",
    ),
    TaskSpec(
        "track_a_l1_filtering", ("filtering",),
        ("positive_dependency", "shared_variable"),
        ("entity -positive-> selected", "status -positive-> selected"),
        track_a_l1_filtering, "track_evaluation",
        track="A", level=1, control_type="atomic_control",
        context_family="selection_counterfactuals", pattern="filtering",
    ),
    TaskSpec(
        "track_a_l1_negative_filtering", ("negative_filtering",),
        ("negative_dependency", "positive_dependency", "shared_variable"),
        ("entity -positive-> available", "blocked -negative-> available"),
        track_a_l1_negative_filtering, "track_evaluation",
        track="A", level=1, control_type="atomic_control",
        context_family="negative_filter_counterfactuals", pattern="negative_filtering",
    ),
    TaskSpec(
        "track_a_l1_join", ("join",),
        ("positive_dependency", "shared_variable"),
        ("left -positive-> matched", "right -positive-> matched"),
        track_a_l1_join, "track_evaluation",
        track="A", level=1, control_type="atomic_control",
        context_family="join_id_permutations", pattern="join",
    ),
    TaskSpec(
        "track_a_l1_closure", ("closure",),
        ("base_recursive", "positive_dependency", "recursive_scc", "shared_variable"),
        ("edge -positive-> closure", "closure -positive-> closure"),
        track_a_l1_closure, "track_evaluation",
        track="A", level=1, control_type="atomic_control",
        context_family="closure_topologies", pattern="closure",
    ),
    TaskSpec(
        "track_a_l2_join_negative_filter", ("join", "negative_filtering"),
        ("negative_dependency", "positive_dependency", "shared_variable"),
        ("matched -positive-> eligible", "blocked -negative-> eligible"),
        track_a_l2_join_negative_filter, "track_evaluation",
        track="A", level=2, control_type="dependent_composition",
        context_family="join_filter_counterfactuals", pattern="join>negative_filtering",
    ),
    TaskSpec(
        "track_a_l2_closure_triage", ("closure", "filtering"),
        ("base_recursive", "positive_dependency", "recursive_scc", "shared_variable"),
        ("edge -positive-> closure", "closure -positive-> closure",
         "closure -positive-> servable"),
        track_a_l2_closure_triage, "track_evaluation",
        track="A", level=2, control_type="dependent_composition",
        context_family="closure_reachable_unreachable_triage", pattern="closure>filtering",
    ),
    TaskSpec(
        "track_a_l2_combination_negative_filter", ("combination", "negative_filtering"),
        ("negative_dependency", "positive_dependency", "shared_variable"),
        ("candidate -positive-> eligible", "blocked -negative-> eligible"),
        track_a_l2_combination_negative_filter, "track_evaluation",
        track="A", level=2, control_type="dependent_composition",
        context_family="candidate_filter_counterfactuals",
        pattern="combination>negative_filtering",
    ),
    TaskSpec(
        "track_a_l2_numeric_pipeline", ("combination", "numeric_filtering"),
        ("comparison", "positive_dependency", "shared_variable"),
        ("candidate -positive-> nearby", "distance -positive-> nearby"),
        track_a_l2_numeric_pipeline, "track_evaluation",
        track="A", level=2, control_type="dependent_composition",
        context_family="candidate_numeric_counterfactuals",
        pattern="combination>numeric_filtering",
    ),
    TaskSpec(
        "track_a_l3_closure_triage_blocked",
        ("closure", "filtering", "negative_filtering"),
        ("base_recursive", "negative_dependency", "positive_dependency",
         "recursive_scc", "shared_variable"),
        ("closure -positive-> reachable_triage",
         "reachable_triage -positive-> servable", "blocked -negative-> servable"),
        track_a_l3_closure_triage_blocked, "track_evaluation",
        track="A", level=3, control_type="dependent_composition",
        context_family="closure_reachable_unreachable_triage",
        pattern="closure>filtering>negative_filtering",
    ),
    TaskSpec(
        "track_a_l3_join_status_blocked",
        ("join", "filtering", "negative_filtering"),
        ("negative_dependency", "positive_dependency", "shared_variable"),
        ("matched -positive-> urgent_match", "urgent_match -positive-> eligible",
         "blocked -negative-> eligible"),
        track_a_l3_join_status_blocked, "track_evaluation",
        track="A", level=3, control_type="dependent_composition",
        context_family="join_status_filter_counterfactuals",
        pattern="join>filtering>negative_filtering",
    ),
    TaskSpec(
        "track_a_l3_combination_distance_blocked",
        ("combination", "numeric_filtering", "negative_filtering"),
        ("comparison", "negative_dependency", "positive_dependency", "shared_variable"),
        ("candidate -positive-> nearby", "nearby -positive-> recommended",
         "blocked -negative-> recommended"),
        track_a_l3_combination_distance_blocked, "track_evaluation",
        track="A", level=3, control_type="dependent_composition",
        context_family="candidate_distance_filter_counterfactuals",
        pattern="combination>numeric_filtering>negative_filtering",
    ),
    TaskSpec(
        "track_a_l3_seen_admission",
        ("assignment", "constraint", "preference"),
        ("comparison", "negative_dependency", "positive_dependency",
         "shared_variable", "weak_constraint"),
        ("injured -positive-> admit", "admit -positive-> constraint",
         "admit -negative-> weak_constraint"),
        track_a_l3_seen_admission, "track_evaluation",
        track="A", level=3, control_type="dependent_composition",
        optimization=True, context_family="seen_admission_ties",
        pattern="disjunction>capacity>fixed_preference",
    ),
    TaskSpec(
        "track_a_l3_length_matched_control",
        ("filtering", "join", "negative_filtering"),
        ("comparison", "negative_dependency", "positive_dependency", "shared_variable"),
        ("independent_rules",),
        track_a_l3_length_matched_control, "track_evaluation",
        track="A", level=3, control_type="length_matched_independent",
        context_family="independent_rule_id_permutations",
        pattern="independent_filter+join+negative+numeric",
    ),
)


TRACK_B_SPECS = (
    TaskSpec(
        "track_b_l1_choice", ("choice",),
        ("choice_rule", "positive_dependency", "shared_variable"),
        ("entity -positive-> assign",), track_b_l1_choice, "track_evaluation",
        track="B", level=1, control_type="atomic_control",
        context_family="choice_domain_permutations", pattern="choice",
    ),
    TaskSpec(
        "track_b_l1_count", ("aggregate",),
        ("aggregate", "comparison", "positive_dependency", "shared_variable"),
        ("allocation -aggregate-> overloaded",), track_b_l1_count, "track_evaluation",
        track="B", level=1, control_type="atomic_control",
        context_family="aggregate_count_counterfactuals", pattern="count",
    ),
    TaskSpec(
        "track_b_l1_variable_weak", ("variable_weak_constraint",),
        ("positive_dependency", "shared_variable", "weak_constraint"),
        ("dispatch -positive-> weak_constraint",),
        track_b_l1_variable_weak, "track_evaluation",
        track="B", level=1, control_type="atomic_control",
        optimization=True, context_family="variable_cost_counterfactuals",
        pattern="variable_weak_constraint",
    ),
    TaskSpec(
        "track_b_l1_count_constraint", ("aggregate_constraint",),
        ("aggregate", "comparison", "positive_dependency", "shared_variable"),
        ("admit -aggregate-> constraint",),
        track_b_l1_count_constraint, "track_evaluation",
        track="B", level=1, control_type="atomic_control",
        context_family="aggregate_constraint_counterfactuals",
        pattern="count_constraint",
    ),
    TaskSpec(
        "track_b_l2_choice_capacity", ("choice", "aggregate_constraint"),
        ("aggregate", "choice_rule", "comparison", "positive_dependency", "shared_variable"),
        ("admit -aggregate-> constraint",),
        track_b_l2_choice_capacity, "track_evaluation",
        track="B", level=2, control_type="dependent_composition",
        context_family="capacity_counterfactuals", pattern="choice>capacity",
    ),
    TaskSpec(
        "track_b_l2_choice_optimization", ("choice", "variable_weak_constraint"),
        ("choice_rule", "positive_dependency", "shared_variable", "weak_constraint"),
        ("admit -positive-> weak_constraint",),
        track_b_l2_choice_optimization, "track_evaluation",
        track="B", level=2, control_type="dependent_composition",
        optimization=True, context_family="choice_cost_ties",
        pattern="choice>variable_optimization",
    ),
    TaskSpec(
        "track_b_l2_choice_unavailable", ("choice", "constraint"),
        ("choice_rule", "positive_dependency", "shared_variable"),
        ("dispatch -positive-> constraint",),
        track_b_l2_choice_unavailable, "track_evaluation",
        track="B", level=2, control_type="dependent_composition",
        context_family="multiple_feasible_resources", pattern="choice>constraint",
    ),
    TaskSpec(
        "track_b_l2_candidate_count", ("combination", "aggregate"),
        ("aggregate", "comparison", "positive_dependency", "shared_variable"),
        ("candidate -aggregate-> crowded",),
        track_b_l2_candidate_count, "track_evaluation",
        track="B", level=2, control_type="dependent_composition",
        context_family="candidate_count_counterfactuals", pattern="combination>count",
    ),
    TaskSpec(
        "track_b_l3_admission",
        ("choice", "aggregate_constraint", "variable_weak_constraint"),
        ("aggregate", "choice_rule", "comparison", "negative_dependency",
         "positive_dependency", "shared_variable", "weak_constraint"),
        ("admit -aggregate-> constraint", "admit -negative-> weak_constraint"),
        track_b_l3_admission, "track_evaluation",
        track="B", level=3, control_type="dependent_composition",
        optimization=True, context_family="admission_capacity_priority_ties",
        pattern="choice>capacity>variable_optimization",
    ),
    TaskSpec(
        "track_b_l3_dispatch", ("choice", "constraint", "variable_weak_constraint"),
        ("choice_rule", "positive_dependency", "shared_variable", "weak_constraint"),
        ("dispatch -positive-> constraint", "dispatch -positive-> weak_constraint"),
        track_b_l3_dispatch, "track_evaluation",
        track="B", level=3, control_type="dependent_composition",
        optimization=True, context_family="dispatch_multiple_feasible_ties",
        pattern="choice>constraint>variable_optimization",
    ),
    TaskSpec(
        "track_b_l3_allocation",
        ("choice", "aggregate_constraint", "variable_weak_constraint"),
        ("aggregate", "choice_rule", "comparison", "positive_dependency",
         "shared_variable", "weak_constraint"),
        ("allocate -aggregate-> constraint", "allocate -positive-> weak_constraint"),
        track_b_l3_allocation, "track_evaluation",
        track="B", level=3, control_type="dependent_composition",
        optimization=True, context_family="allocation_capacity_cost_ties",
        pattern="choice>capacity>variable_optimization",
    ),
    TaskSpec(
        "track_b_l3_length_matched_control",
        ("choice", "aggregate_constraint", "variable_weak_constraint"),
        ("aggregate", "choice_rule", "comparison", "positive_dependency",
         "shared_variable", "weak_constraint"),
        ("independent_rules",),
        track_b_l3_length_matched_control, "track_evaluation",
        track="B", level=3, control_type="length_matched_independent",
        optimization=True, context_family="independent_rich_rule_permutations",
        pattern="independent_choice+count+constraint+weak",
    ),
)


def observed_features(program: str) -> set[str]:
    """Extract the local structural features used by the split protocol.

    This is intentionally a lightweight checker for programs emitted by this
    generator, not a general ASP parser; Clingo remains the syntax/grounding
    authority.  Its purpose is to prevent metadata labels from silently
    drifting away from their generated rules.
    """
    observed: set[str] = set()
    lines = [line.strip() for line in program.splitlines() if line.strip()]
    if any(
        re.search(r"[a-z][a-zA-Z0-9_]*\s*\(",
                  line.split(":-", 1)[1] if ":-" in line else line.split(":~", 1)[1])
        for line in lines if ":-" in line or ":~" in line
    ):
        observed.add("positive_dependency")
    if "not " in program:
        observed.add("negative_dependency")
    without_aggregates = re.sub(r"#count\s*\{[^}]*\}", "", program)
    if "{" in without_aggregates and "}" in without_aggregates:
        observed.add("choice_rule")
    if ":~" in program:
        observed.add("weak_constraint")
    if "#count" in program:
        observed.add("aggregate")
    if re.search(
        r"(?:>=|<=|!=|==|(?<!:)>(?!=)|<(?!=))\s*(?:-?\d+|[A-Z][A-Za-z0-9_]*)",
        program,
    ):
        observed.add("comparison")

    # A variable repeated in one rule connects at least two atom/term
    # positions and therefore realizes the shared-variable motif.
    for line in lines:
        variables = re.findall(r"\b[A-Z][A-Za-z0-9_]*\b", line)
        if len(variables) != len(set(variables)):
            observed.add("shared_variable")
            break

    recursive_heads: set[str] = set()
    base_heads: set[str] = set()
    for line in lines:
        if ":-" not in line or line.startswith(":-") or line.startswith(":~"):
            continue
        head, body = line.split(":-", 1)
        match = re.search(r"([a-z][a-zA-Z0-9_]*)\s*\(", head)
        if not match:
            continue
        predicate = match.group(1)
        if re.search(rf"\b{re.escape(predicate)}\s*\(", body):
            recursive_heads.add(predicate)
        else:
            base_heads.add(predicate)
    if recursive_heads:
        observed.add("recursive_scc")
    if recursive_heads & base_heads:
        observed.add("base_recursive")
    return observed


def _weak_weights_are_constant(program: str) -> bool:
    """True iff every weak-constraint weight is an integer literal.

    scenario-up only ever saw fixed-weight weak constraints (the preference
    pattern, e.g. ``[3@1,X]``); a variable weight such as ``[T@1,X]`` is Track B
    syntax.  In a ``[weight@priority, tuple]`` annotation the weight is the token
    before ``@``.
    """
    for match in re.finditer(r"\[\s*([^@\[\]]+?)\s*@", program):
        if not re.fullmatch(r"-?\d+", match.group(1).strip()):
            return False
    return True


def _has_cross_rule_dependency(program: str) -> bool:
    """Return whether a predicate produced by one rule is consumed by another."""
    lines = [line.strip() for line in program.splitlines() if line.strip()]
    heads: list[set[str]] = []
    bodies: list[str] = []
    for line in lines:
        if line.startswith(":-") or line.startswith(":~"):
            heads.append(set())
            bodies.append(line[2:])
            continue
        head, separator, body = line.partition(":-")
        choice = re.search(r"\{(.*?)\}", head)
        if choice:
            # Choice-rule head, with OR without a body (e.g. a bodyless
            # `1 { admit(X) : entity(X) } 1.`).  Strip each element's aggregate
            # condition (":cond") before reading the produced predicates.
            head_source = ";".join(
                element.split(":", 1)[0] for element in choice.group(1).split(";")
            )
        else:
            head_source = head
        heads.append(set(re.findall(r"\b([a-z][A-Za-z0-9_]*)\s*\(", head_source)))
        bodies.append(body)

    for producer_index, produced in enumerate(heads):
        for predicate in produced:
            for consumer_index, body in enumerate(bodies):
                if producer_index != consumer_index and re.search(
                    rf"\b{re.escape(predicate)}\s*\(", body
                ):
                    return True
    return False


def _infer_target_predicates(program: str) -> tuple[tuple[str, int], ...]:
    """Infer a conservative projection signature for legacy builders."""
    targets: list[tuple[str, int]] = []
    fallback_bodies: list[str] = []

    def add_first_atom(source: str) -> bool:
        match = re.search(r"\b([a-z][A-Za-z0-9_]*)\s*\(([^()]*)\)", source)
        if not match:
            return False
        arity = 0 if not match.group(2).strip() else match.group(2).count(",") + 1
        signature = (match.group(1), arity)
        if signature not in targets:
            targets.append(signature)
        return True

    for line in (part.strip() for part in program.splitlines() if part.strip()):
        if line.startswith(":-") or line.startswith(":~"):
            fallback_bodies.append(line[2:])
            continue
        head = line.split(":-", 1)[0]
        choice = re.search(r"\{(.*?)\}", head)
        if choice:
            add_first_atom(choice.group(1).split(":", 1)[0])
        else:
            for disjunct in head.split("|"):
                add_first_atom(disjunct)

    if not targets:
        for body in fallback_bodies:
            if add_first_atom(body):
                break
    return tuple(targets)


def validate_design() -> None:
    specs = (
        ATOMIC_SPECS + SEEN_COMPOSITE_SPECS + HELDOUT_COMPOSITE_SPECS
        + TRACK_A_SPECS + TRACK_B_SPECS
    )
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("task names must be unique")

    unknown = {feature for spec in specs for feature in spec.features} - FEATURES
    if unknown:
        raise ValueError(f"unknown structural features: {sorted(unknown)}")

    seen_combinations = {tuple(sorted(spec.components)) for spec in SEEN_COMPOSITE_SPECS}
    heldout_combinations = {tuple(sorted(spec.components)) for spec in HELDOUT_COMPOSITE_SPECS}
    leaked = seen_combinations & heldout_combinations
    if leaked:
        raise ValueError(f"held-out component combinations leaked into training: {sorted(leaked)}")

    training_features = {feature for spec in ATOMIC_SPECS + SEEN_COMPOSITE_SPECS
                         for feature in spec.features}
    test_features = {feature for spec in HELDOUT_COMPOSITE_SPECS for feature in spec.features}
    missing = test_features - training_features
    if missing:
        raise ValueError(f"held-out tests contain unseen local features: {sorted(missing)}")

    atomic_components = {component for spec in ATOMIC_SPECS for component in spec.components}
    missing_components = {component for spec in HELDOUT_COMPOSITE_SPECS
                          for component in spec.components} - atomic_components
    if missing_components:
        raise ValueError(f"held-out tests contain unseen atomic components: {sorted(missing_components)}")

    level_counts = Counter((spec.track, spec.level) for spec in TRACK_A_SPECS + TRACK_B_SPECS)
    underfilled = {
        f"{track}-L{level}": level_counts[(track, level)]
        for track in ("A", "B")
        for level in (1, 2, 3)
        if level_counts[(track, level)] < 4
    }
    if underfilled:
        raise ValueError(f"each track/level needs at least four families: {underfilled}")

    for spec in TRACK_A_SPECS:
        example = spec.builder(random.Random(17), TRAIN_VOCAB)
        feats = observed_features(example.answer)
        # scenario-up never saw brace-choice or #count aggregates.
        forbidden = {"choice_rule", "aggregate"} & feats
        if forbidden:
            raise ValueError(
                f"{spec.name} uses syntax unseen by scenario-up "
                f"(choice/aggregate): {sorted(forbidden)}"
            )
        # It DID see fixed-weight weak constraints (the preference pattern,
        # including `:~ p(X), not assign(X,"l"). [1@1,X]`), but never
        # variable-weight ones such as `[T@1,X]`; keep those in Track B.
        if "weak_constraint" in feats and not _weak_weights_are_constant(example.answer):
            raise ValueError(
                f"{spec.name} uses a variable-weight weak constraint (Track B only)"
            )

    for spec in TRACK_A_SPECS + TRACK_B_SPECS:
        if spec.level >= 2 and spec.control_type == "dependent_composition":
            example = spec.builder(random.Random(19), TRAIN_VOCAB)
            if not _has_cross_rule_dependency(example.answer):
                raise ValueError(f"{spec.name} lacks a real cross-rule dependency")

    # Explicit optimization metadata is authoritative at evaluation time; here we
    # assert it agrees with the program syntax so the two never silently drift.
    for spec in TRACK_A_SPECS + TRACK_B_SPECS:
        if spec.optimization is not None:
            example = spec.builder(random.Random(23), TRAIN_VOCAB)
            syntactic = bool(re.search(r":~|#\s*(?:minimize|maximize)\b", example.answer))
            if bool(spec.optimization) != syntactic:
                raise ValueError(
                    f"{spec.name} optimization metadata ({spec.optimization}) "
                    f"disagrees with program syntax ({syntactic})"
                )

    sv.assert_clean()

    # Check both lexical regimes because builders may select different symbols
    # while their structural metadata must remain invariant.
    for spec in specs:
        for vocab in (TRAIN_VOCAB, TEST_VOCAB):
            example = spec.builder(random.Random(13), vocab)
            observed = observed_features(example.answer)
            missing_markers = set(spec.features) - observed
            if missing_markers:
                raise ValueError(
                    f"{spec.name} declares features not found in its program: {sorted(missing_markers)}")


def _metadata(spec: TaskSpec, example: Example, split: str, vocabulary_split: str) -> dict:
    rich_features = {"choice_rule", "aggregate", "weak_constraint"}
    track = spec.track
    if track == "auto":
        track = "B" if rich_features & set(spec.features) else "A"
    level = spec.level or min(3, max(1, len(spec.components)))
    if spec.control_type == "auto":
        control_type = {
            "atomic": "atomic_control",
            "seen_composite": "seen_composition",
            "heldout_composite": "heldout_composition",
        }.get(spec.kind, "evaluation")
    else:
        control_type = spec.control_type
    targets = example.target_predicates or spec.target_predicates
    if not targets:
        targets = _infer_target_predicates(example.answer)
    optimization = spec.optimization if spec.optimization is not None else ":~" in example.answer
    context_family = spec.context_family or spec.name
    pattern = spec.pattern or ">".join(spec.components)
    return {
        "task": spec.name,
        "pattern": pattern,
        "track": track,
        "level": level,
        "control_type": control_type,
        "target_predicates": json.dumps(
            [{"name": name, "arity": arity} for name, arity in targets],
            ensure_ascii=False,
        ),
        "optimization": optimization,
        "context_family": context_family,
        "components": json.dumps(spec.components),
        "features": json.dumps(spec.features),
        "dependency_edges": json.dumps(spec.dependency_edges),
        "rule_count": len(example.answer.splitlines()),
        "composition_split": spec.kind,
        "vocabulary_split": vocabulary_split,
        "split": split,
    }


def _fact_signatures(facts: str) -> list[str]:
    """Extract ``name/arity`` signatures from a ground fact string.

    Arity-aware: a vocab draw can collide with a hardcoded predicate name
    (e.g. the resource pool word "dispatch" next to the constraint tasks'
    decision predicate dispatch/2), giving the same name two arities — both
    must be listed, or the interface line misclassifies one of them.
    """
    signatures: set[tuple[str, int]] = set()
    for match in re.finditer(r"\b([a-z][A-Za-z0-9_]*)\s*\(([^()]*)\)", facts or ""):
        name, args = match.group(1), match.group(2)
        arity, depth, in_string = 1, 0, False
        index = 0
        while index < len(args):
            char = args[index]
            if in_string:
                if char == "\\":
                    index += 2
                    continue
                if char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char in "([":
                depth += 1
            elif char in ")]":
                depth -= 1
            elif char == "," and depth == 0:
                arity += 1
            index += 1
        if not args.strip():
            arity = 0
        signatures.add((name, arity))
    return [f"{name}/{arity}" for name, arity in sorted(signatures)]


def _interface_line(example: Example) -> str:
    """Declare the input schema and the REQUIRED output predicate names.

    Without this, a task silently tests whether the model guesses the reference
    naming convention and fact arity (e.g. writing the trained ``select/1`` over
    a binary ``transport(X,C)`` when the reference expects ``selected/1`` over a
    unary ``transport(X)`` plus ``road(X,"...")``).  That confound has nothing to
    do with compositional ability, so the interface is stated explicitly and the
    evaluation still only observes the declared output predicates.
    """
    parts = []
    inputs = _fact_signatures(example.facts)
    input_set = set(inputs)
    if inputs:
        parts.append(f"Input predicates: {', '.join(inputs)}.")
    outputs = [f"{name}/{arity}" for name, arity in example.target_predicates]
    # An observed predicate that is ALSO an input (e.g. ``dispatch/2`` in the
    # constraint/preference tasks, where the model must constrain given facts
    # rather than define a new predicate) must not be presented as an output
    # to "name": that instruction would be misleading and unfollowable.
    declared = [signature for signature in outputs if signature not in input_set]
    observed = [signature for signature in outputs if signature in input_set]
    if declared:
        parts.append(
            f"Name the output predicate{'s' if len(declared) > 1 else ''} exactly: "
            f"{', '.join(declared)}."
        )
    if observed:
        plural = len(observed) > 1
        parts.append(
            f"The evaluation observes {', '.join(observed)} as given by the facts; "
            f"constrain {'them' if plural else 'it'} rather than define "
            f"{'them' if plural else 'it'}."
        )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Training-time surface diversification (ATTRIBUTION_AND_ROADMAP.md §2.6).
#
# The phrasing-control experiment showed the scenario models act as
# stimulus-response templates: a fixed head name per family becomes a naming
# REFLEX that overrides the interface declaration (25/39 hybrid
# negative-filtering failures wrote the trained ``select`` over the declared
# ``available``), and a single fixed preamble makes every surface cue
# memorizable.  Training rows therefore (a) rename declared output predicates
# in half the rows so the declaration must be READ, and (b) vary the preamble.
# Test rows keep their canonical names and preamble.  HOWEVER, materializing
# resolved targets in _row means EVERY row — test rows included — now carries
# the "Name the output predicate(s) exactly" declaration.  Track A/B test
# prompts already declared outputs and are byte-identical to earlier rounds,
# but 9 seen/heldout composite tasks whose builders left targets empty
# (assignment_constraint, combination_filtering, join_negative_filtering,
# closure_constraint, filtering_preference, assignment_aggregate_comparison,
# assignment_preference, closure_filtering, combination_aggregate_comparison)
# gain the declaration sentence: their scores are NOT comparable to earlier
# rounds without re-baselining the old model on the regenerated CSVs.  The
# manifest records this as protocol["universal_output_declaration"].
# ---------------------------------------------------------------------------

OUTPUT_NAME_POOL = (
    "picked", "flagged", "kept", "marked", "wanted", "chosen", "noted",
    "tagged", "shortlist", "held", "cleared", "queued", "slotted", "binned",
    "sorted", "ranked",
)
RENAME_PROBABILITY = 0.5

TRAIN_PREAMBLE_VARIANTS = (
    "Write an ASP program for the following medical-rescue problem.",
    "Develop an ASP program for the following medical-rescue problem.",
    "Create an ASP program that solves this medical-rescue task.",
    "Construct an ASP encoding for the medical-rescue problem below.",
    "Produce an answer set program for this medical-rescue scenario.",
    "Formulate an ASP program for the described medical-rescue problem.",
    "Give an ASP program solving the following medical-rescue problem.",
    "Encode the following medical-rescue problem as an ASP program.",
)


def _resolve_targets(spec: TaskSpec, example: Example) -> tuple[tuple[str, int], ...]:
    """Resolve targets with the same fallback chain _metadata uses."""
    targets = example.target_predicates or spec.target_predicates
    if not targets:
        targets = _infer_target_predicates(example.answer)
    return tuple((str(name), int(arity)) for name, arity in targets)


def _rename_output_predicates(example: Example, rng: random.Random) -> Example:
    if not example.target_predicates or rng.random() >= RENAME_PROBABILITY:
        return example
    text_blob = " ".join(
        (example.question, example.answer, example.facts, example.background))
    taken = set(re.findall(r"\b[a-z][A-Za-z0-9_]*\b", text_blob))
    # An output predicate that also occurs in the facts (or any hidden-context
    # facts) is an EDB predicate observed as-is (e.g. ``dispatch`` in the
    # constraint/preference tasks).  Renaming it in the rules but not in the
    # facts would silently change the semantics, so such names are kept.
    fact_tokens = set(re.findall(r"\b[a-z][A-Za-z0-9_]*\b", example.facts))
    for context in example.hidden_contexts:
        fact_tokens |= set(re.findall(r"\b[a-z][A-Za-z0-9_]*\b", context.get("facts", "")))
    renames: dict[str, str] = {}
    for name, _arity in example.target_predicates:
        if name in fact_tokens:
            continue
        candidates = [c for c in OUTPUT_NAME_POOL
                      if c not in taken and c not in renames.values()]
        if not candidates:
            return example
        renames[name] = rng.choice(candidates)
    if not renames:
        return example

    def rewrite(text: str) -> str:
        for old, new in renames.items():
            text = re.sub(rf"\b{re.escape(old)}\b", new, text)
        return text

    return replace(
        example,
        question=rewrite(example.question),
        answer=rewrite(example.answer),
        background=rewrite(example.background),
        target_predicates=tuple(
            (renames.get(name, name), arity)
            for name, arity in example.target_predicates),
    )


def _vary_preamble(example: Example, rng: random.Random) -> Example:
    canonical = TRAIN_PREAMBLE_VARIANTS[0]
    if not example.question.startswith(canonical):
        return example
    opener = rng.choice(TRAIN_PREAMBLE_VARIANTS)
    return replace(example, question=opener + example.question[len(canonical):])


def _row(spec: TaskSpec, rng: random.Random, vocab: Vocabulary, split: str,
         vocabulary_split: str, training: bool) -> dict:
    example = spec.builder(rng, vocab)
    # Materialize the resolved projection signature on the example so that the
    # interface line declares the outputs for EVERY row (legacy builders with
    # empty targets previously produced training questions with no output
    # declaration, which is exactly the reflex-forming setup §2.6 indicts).
    example = replace(example, target_predicates=_resolve_targets(spec, example))
    if training:
        example = _rename_output_predicates(example, rng)
        example = _vary_preamble(example, rng)
    interface = _interface_line(example)
    if interface:
        example = replace(example, question=f"{example.question} {interface}")
    row = _metadata(spec, example, split, vocabulary_split)
    contexts = example.hidden_contexts or _default_hidden_contexts(example.facts)
    context_metadata = {
        "family": row["context_family"],
        "count": len(contexts),
        "materialized": not training,
        "coverage_tags": sorted({
            tag for context in contexts for tag in context.get("tags", [])
        }),
    }
    row["hidden_context_count"] = len(contexts)
    row["hidden_contexts"] = (
        "[]" if training else json.dumps(contexts, ensure_ascii=False, separators=(",", ":"))
    )
    row["context_metadata"] = json.dumps(
        context_metadata, ensure_ascii=False, separators=(",", ":")
    )
    # Prepended to BOTH reference and candidate at solve time (isolated controls).
    row["background_asp"] = example.background
    if training:
        # Training rows also carry their facts: the trainer only reads
        # question/answer, but the clingo validator needs the facts to be
        # meaningful.  Without them a cardinality-bounded choice such as
        # `1 { admit(X) : e(X) } 1.` is trivially UNSAT (nothing to choose from)
        # and would be reported as a broken program.
        row = {"question": example.question, "answer": example.answer,
               "fact": example.facts, **row}
    else:
        row = {"prompt": example.question, "answer": example.answer, "fact": example.facts, **row}
    return row


def _balanced_rows(specs: Sequence[TaskSpec], size: int, rng: random.Random,
                   vocab: Vocabulary, split: str, vocabulary_split: str,
                   training: bool) -> list[dict]:
    rows = []
    order = list(specs)
    rng.shuffle(order)
    for i in range(size):
        if i and i % len(order) == 0:
            rng.shuffle(order)
        rows.append(_row(order[i % len(order)], rng, vocab, split, vocabulary_split, training))
    rng.shuffle(rows)
    return rows


def training_rows(size: int, seed: int, atomic_ratio: float, split: str) -> list[dict]:
    if not 0.0 <= atomic_ratio <= 1.0:
        raise ValueError("atomic_ratio must lie in [0, 1]")
    rng = random.Random(seed)
    n_atomic = round(size * atomic_ratio)
    rows = _balanced_rows(ATOMIC_SPECS, n_atomic, rng, TRAIN_VOCAB, split, "iid", True)
    rows += _balanced_rows(SEEN_COMPOSITE_SPECS, size - n_atomic, rng, TRAIN_VOCAB,
                           split, "iid", True)
    rng.shuffle(rows)
    return rows


def test_rows(specs: Sequence[TaskSpec], per_task: int, seed: int,
              vocab: Vocabulary, composition_split: str, vocabulary_split: str) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    # Keep a stable round-robin task order for human inspection while metadata
    # makes the evaluator independent of row order.
    for _ in range(per_task):
        for spec in specs:
            rows.append(_row(spec, rng, vocab, composition_split, vocabulary_split, False))
    return rows


def _validate_with_clingo(
    rows: Sequence[dict],
    label: str,
    validate_hidden_contexts: bool = False,
) -> None:
    try:
        from clingo import Control
    except ImportError as exc:  # pragma: no cover - dependency is in LLASP requirements
        raise RuntimeError("clingo is required to validate generated ASP programs") from exc

    def validate_program(row: dict, row_label: str, facts: str) -> None:
        program = row["answer"] + "\n" + facts
        messages: list[str] = []
        ctl = Control(["1", "--warn=none"],
                      logger=lambda _code, message: messages.append(message), message_limit=20)
        try:
            ctl.add("base", [], program)
            ctl.ground([("base", [])])
            result = ctl.solve()
        except Exception as exc:
            detail = "\n".join(messages)
            raise ValueError(
                f"{label} {row_label} ({row['task']}) is invalid: {detail}"
            ) from exc
        if not result.satisfiable:
            raise ValueError(
                f"{label} {row_label} ({row['task']}) is unexpectedly unsatisfiable"
            )

    for i, row in enumerate(rows):
        validate_program(row, f"row {i}", row.get("fact", ""))
        if validate_hidden_contexts:
            contexts = json.loads(row.get("hidden_contexts", "[]"))
            if not 10 <= len(contexts) <= 20:
                raise ValueError(
                    f"{label} row {i} ({row['task']}) has {len(contexts)} hidden contexts"
                )
            for context in contexts:
                validate_program(
                    row,
                    f"row {i} hidden context {context.get('id', '<unknown>')}",
                    context["facts"],
                )


def _coverage(rows: Sequence[dict]) -> dict:
    task_counts = Counter(row["task"] for row in rows)
    feature_counts: Counter[str] = Counter()
    track_level_counts: Counter[str] = Counter()
    control_counts: Counter[str] = Counter()
    for row in rows:
        feature_counts.update(json.loads(row["features"]))
        track_level_counts[f"{row['track']}-L{row['level']}"] += 1
        control_counts[row["control_type"]] += 1
    return {
        "rows": len(rows),
        "tasks": dict(sorted(task_counts.items())),
        "features": dict(sorted(feature_counts.items())),
        "track_levels": dict(sorted(track_level_counts.items())),
        "control_types": dict(sorted(control_counts.items())),
    }


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[write] {path} ({len(rows)} rows)")


def build_datasets(output_dir: Path, train_size: int, val_size: int,
                   test_size_per_task: int, seed: int, atomic_ratio: float,
                   validate_clingo: bool = True) -> dict:
    validate_design()
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "train_compositional.csv": training_rows(train_size, seed, atomic_ratio, "train"),
        "val_compositional.csv": training_rows(val_size, seed + 1, atomic_ratio, "validation"),
        "test_compositional_seen_iid.csv": test_rows(
            SEEN_COMPOSITE_SPECS, test_size_per_task, seed + 2, TRAIN_VOCAB, "seen", "iid"),
        "test_compositional_seen_ood.csv": test_rows(
            SEEN_COMPOSITE_SPECS, test_size_per_task, seed + 3, TEST_VOCAB, "seen", "ood"),
        "test_compositional_heldout_iid.csv": test_rows(
            HELDOUT_COMPOSITE_SPECS, test_size_per_task, seed + 4, TRAIN_VOCAB, "heldout", "iid"),
        "test_compositional_heldout_ood.csv": test_rows(
            HELDOUT_COMPOSITE_SPECS, test_size_per_task, seed + 5, TEST_VOCAB, "heldout", "ood"),
        "test_compositional_track_a_iid.csv": test_rows(
            TRACK_A_SPECS, test_size_per_task, seed + 6, TRAIN_VOCAB, "track_a", "iid"),
        "test_compositional_track_a_ood.csv": test_rows(
            TRACK_A_SPECS, test_size_per_task, seed + 7, TEST_VOCAB, "track_a", "ood"),
        "test_compositional_track_b_iid.csv": test_rows(
            TRACK_B_SPECS, test_size_per_task, seed + 8, TRAIN_VOCAB, "track_b", "iid"),
        "test_compositional_track_b_ood.csv": test_rows(
            TRACK_B_SPECS, test_size_per_task, seed + 9, TEST_VOCAB, "track_b", "ood"),
    }

    if validate_clingo:
        # Validate all rows for small/smoke datasets.  For very large generated
        # sets, symbol substitutions do not change the rule grammar, so one row
        # per task is enough; every hidden context is still validated for that
        # representative row.
        for name, rows in datasets.items():
            if len(rows) <= 5000:
                check_rows = rows
            else:
                first_by_task = {}
                for row in rows:
                    first_by_task.setdefault(row["task"], row)
                check_rows = list(first_by_task.values())
            _validate_with_clingo(check_rows, name)
            if name.startswith("test_"):
                first_by_task = {}
                for row in rows:
                    first_by_task.setdefault(row["task"], row)
                _validate_with_clingo(
                    list(first_by_task.values()),
                    f"{name} hidden-context suite",
                    validate_hidden_contexts=True,
                )

    for name, rows in datasets.items():
        _write_csv(output_dir / name, rows)

    manifest = {
        "protocol": {
            "seed": seed,
            "atomic_ratio": atomic_ratio,
            "train_vocabulary_disjoint_from_test": True,
            "heldout_exact_compositions_absent_from_training": True,
            "all_heldout_atomic_components_seen_in_training": True,
            "all_heldout_local_features_seen_in_training": True,
            "explicit_track_a_scenario_up_syntax": True,
            "explicit_track_b_rich_syntax": True,
            "hidden_contexts_per_evaluation_program": "10-20",
            # Benchmark v2: every row (test rows included) declares its output
            # predicates; 9 seen/heldout prompts differ from pre-v2 rounds.
            "universal_output_declaration": True,
        },
        "atomic_specs": [asdict(spec) | {"builder": spec.builder.__name__}
                         for spec in ATOMIC_SPECS],
        "seen_composite_specs": [asdict(spec) | {"builder": spec.builder.__name__}
                                 for spec in SEEN_COMPOSITE_SPECS],
        "heldout_composite_specs": [asdict(spec) | {"builder": spec.builder.__name__}
                                    for spec in HELDOUT_COMPOSITE_SPECS],
        "track_a_specs": [asdict(spec) | {"builder": spec.builder.__name__}
                          for spec in TRACK_A_SPECS],
        "track_b_specs": [asdict(spec) | {"builder": spec.builder.__name__}
                          for spec in TRACK_B_SPECS],
        "datasets": {name: _coverage(rows) for name, rows in datasets.items()},
    }
    # asdict includes a non-serializable function; the union above replaces it.
    for group in (
        "atomic_specs", "seen_composite_specs", "heldout_composite_specs",
        "track_a_specs", "track_b_specs",
    ):
        for item in manifest[group]:
            item["builder"] = str(item["builder"])
    manifest_path = output_dir / "compositional_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[write] {manifest_path}")
    return datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data", type=Path)
    parser.add_argument("--train-size", default=int(os.environ.get("COMPOSITIONAL_TRAIN_SIZE", "100000")),
                        type=int, help="total training rows")
    parser.add_argument("--val-size", default=int(os.environ.get("COMPOSITIONAL_VAL_SIZE", "20000")),
                        type=int, help="total validation rows")
    parser.add_argument("--test-size-per-task",
                        default=int(os.environ.get("COMPOSITIONAL_TEST_PER_TASK", "1000")),
                        type=int)
    parser.add_argument("--atomic-ratio", default=0.35, type=float,
                        help="fraction of train/validation rows devoted to atomic controls")
    parser.add_argument("--seed", default=20260713, type=int)
    parser.add_argument("--skip-clingo-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_datasets(args.output_dir, args.train_size, args.val_size,
                   args.test_size_per_task, args.seed, args.atomic_ratio,
                   validate_clingo=not args.skip_clingo_validation)


if __name__ == "__main__":
    main()
