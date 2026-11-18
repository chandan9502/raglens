import random

import pytest

from raglens.metrics import binary_metrics, roc_auc


def test_binary_metrics_counts_and_scores():
    flagged = [True, True, False, False, True]
    unsupported = [True, False, True, False, True]
    m = binary_metrics(flagged, unsupported)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 1)
    assert m["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["f1"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["accuracy"] == 0.6


def test_binary_metrics_nothing_flagged_is_zero_not_an_error():
    m = binary_metrics([False, False], [True, False])
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0
    assert m["accuracy"] == 0.5


def test_binary_metrics_rejects_bad_input():
    with pytest.raises(ValueError):
        binary_metrics([True], [True, False])
    with pytest.raises(ValueError):
        binary_metrics([], [])


def test_auc_perfect_reversed_and_ties():
    unsupported = [True, True, False, False]
    assert roc_auc([0.1, 0.2, 0.8, 0.9], unsupported) == 1.0  # unsupported score lower: perfect
    assert roc_auc([0.8, 0.9, 0.1, 0.2], unsupported) == 0.0  # perfectly backwards
    assert roc_auc([0.5, 0.5, 0.5, 0.5], unsupported) == 0.5  # all ties: chance


def test_auc_is_none_with_one_class():
    assert roc_auc([0.1, 0.2], [True, True]) is None
    assert roc_auc([0.1, 0.2], [False, False]) is None


def test_auc_matches_brute_force_pair_counting():
    rng = random.Random(0)
    scores = [round(rng.random(), 1) for _ in range(200)]  # rounding creates many ties
    unsupported = [rng.random() < 0.3 for _ in range(200)]
    pos = [s for s, u in zip(scores, unsupported, strict=True) if u]
    neg = [s for s, u in zip(scores, unsupported, strict=True) if not u]
    wins = sum(1.0 if p < n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    assert roc_auc(scores, unsupported) == pytest.approx(wins / (len(pos) * len(neg)), abs=1e-4)
