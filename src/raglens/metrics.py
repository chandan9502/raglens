"""Small, dependency-free metrics for comparing RAGLens with human labels.

Convention used everywhere in RAGLens: the *positive* class is "unsupported".
A "flag" is RAGLens saying "this answer needs a human to look at it".

- precision: of the answers RAGLens flagged, how many were really unsupported?
- recall: of the really unsupported answers, how many did RAGLens flag?
- F1: the harmonic mean of precision and recall (one number to compare settings).
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Sequence


def binary_metrics(flagged: Sequence[bool], unsupported: Sequence[bool]) -> dict[str, float]:
    """Compare RAGLens flags with labels.

    ``flagged[i]`` is True if RAGLens flagged example ``i``; ``unsupported[i]``
    is True if a human labelled it unsupported. Precision, recall and F1 are
    0.0 when they are undefined (for example, nothing was flagged), which is
    also scikit-learn's default.

    Raises:
        ValueError: if the two sequences differ in length or are empty.
    """
    if len(flagged) != len(unsupported):
        raise ValueError("flagged and unsupported must have the same length")
    if not flagged:
        raise ValueError("need at least one example")

    tp = sum(1 for f, u in zip(flagged, unsupported, strict=True) if f and u)  # caught
    fp = sum(1 for f, u in zip(flagged, unsupported, strict=True) if f and not u)  # false alarm
    fn = sum(1 for f, u in zip(flagged, unsupported, strict=True) if not f and u)  # missed
    tn = len(flagged) - tp - fp - fn  # correctly left alone

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round((tp + tn) / len(flagged), 4),
    }


def roc_auc(scores: Sequence[float], unsupported: Sequence[bool]) -> float | None:
    """How well a score separates unsupported from supported answers.

    Returns the chance that a randomly chosen unsupported answer gets a
    *lower* score than a randomly chosen supported one (ties count as half).
    1.0 means the score ranks them perfectly, 0.5 means no better than a coin
    flip. This is the ROC AUC, and it does not depend on any threshold.

    Returns None when only one class is present, because the value is undefined.
    """
    if len(scores) != len(unsupported):
        raise ValueError("scores and unsupported must have the same length")
    pos = [s for s, u in zip(scores, unsupported, strict=True) if u]
    neg = sorted(s for s, u in zip(scores, unsupported, strict=True) if not u)
    if not pos or not neg:
        return None

    wins = 0.0
    for score in pos:
        below_or_equal = bisect_right(neg, score)  # supported answers scoring <= this one
        below = bisect_left(neg, score)  # supported answers scoring < this one
        higher = len(neg) - below_or_equal  # supported answers scoring higher: a "win"
        ties = below_or_equal - below
        wins += higher + 0.5 * ties
    return round(wins / (len(pos) * len(neg)), 4)
