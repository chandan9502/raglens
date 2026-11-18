"""Choose verdict thresholds from labelled examples.

The default thresholds are hand-picked. Calibration tries every threshold on a
0.01 grid, measures how well the resulting verdicts match human labels, and
reports the one with the best F1. Scores are computed once per example; only
the cheap verdict step is repeated for each candidate threshold.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from .evaluator import DEFAULT_THRESHOLDS, VERDICTS, EvaluationReport, Thresholds, decide_verdict
from .metrics import binary_metrics, roc_auc

# The verdict levels that --fail-on accepts, mapped to the threshold that moves
# that boundary. "partially_supported" flags anything that is not
# likely_supported, so the support threshold decides it. "needs_review" flags
# only the worst verdict, so the review threshold decides it.
TUNED_THRESHOLD = {"partially_supported": "support", "needs_review": "review"}


def is_flagged(verdict: str, fail_on: str) -> bool:
    """True if ``verdict`` is ``fail_on`` or worse (the meaning of ``--fail-on``)."""
    return VERDICTS.index(verdict) >= VERDICTS.index(fail_on)


def _candidates(fail_on: str, base: Thresholds) -> list[float]:
    """Thresholds to try, on a 0.01 grid, keeping review <= support valid."""
    # round(..., 6) removes float noise such as 0.29 * 100 == 28.999999999999996.
    if fail_on == "partially_supported":
        low = math.ceil(round(base.review * 100, 6))
        return [i / 100 for i in range(low, 101)]
    high = math.floor(round(base.support * 100, 6))
    return [i / 100 for i in range(0, high + 1)]


def calibrate(
    reports: Sequence[EvaluationReport],
    unsupported: Sequence[bool],
    fail_on: str = "partially_supported",
    base: Thresholds = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Find the threshold that best matches labels for one ``--fail-on`` level.

    Args:
        reports: one report per labelled example (from :func:`raglens.evaluate`).
        unsupported: True where a human labelled the answer unsupported.
        fail_on: the verdict level you would pass to ``--fail-on``.
        base: thresholds kept fixed while one of them is varied.

    Returns:
        A dict with the metrics at the current threshold (``current``), at the
        best threshold (``best``), the full ``curve`` and ``answer_support_auc``.

    Raises:
        ValueError: on mismatched lengths, an unknown ``fail_on``, or labels
            that contain only one class.
    """
    if fail_on not in TUNED_THRESHOLD:
        raise ValueError(f"fail_on must be one of {sorted(TUNED_THRESHOLD)}, got {fail_on!r}")
    if len(reports) != len(unsupported):
        raise ValueError("reports and unsupported must have the same length")
    if not any(unsupported) or all(unsupported):
        raise ValueError("calibration needs both supported and unsupported labelled examples")

    name = TUNED_THRESHOLD[fail_on]

    def measure(thresholds: Thresholds) -> dict[str, float]:
        verdicts = [
            decide_verdict(r.answer_support, r.context_relevance, bool(r.warnings), thresholds, r.abstained)
            for r in reports
        ]
        return binary_metrics([is_flagged(v, fail_on) for v in verdicts], unsupported)

    current = getattr(base, name)
    curve = [{"threshold": t, **measure(replace(base, **{name: t}))} for t in _candidates(fail_on, base)]
    # Best F1; on a tie prefer higher precision, then the value closest to the current one.
    best = max(curve, key=lambda row: (row["f1"], row["precision"], -abs(row["threshold"] - current)))

    return {
        "fail_on": fail_on,
        "tuned_threshold": name,
        "examples": len(reports),
        "unsupported_examples": sum(1 for u in unsupported if u),
        "current": {"threshold": current, **measure(base)},
        "best": best,
        "answer_support_auc": roc_auc([r.answer_support for r in reports], unsupported),
        "curve": curve,
    }
