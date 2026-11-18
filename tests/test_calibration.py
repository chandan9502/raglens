import math

import pytest

from raglens import DEFAULT_THRESHOLDS, EvaluationReport, Thresholds, calibrate, evaluate
from raglens.calibration import _candidates, is_flagged
from raglens.evaluator import decide_verdict

PARAPHRASE = (
    "Is the API rate limited?",
    "The public API allows 100 requests per minute for each key.",
    "Yes, each key can make 100 requests per minute.",  # answer_support 0.75
)


def _report(support: float, relevance: float = 1.0, warnings: tuple[str, ...] = ()) -> EvaluationReport:
    """A report with chosen scores, so calibration tests do not depend on tokenization."""
    return EvaluationReport(
        question="q",
        context_relevance=relevance,
        answer_support=support,
        unsupported_terms=[],
        verdict=decide_verdict(support, relevance, bool(warnings)),
        explanations=[],
        limitations=[],
        warnings=list(warnings),
    )


# --- Thresholds -------------------------------------------------------------


def test_default_thresholds_are_unchanged():
    assert DEFAULT_THRESHOLDS == Thresholds(support=0.8, review=0.5, relevance=0.25)


@pytest.mark.parametrize(
    "kwargs",
    [{"support": 1.5}, {"review": -0.1}, {"relevance": math.nan}, {"support": True}, {"support": "0.8"}],
)
def test_thresholds_reject_invalid_values(kwargs):
    with pytest.raises(ValueError):
        Thresholds(**kwargs)


def test_review_cannot_exceed_support():
    with pytest.raises(ValueError, match="review threshold"):
        Thresholds(support=0.6, review=0.7)


# --- decide_verdict -----------------------------------------------------------


@pytest.mark.parametrize(
    ("support", "relevance", "warn", "expected"),
    [
        (0.8, 0.25, False, "likely_supported"),  # both exactly on the boundary
        (0.7999, 1.0, False, "partially_supported"),
        (0.5, 1.0, False, "partially_supported"),
        (0.4999, 1.0, False, "needs_review"),
        (1.0, 0.2499, False, "needs_review"),
        (1.0, 1.0, True, "partially_supported"),  # a warning blocks likely_supported...
        (0.4, 1.0, True, "needs_review"),  # ...but never makes a verdict better
    ],
)
def test_decide_verdict_boundaries(support, relevance, warn, expected):
    assert decide_verdict(support, relevance, warn) == expected


def test_custom_threshold_changes_verdict_but_not_scores():
    default = evaluate(*PARAPHRASE)
    relaxed = evaluate(*PARAPHRASE, thresholds=Thresholds(support=0.7))
    assert default.answer_support == relaxed.answer_support == 0.75
    assert default.verdict == "partially_supported"
    assert relaxed.verdict == "likely_supported"


def test_verdict_agrees_with_the_rounded_scores_in_the_report():
    cases = [
        PARAPHRASE,
        ("What is the request timeout?", "Requests time out after 30 seconds.", "They time out after 60."),
        ("Where is the office?", "The company's only office is in Glasgow.", "The office is in Edinburgh."),
    ]
    for case in cases:
        r = evaluate(*case)
        assert r.verdict == decide_verdict(r.answer_support, r.context_relevance, bool(r.warnings))


# --- calibrate ------------------------------------------------------------------


def test_is_flagged_follows_fail_on_meaning():
    assert is_flagged("needs_review", "partially_supported")
    assert is_flagged("partially_supported", "partially_supported")
    assert not is_flagged("likely_supported", "partially_supported")
    assert not is_flagged("partially_supported", "needs_review")


def test_calibrate_finds_the_support_threshold_that_separates_the_labels():
    # Unsupported answers score 0.82-0.90, supported ones 0.95-1.0. Any support
    # threshold in (0.90, 0.95] separates them; 0.91 is the closest to 0.8.
    reports = [_report(s) for s in (0.82, 0.85, 0.90, 0.95, 1.0, 1.0)]
    unsupported = [True, True, True, False, False, False]
    result = calibrate(reports, unsupported, fail_on="partially_supported")
    assert result["tuned_threshold"] == "support"
    assert result["current"]["threshold"] == 0.8
    assert result["current"]["f1"] == 0.0  # at 0.8 nothing is flagged
    assert result["best"]["threshold"] == 0.91
    assert result["best"]["f1"] == 1.0
    assert result["answer_support_auc"] == 1.0


def test_calibrate_needs_review_tunes_the_review_threshold():
    # Separated by any review threshold in (0.58, 0.70]; 0.59 is the closest to 0.5.
    reports = [_report(s) for s in (0.55, 0.58, 0.70, 0.75)]
    result = calibrate(reports, [True, True, False, False], fail_on="needs_review")
    assert result["tuned_threshold"] == "review"
    assert result["best"]["threshold"] == 0.59
    assert result["best"]["f1"] == 1.0


def test_calibrated_threshold_reproduces_its_measured_metrics():
    reports = [_report(s) for s in (0.82, 0.85, 0.90, 0.95, 1.0, 1.0)]
    unsupported = [True, True, True, False, False, False]
    best = calibrate(reports, unsupported)["best"]
    thresholds = Thresholds(support=best["threshold"])
    verdicts = [decide_verdict(r.answer_support, r.context_relevance, False, thresholds) for r in reports]
    flags = [is_flagged(v, "partially_supported") for v in verdicts]
    assert flags == unsupported


def test_calibrate_rejects_bad_input():
    reports = [_report(0.9), _report(0.4)]
    with pytest.raises(ValueError, match="both supported and unsupported"):
        calibrate(reports, [True, True])
    with pytest.raises(ValueError):
        calibrate(reports, [True])
    with pytest.raises(ValueError):
        calibrate(reports, [True, False], fail_on="likely_supported")


def test_candidate_grid_has_no_float_noise():
    base = Thresholds(support=0.57, review=0.29)
    assert _candidates("partially_supported", base)[0] == 0.29  # not 0.28 or 0.30
    assert _candidates("needs_review", base)[-1] == 0.57
