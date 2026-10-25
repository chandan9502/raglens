import pytest
from raglens import evaluate


def test_supported_answer_has_high_lexical_support():
    report = evaluate(
        "Which database stores records?",
        "The database stores customer records in PostgreSQL.",
        "Customer records are in PostgreSQL.",
    )
    assert report.answer_support >= 0.8
    assert report.verdict == "likely_supported"


def test_unmatched_answer_terms_are_flagged():
    report = evaluate(
        "Which database is used?",
        "The service stores records in PostgreSQL.",
        "It uses PostgreSQL with automatic quantum encryption.",
    )
    assert "quantum" in report.unsupported_terms
    assert report.verdict in {"needs_review", "partially_supported"}


def test_rejects_empty_inputs():
    with pytest.raises(ValueError):
        evaluate("", "some context", "some answer")


def test_report_is_json_friendly():
    report = evaluate("What stores records?", "PostgreSQL stores records.", "PostgreSQL stores records.")
    data = report.to_dict()
    assert isinstance(data["limitations"], list)
    assert 0.0 <= data["answer_support"] <= 1.0
