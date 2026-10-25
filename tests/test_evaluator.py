import json

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
        "The service stores records in a PostgreSQL database.",
        "It uses PostgreSQL with automatic quantum encryption.",
    )
    assert "quantum" in report.unsupported_terms
    assert report.verdict in {"needs_review", "partially_supported"}


@pytest.mark.parametrize("bad", ["", "   ", None, 42])
def test_rejects_bad_inputs(bad):
    with pytest.raises(ValueError):
        evaluate(bad, "some context", "some answer")
    with pytest.raises(ValueError):
        evaluate("a question", bad, "some answer")
    with pytest.raises(ValueError):
        evaluate("a question", "some context", bad)


def test_report_is_json_friendly():
    report = evaluate("What stores records?", "PostgreSQL stores records.", "PostgreSQL stores records.")
    data = report.to_dict()
    json.dumps(data)  # must not raise
    assert isinstance(data["limitations"], list)
    assert isinstance(data["warnings"], list)
    assert 0.0 <= data["answer_support"] <= 1.0


def test_readme_quickstart_example_is_likely_supported():
    # Regression: this exact example is shown in the README.
    report = evaluate(
        "What database is used?",
        "The service stores records in a PostgreSQL database.",
        "It uses PostgreSQL.",
    )
    assert report.verdict == "likely_supported"
    assert report.answer_support == 1.0
    assert report.unsupported_terms == []


def test_word_forms_match():
    report = evaluate("Which services store files?", "The service stored the file.", "Services store files.")
    assert report.answer_support == 1.0


def test_changed_number_is_warned_and_not_likely_supported():
    report = evaluate(
        "What is the request timeout?",
        "Requests time out after 30 seconds.",
        "Requests time out after 60 seconds.",
    )
    assert "60" in report.unsupported_terms
    assert any("60" in w for w in report.warnings)
    assert report.verdict != "likely_supported"


def test_negation_mismatch_is_warned():
    report = evaluate(
        "Does the server support SSL?",
        "The server does not support SSL.",
        "The server supports SSL.",
    )
    assert any("Negation" in w for w in report.warnings)
    assert report.verdict != "likely_supported"


def test_matching_negation_is_not_warned():
    report = evaluate(
        "Does the server support SSL?",
        "The server does not support SSL.",
        "The server does not support SSL.",
    )
    assert report.warnings == []


def test_off_topic_context_needs_review():
    report = evaluate(
        "What database stores the records?",
        "The frontend is built with React and served by nginx.",
        "Records are stored in MongoDB.",
    )
    assert report.verdict == "needs_review"
    assert report.context_relevance < 0.25


def test_question_of_only_stop_words_does_not_crash():
    report = evaluate("What is it?", "Some context here.", "Some answer here.")
    assert report.context_relevance == 0.0
    assert report.verdict == "needs_review"


def test_known_limitation_keyword_missing_from_context():
    # Documented in the README: the context answers the question, but never uses
    # the word "database", so the lexical relevance score is 0.
    report = evaluate(
        "What database is used?",
        "The service stores records in PostgreSQL.",
        "It uses PostgreSQL.",
    )
    assert report.context_relevance == 0.0
    assert report.verdict == "needs_review"
