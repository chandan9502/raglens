import json

import pytest

from raglens import VERDICTS, Thresholds, calibrate, evaluate
from raglens.calibration import is_flagged
from raglens.cli import main

CONTEXT = "The cafe is in Paris. It sells coffee and cake."


@pytest.mark.parametrize(
    "answer",
    [
        "Unable to answer based on given passages.",
        "I don't know.",
        "Sorry, I am unable to answer that.",
        "The passages do not contain enough information to answer this question.",
        "There is no information about the opening date.",
        "The opening date is not mentioned in the passages.",
        "The passages do not say when the cafe opened.",  # restating the question is allowed
    ],
)
def test_refusals_are_abstentions(answer):
    report = evaluate("When did the cafe open?", CONTEXT, answer)
    assert report.abstained is True
    assert report.verdict == "abstained"
    assert not any("Negation" in w for w in report.warnings)  # refusals are worded negatively
    assert "declines to answer" in report.explanations[1]


@pytest.mark.parametrize(
    "answer",
    [
        "The cafe opened in 1990.",  # an ordinary claim
        "No date is given, but it opened in 1990 in Lyon.",  # refusal plus new claims
        "Unable to answer based on given passages. It opened in Lyon.",  # a refusal sentence plus a claim
        "The opening date is not mentioned, but the owner is Marie.",  # a new name is a possible claim
        "The cafe was not included in the guide.",  # a negative claim about the world, not the text
    ],
)
def test_answers_with_claims_are_not_abstentions(answer):
    report = evaluate("When did the cafe open?", CONTEXT, answer)
    assert report.abstained is False
    assert report.verdict != "abstained"


def test_scores_are_the_same_with_or_without_detection():
    refusal = "Unable to answer based on given passages."
    on = evaluate("When did the cafe open?", CONTEXT, refusal)
    off = evaluate("When did the cafe open?", CONTEXT, refusal, detect_abstentions=False)
    assert on.answer_support == off.answer_support
    assert on.context_relevance == off.context_relevance
    assert off.abstained is False
    assert off.verdict == "needs_review"


def test_custom_abstention_phrase():
    answer = "Sorry, that is not in our help centre."
    assert evaluate("Do you ship abroad?", CONTEXT, answer).verdict != "abstained"
    report = evaluate("Do you ship abroad?", CONTEXT, answer, abstention_phrases=["not in our help centre"])
    assert report.verdict == "abstained"


def test_abstention_phrases_must_be_a_list_of_words():
    with pytest.raises(TypeError):
        evaluate("q?", CONTEXT, "a", abstention_phrases="not in our help centre")
    with pytest.raises(ValueError):
        evaluate("q?", CONTEXT, "a", abstention_phrases=["..."])


def test_abstained_is_never_flagged():
    assert VERDICTS[0] == "abstained"
    for level in ("partially_supported", "needs_review"):
        assert not is_flagged("abstained", level)


def test_calibration_keeps_abstentions_unflagged():
    refusal = evaluate("When did the cafe open?", CONTEXT, "Unable to answer based on given passages.")
    claim = evaluate("When did the cafe open?", CONTEXT, "The cafe opened in 1990 in Lyon.")
    result = calibrate([refusal, claim], [False, True], fail_on="needs_review")
    for point in result["curve"]:  # at every threshold the refusal stays unflagged
        assert point["fp"] == 0


def test_thresholds_do_not_change_an_abstention():
    strict = Thresholds(support=1.0, review=1.0, relevance=1.0)
    report = evaluate("When did the cafe open?", CONTEXT, "I don't know.", thresholds=strict)
    assert report.verdict == "abstained"


# --- CLI --------------------------------------------------------------------------

REFUSAL_ARGS = ["evaluate", "--question", "When did the cafe open?", "--context", CONTEXT,
                "--answer", "Unable to answer based on given passages."]


def test_cli_refusal_passes_fail_on(capsys):
    assert main([*REFUSAL_ARGS, "--fail-on", "partially_supported"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["verdict"] == "abstained"
    assert out["abstained"] is True


def test_cli_no_abstentions_flag(capsys):
    assert main([*REFUSAL_ARGS, "--no-abstentions", "--fail-on", "needs_review"]) == 1
    assert json.loads(capsys.readouterr().out)["verdict"] == "needs_review"


def test_cli_abstention_phrase_is_repeatable(capsys):
    main(["evaluate", "--question", "q?", "--context", CONTEXT, "--answer", "Ask our team instead.",
          "--abstention-phrase", "not in our docs", "--abstention-phrase", "ask our team instead"])
    assert json.loads(capsys.readouterr().out)["verdict"] == "abstained"


def test_cli_rejects_empty_abstention_phrase(capsys):
    with pytest.raises(SystemExit) as exc:
        main([*REFUSAL_ARGS, "--abstention-phrase", "..."])
    assert exc.value.code == 2
    assert "abstention-phrase" in capsys.readouterr().err


def test_batch_counts_abstentions(tmp_path, capsys):
    path = tmp_path / "rows.jsonl"
    rows = [
        {"question": "When did the cafe open?", "context": CONTEXT, "answer": "I don't know."},
        {"question": "Where is the cafe?", "context": CONTEXT, "answer": "The cafe is in Paris."},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    main(["batch", str(path)])
    counts = json.loads(capsys.readouterr().out)["summary"]["verdict_counts"]
    assert counts["abstained"] == 1
    assert counts["likely_supported"] == 1
