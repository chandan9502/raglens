import json
import subprocess
import sys
from pathlib import Path

import pytest

from raglens.cli import main

CONTEXT = "The service stores records in a PostgreSQL database."
ARGS = ["evaluate", "--question", "What database is used?",
        "--context", CONTEXT, "--answer", "It uses PostgreSQL."]


def test_evaluate_prints_json(capsys):
    assert main(ARGS) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["verdict"] == "likely_supported"


def test_pretty_output_is_multiline(capsys):
    main([*ARGS, "--pretty"])
    assert "\n" in capsys.readouterr().out.strip()


def test_context_and_answer_from_files(tmp_path, capsys):
    ctx = tmp_path / "ctx.txt"
    ans = tmp_path / "ans.txt"
    ctx.write_text(CONTEXT, encoding="utf-8")
    ans.write_text("It uses PostgreSQL.", encoding="utf-8")
    code = main(["evaluate", "--question", "What database is used?",
                 "--context-file", str(ctx), "--answer-file", str(ans)])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "likely_supported"


def test_missing_context_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["evaluate", "--question", "q?", "--answer", "a"])
    assert exc.value.code == 2
    assert "--context" in capsys.readouterr().err


def test_both_inline_and_file_is_rejected(tmp_path):
    f = tmp_path / "c.txt"
    f.write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(["evaluate", "--question", "q?", "--context", "c", "--context-file", str(f), "--answer", "a"])
    assert exc.value.code == 2


def test_missing_file_is_a_usage_error():
    with pytest.raises(SystemExit) as exc:
        main(["evaluate", "--question", "q?", "--context-file", "nope.txt", "--answer", "a"])
    assert exc.value.code == 2


def test_empty_input_is_a_usage_error():
    with pytest.raises(SystemExit) as exc:
        main(["evaluate", "--question", "", "--context", "c", "--answer", "a"])
    assert exc.value.code == 2


def test_fail_on_exit_codes():
    bad = ["evaluate", "--question", "What database stores the records?",
           "--context", "The frontend is built with React.", "--answer", "Records are in MongoDB."]
    assert main(bad) == 0
    assert main([*bad, "--fail-on", "needs_review"]) == 1
    assert main([*ARGS, "--fail-on", "needs_review"]) == 0


def test_batch_summary(tmp_path, capsys):
    path = tmp_path / "d.jsonl"
    rows = [
        {"question": "What database is used?", "context": CONTEXT, "answer": "It uses PostgreSQL."},
        {"question": "What database stores the records?", "context": "The frontend is React.",
         "answer": "Records are in MongoDB."},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")
    assert main(["batch", str(path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["summary"]["examples"] == 2
    assert out["summary"]["verdict_counts"]["likely_supported"] == 1
    assert out["summary"]["verdict_counts"]["needs_review"] == 1
    assert main(["batch", str(path), "--fail-on", "needs_review"]) == 1


def test_batch_bad_json_reports_line_number(tmp_path, capsys):
    path = tmp_path / "bad.jsonl"
    path.write_text("not json\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(["batch", str(path)])
    assert exc.value.code == 2
    assert "line 1" in capsys.readouterr().err


def test_batch_missing_keys(tmp_path, capsys):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"question": "q?"}\n', encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["batch", str(path)])
    assert "question" in capsys.readouterr().err


def test_batch_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["batch", str(path)])


def test_python_dash_m_works():
    result = subprocess.run([sys.executable, "-m", "raglens", "--version"], capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout.startswith("raglens ")


# --- thresholds, labels and calibration (0.3.0) ---------------------------------

LABELLED = Path(__file__).resolve().parent.parent / "examples" / "labelled_sample.jsonl"
PARAPHRASE = ["evaluate", "--question", "Is the API rate limited?",
              "--context", "The public API allows 100 requests per minute for each key.",
              "--answer", "Yes, each key can make 100 requests per minute."]


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return str(path)


def test_support_threshold_flag_changes_verdict(capsys):
    main(PARAPHRASE)
    assert json.loads(capsys.readouterr().out)["verdict"] == "partially_supported"
    main([*PARAPHRASE, "--support-threshold", "0.7"])
    assert json.loads(capsys.readouterr().out)["verdict"] == "likely_supported"


@pytest.mark.parametrize("bad", [["--support-threshold", "1.5"], ["--support-threshold", "abc"],
                                 ["--review-threshold", "0.9"]])
def test_invalid_thresholds_are_usage_errors(bad, capsys):
    with pytest.raises(SystemExit) as exc:
        main([*ARGS, *bad])
    assert exc.value.code == 2
    assert "threshold" in capsys.readouterr().err


def test_batch_reports_label_metrics(capsys):
    assert main(["batch", str(LABELLED)]) == 0
    metrics = json.loads(capsys.readouterr().out)["summary"]["label_metrics"]
    assert metrics["labelled_examples"] == 11
    assert metrics["unsupported_examples"] == 5
    loose = metrics["fail_on_partially_supported"]
    strict = metrics["fail_on_needs_review"]
    assert loose["tp"] + loose["fn"] == strict["tp"] + strict["fn"] == 5
    # needs_review is the stricter gate, so it can never flag more answers.
    assert strict["tp"] + strict["fp"] <= loose["tp"] + loose["fp"]
    assert 0.0 <= metrics["answer_support_auc"] <= 1.0


def test_batch_without_labels_has_no_label_metrics(capsys):
    main(["batch", str(LABELLED.parent / "sample.jsonl")])
    assert "label_metrics" not in json.loads(capsys.readouterr().out)["summary"]


def test_invalid_label_reports_line_number(tmp_path, capsys):
    path = _write_jsonl(tmp_path / "bad.jsonl", [
        {"question": "q?", "context": "c", "answer": "a", "label": "supported"},
        {"question": "q?", "context": "c", "answer": "a", "label": "maybe"},
    ])
    with pytest.raises(SystemExit) as exc:
        main(["batch", path])
    assert exc.value.code == 2
    assert "line 2" in capsys.readouterr().err


def test_calibrate_outputs_best_and_current(capsys):
    assert main(["calibrate", str(LABELLED)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["fail_on"] == "partially_supported"
    assert out["tuned_threshold"] == "support"
    assert out["current"]["threshold"] == 0.8
    assert out["best"]["f1"] >= out["current"]["f1"]
    assert out["curve"][0]["threshold"] == 0.5 and out["curve"][-1]["threshold"] == 1.0
    assert "separate labelled data" in out["note"]


def test_calibrate_no_curve_and_needs_review(capsys):
    main(["calibrate", str(LABELLED), "--fail-on", "needs_review", "--no-curve"])
    out = json.loads(capsys.readouterr().out)
    assert out["tuned_threshold"] == "review"
    assert "curve" not in out


def test_calibrate_requires_labels_of_both_kinds(tmp_path, capsys):
    unlabelled = _write_jsonl(tmp_path / "u.jsonl", [{"question": "q?", "context": "c", "answer": "a"}])
    with pytest.raises(SystemExit):
        main(["calibrate", unlabelled])
    assert "no labelled lines" in capsys.readouterr().err

    one_class = _write_jsonl(tmp_path / "one.jsonl", [
        {"question": "q?", "context": "c", "answer": "a", "label": "supported"},
    ])
    with pytest.raises(SystemExit):
        main(["calibrate", one_class])
    assert "both supported and unsupported" in capsys.readouterr().err
