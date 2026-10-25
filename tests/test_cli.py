import json
import subprocess
import sys

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
