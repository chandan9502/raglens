"""Command-line interface for RAGLens."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .evaluator import VERDICTS, evaluate

# Higher number = worse. Used by --fail-on.
_SEVERITY = {"likely_supported": 0, "partially_supported": 1, "needs_review": 2}


def _text_arg(parser: argparse.ArgumentParser, name: str, value: str | None, path: str | None) -> str:
    """Return text given either inline (--name) or from a file (--name-file)."""
    if value is not None and path is not None:
        parser.error(f"use either --{name} or --{name}-file, not both")
    if path is not None:
        try:
            return Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            parser.error(f"cannot read --{name}-file {path!r}: {exc}")
    if value is None:
        parser.error(f"one of --{name} or --{name}-file is required")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="raglens", description="Run explainable lexical checks on RAG answers."
    )
    parser.add_argument("--version", action="version", version=f"raglens {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    fail_help = "Exit with code 1 if a verdict is this bad or worse (for CI)."

    ev = sub.add_parser("evaluate", help="Evaluate one question/context/answer triple")
    ev.add_argument("--question", required=True, help="User question")
    for name, what in (("context", "Retrieved context"), ("answer", "Generated answer")):
        ev.add_argument(f"--{name}", help=f"{what} (text)")
        ev.add_argument(f"--{name}-file", help=f"{what} (path to a UTF-8 text file)")
    ev.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    ev.add_argument("--fail-on", choices=["partially_supported", "needs_review"], help=fail_help)

    ba = sub.add_parser(
        "batch", help="Evaluate a JSONL file (one {question, context, answer} object per line)"
    )
    ba.add_argument("path", help="Path to a .jsonl file")
    ba.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    ba.add_argument("--fail-on", choices=["partially_supported", "needs_review"], help=fail_help)
    return parser


def _dump(data: object, pretty: bool) -> str:
    return json.dumps(data, indent=2 if pretty else None, ensure_ascii=False)


def _run_evaluate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    context = _text_arg(parser, "context", args.context, args.context_file)
    answer = _text_arg(parser, "answer", args.answer, args.answer_file)
    try:
        report = evaluate(args.question, context, answer)
    except ValueError as exc:
        parser.error(str(exc))
    print(_dump(report.to_dict(), args.pretty))
    if args.fail_on and _SEVERITY[report.verdict] >= _SEVERITY[args.fail_on]:
        return 1
    return 0


def _run_batch(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    try:
        lines = Path(args.path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        parser.error(f"cannot read {args.path!r}: {exc}")

    results = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            report = evaluate(row["question"], row["context"], row["answer"])
        except json.JSONDecodeError as exc:
            parser.error(f"line {number}: invalid JSON ({exc.msg})")
        except (KeyError, TypeError):
            parser.error(f"line {number}: each line needs 'question', 'context' and 'answer'")
        except ValueError as exc:
            parser.error(f"line {number}: {exc}")
        results.append(report)

    if not results:
        parser.error(f"{args.path!r} contains no examples")

    counts = Counter(r.verdict for r in results)
    summary = {
        "examples": len(results),
        "verdict_counts": {v: counts.get(v, 0) for v in VERDICTS},
        "mean_context_relevance": round(sum(r.context_relevance for r in results) / len(results), 4),
        "mean_answer_support": round(sum(r.answer_support for r in results) / len(results), 4),
    }
    print(_dump({"summary": summary, "results": [r.to_dict() for r in results]}, args.pretty))
    if args.fail_on:
        limit = _SEVERITY[args.fail_on]
        if any(_SEVERITY[r.verdict] >= limit for r in results):
            return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        return _run_evaluate(parser, args)
    return _run_batch(parser, args)


if __name__ == "__main__":
    sys.exit(main())
