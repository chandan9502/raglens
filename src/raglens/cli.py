"""Command-line interface for RAGLens."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import __version__
from .calibration import calibrate, is_flagged
from .evaluator import DEFAULT_THRESHOLDS, VERDICTS, EvaluationReport, Thresholds, evaluate
from .metrics import binary_metrics, roc_auc

_FAIL_ON_CHOICES = ["partially_supported", "needs_review"]
_LABELS = ("supported", "unsupported")


@dataclass(frozen=True)
class _Example:
    """One parsed JSONL line."""

    line: int
    question: str
    context: str
    answer: str
    label: str | None  # "supported", "unsupported" or None when unlabelled


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


def _unit_interval(text: str) -> float:
    """argparse type: a number from 0 to 1."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not 0.0 <= value <= 1.0:  # also rejects nan
        raise argparse.ArgumentTypeError(f"must be from 0 to 1, got {text}")
    return value


def _add_verdict_args(parser: argparse.ArgumentParser) -> None:
    """Flags that change how scores become a verdict (shared by every command)."""
    group = parser.add_argument_group("verdict thresholds (defaults are hand-picked, not tuned)")
    d = DEFAULT_THRESHOLDS
    group.add_argument(
        "--support-threshold", type=_unit_interval, default=d.support, metavar="X",
        help=f"answer_support needed for likely_supported (default {d.support})",
    )
    group.add_argument(
        "--review-threshold", type=_unit_interval, default=d.review, metavar="X",
        help=f"answer_support below this gives needs_review (default {d.review})",
    )
    group.add_argument(
        "--relevance-threshold", type=_unit_interval, default=d.relevance, metavar="X",
        help=f"context_relevance below this gives needs_review (default {d.relevance})",
    )
    refusals = parser.add_argument_group("refusals (answers that decline to answer)")
    refusals.add_argument(
        "--abstention-phrase", action="append", default=[], metavar="TEXT",
        help='Extra wording your system uses to decline, e.g. "not in our help centre" (repeatable)',
    )
    refusals.add_argument(
        "--no-abstentions", action="store_true",
        help="Do not recognise refusals; score them like any other answer",
    )


def _evaluate_options(parser: argparse.ArgumentParser, args: argparse.Namespace) -> dict[str, Any]:
    """Keyword arguments for evaluate(), built from the shared flags."""
    for phrase in args.abstention_phrase:
        if not any(ch.isalnum() for ch in phrase):
            parser.error(f"--abstention-phrase needs at least one word, got {phrase!r}")
    try:
        thresholds = Thresholds(
            support=args.support_threshold, review=args.review_threshold, relevance=args.relevance_threshold
        )
    except ValueError as exc:
        parser.error(str(exc))
    return {
        "thresholds": thresholds,
        "abstention_phrases": tuple(args.abstention_phrase),
        "detect_abstentions": not args.no_abstentions,
    }


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
    ev.add_argument("--fail-on", choices=_FAIL_ON_CHOICES, help=fail_help)
    _add_verdict_args(ev)

    ba = sub.add_parser(
        "batch",
        help="Evaluate a JSONL file (one {question, context, answer} object per line, optional label)",
    )
    ba.add_argument("path", help="Path to a .jsonl file")
    ba.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    ba.add_argument("--fail-on", choices=_FAIL_ON_CHOICES, help=fail_help)
    _add_verdict_args(ba)

    ca = sub.add_parser(
        "calibrate",
        help="Suggest a threshold from labelled JSONL (each line needs label: supported/unsupported)",
    )
    ca.add_argument("path", help="Path to a labelled .jsonl file")
    ca.add_argument(
        "--fail-on", choices=_FAIL_ON_CHOICES, default="partially_supported",
        help="The --fail-on level you plan to use; calibration tunes the threshold behind it "
        "(default partially_supported)",
    )
    ca.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    ca.add_argument("--no-curve", action="store_true", help="Leave out the per-threshold curve")
    _add_verdict_args(ca)
    return parser


def _dump(data: object, pretty: bool) -> str:
    return json.dumps(data, indent=2 if pretty else None, ensure_ascii=False)


def _load_examples(parser: argparse.ArgumentParser, path: str) -> list[_Example]:
    """Read a JSONL file, reporting the line number of any problem."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        parser.error(f"cannot read {path!r}: {exc}")

    examples = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            example = _Example(number, row["question"], row["context"], row["answer"], row.get("label"))
        except json.JSONDecodeError as exc:
            parser.error(f"line {number}: invalid JSON ({exc.msg})")
        except (KeyError, TypeError, AttributeError):
            parser.error(f"line {number}: each line needs 'question', 'context' and 'answer'")
        if example.label is not None and example.label not in _LABELS:
            parser.error(
                f"line {number}: 'label' must be 'supported' or 'unsupported', got {example.label!r}"
            )
        examples.append(example)

    if not examples:
        parser.error(f"{path!r} contains no examples")
    return examples


def _evaluate_all(
    parser: argparse.ArgumentParser, examples: Sequence[_Example], options: dict[str, Any]
) -> list[EvaluationReport]:
    reports = []
    for ex in examples:
        try:
            reports.append(evaluate(ex.question, ex.context, ex.answer, **options))
        except ValueError as exc:
            parser.error(f"line {ex.line}: {exc}")
    return reports


def _label_metrics(examples: Sequence[_Example], reports: Sequence[EvaluationReport]) -> dict[str, Any]:
    """How well verdicts agree with labels, for each --fail-on level."""
    labelled: list[EvaluationReport] = []
    unsupported: list[bool] = []
    for ex, report in zip(examples, reports, strict=True):
        if ex.label is not None:
            labelled.append(report)
            unsupported.append(ex.label == "unsupported")
    metrics: dict[str, Any] = {
        "labelled_examples": len(labelled),
        "unsupported_examples": sum(unsupported),
    }
    for level in _FAIL_ON_CHOICES:
        flags = [is_flagged(r.verdict, level) for r in labelled]
        metrics[f"fail_on_{level}"] = binary_metrics(flags, unsupported)
    metrics["answer_support_auc"] = roc_auc([r.answer_support for r in labelled], unsupported)
    return metrics


def _run_evaluate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    options = _evaluate_options(parser, args)
    context = _text_arg(parser, "context", args.context, args.context_file)
    answer = _text_arg(parser, "answer", args.answer, args.answer_file)
    try:
        report = evaluate(args.question, context, answer, **options)
    except ValueError as exc:
        parser.error(str(exc))
    print(_dump(report.to_dict(), args.pretty))
    if args.fail_on and is_flagged(report.verdict, args.fail_on):
        return 1
    return 0


def _run_batch(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    options = _evaluate_options(parser, args)
    examples = _load_examples(parser, args.path)
    results = _evaluate_all(parser, examples, options)

    counts = Counter(r.verdict for r in results)
    summary: dict[str, Any] = {
        "examples": len(results),
        "verdict_counts": {v: counts.get(v, 0) for v in VERDICTS},
        "mean_context_relevance": round(sum(r.context_relevance for r in results) / len(results), 4),
        "mean_answer_support": round(sum(r.answer_support for r in results) / len(results), 4),
    }
    if any(ex.label is not None for ex in examples):
        summary["label_metrics"] = _label_metrics(examples, results)

    print(_dump({"summary": summary, "results": [r.to_dict() for r in results]}, args.pretty))
    if args.fail_on and any(is_flagged(r.verdict, args.fail_on) for r in results):
        return 1
    return 0


def _run_calibrate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    options = _evaluate_options(parser, args)
    examples = [ex for ex in _load_examples(parser, args.path) if ex.label is not None]
    if not examples:
        parser.error(f"{args.path!r} has no labelled lines; add \"label\": \"supported\" or \"unsupported\"")
    reports = _evaluate_all(parser, examples, options)
    try:
        result = calibrate(
            reports,
            [ex.label == "unsupported" for ex in examples],
            fail_on=args.fail_on,
            base=options["thresholds"],
        )
    except ValueError as exc:
        parser.error(str(exc))
    result["note"] = (
        "This threshold was chosen on this file only. Check it on separate labelled data "
        "before relying on it."
    )
    if args.no_curve:
        del result["curve"]
    print(_dump(result, args.pretty))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        return _run_evaluate(parser, args)
    if args.command == "batch":
        return _run_batch(parser, args)
    return _run_calibrate(parser, args)


if __name__ == "__main__":
    sys.exit(main())
