"""Measure RAGLens against human/benchmark labels and write benchmarks/RESULTS.md.

Usage (from the repository root, after `pip install -e .`):

    python benchmarks/prepare_data.py
    python benchmarks/run_benchmarks.py

Method, in one paragraph: every answer is scored once with RAGLens. Thresholds
are chosen with `calibrate` on the RAGTruth *train* split only, then applied
unchanged to the RAGTruth *test* split and to HaluEval. Nothing is tuned on the
data it is reported on, so the test numbers are an honest estimate.
"""
from __future__ import annotations

import datetime as dt
import json
import platform
import sys
from collections.abc import Sequence
from pathlib import Path

import raglens
from raglens import EvaluationReport, Thresholds, calibrate, evaluate
from raglens.calibration import is_flagged
from raglens.evaluator import decide_verdict
from raglens.metrics import binary_metrics, roc_auc

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
LEVELS = ("partially_supported", "needs_review")

Dataset = tuple[list[EvaluationReport], list[bool]]


def load(name: str, detect_abstentions: bool = True) -> Dataset:
    """Score every row of a prepared file once. Returns reports and labels."""
    path = DATA / name
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python benchmarks/prepare_data.py")
    reports, unsupported = [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            question, context, answer = item["question"], item["context"], item["answer"]
            reports.append(evaluate(question, context, answer, detect_abstentions=detect_abstentions))
            unsupported.append(item["label"] == "unsupported")
    return reports, unsupported


def metrics_at(reports: Sequence[EvaluationReport], unsupported: Sequence[bool],
               thresholds: Thresholds, level: str) -> dict:
    """Metrics if you ran `--fail-on level` with these thresholds."""
    verdicts = [
        decide_verdict(r.answer_support, r.context_relevance, bool(r.warnings), thresholds, r.abstained)
        for r in reports
    ]
    return binary_metrics([is_flagged(v, level) for v in verdicts], unsupported)


def tune(train: Dataset) -> dict[str, tuple[str, float, Thresholds]]:
    """Pick one threshold per --fail-on level on the train split only."""
    result = {}
    for level in LEVELS:
        best = calibrate(*train, fail_on=level)
        name, value = best["tuned_threshold"], best["best"]["threshold"]
        result[level] = (name, value, Thresholds(**{name: value}))
    return result


def flag_everything(unsupported: Sequence[bool]) -> dict:
    return binary_metrics([True] * len(unsupported), unsupported)


def row(name: str, m: dict) -> str:
    return (f"| {name} | {m['tp'] + m['fp']:,} | {m['precision']:.3f} | {m['recall']:.3f} "
            f"| **{m['f1']:.3f}** | {m['accuracy']:.3f} |")


HEADER = "| Setting | Flagged | Precision | Recall | F1 | Accuracy |\n|---|---:|---:|---:|---:|---:|"

NOTES = [
    "- Only the QA task of RAGTruth is used (not summarisation or data-to-text). A response is "
    "*unsupported* if annotators marked at least one hallucinated span in it. That includes spans "
    "annotators tagged `implicit_true` (true, but not stated in the passages), because RAGLens "
    "checks support by the passages.",
    "- RAGTruth's prompt told models to reply \"Unable to answer based on given passages.\" when the "
    "passages lacked the answer. RAGLens recognises refusals with its built-in, general patterns; no "
    "dataset-specific phrase is passed.",
    "- HaluEval QA labels come from how the data was built: the HotpotQA answer is *supported*, a "
    "ChatGPT-written plausible wrong answer is *unsupported*. Right answers are short (median 2 "
    "words) while wrong ones are sentences (median 9 words), a difference that suits a "
    "word-overlap check. A few of the \"wrong\" answers are refusals (\"The television series is not "
    "specified.\"); RAGLens marks those `abstained`, which counts as a miss here.",
    "- Thresholds were chosen with `raglens calibrate` (best F1 on a 0.01 grid) on RAGTruth train. "
    "The test split and HaluEval were never used for tuning.",
    "- The model-based detectors evaluated in the [RAGTruth paper](https://arxiv.org/abs/2401.00396) "
    "report much higher response-level F1 (across all three tasks, so not directly comparable). "
    "RAGLens is a no-model baseline; these numbers show what word overlap alone can do.",
]


def results_table(title: str, data: Dataset, tuned: dict) -> list[str]:
    reports, unsupported = data
    n, k = len(unsupported), sum(unsupported)
    lines = [f"## {title}", "", f"{n:,} answers, {k:,} unsupported ({k / n:.1%}).", "", HEADER,
             row("Flag everything (reference)", flag_everything(unsupported))]
    for level in LEVELS:
        lines.append(row(f"Default thresholds, `--fail-on {level}`",
                         metrics_at(reports, unsupported, Thresholds(), level)))
    for level in LEVELS:
        name, value, thresholds = tuned[level]
        lines.append(row(f"Tuned on RAGTruth train ({name} = {value}), `--fail-on {level}`",
                         metrics_at(reports, unsupported, thresholds, level)))
    auc = roc_auc([r.answer_support for r in reports], unsupported)
    abstained = [u for r, u in zip(reports, unsupported, strict=True) if r.abstained]
    lines += ["", f"`answer_support` ROC AUC: **{auc:.3f}** (0.5 = coin flip, 1.0 = perfect ranking). "
              f"Answers marked `abstained` (refusals): {len(abstained):,}, of which {sum(abstained):,} "
              "labelled unsupported.", ""]
    return lines


def refusal_ablation(test_on: Dataset, test_off: Dataset, tuned_on: dict, tuned_off: dict) -> list[str]:
    """Same test split with and without refusal detection, each tuned on train its own way."""
    lines = ["## Effect of recognising refusals (RAGTruth QA test)", "",
             "Each configuration has its own thresholds, tuned on the train split.", "", HEADER]
    for label, (reports, unsupported), tuned in (("off", test_off, tuned_off), ("on", test_on, tuned_on)):
        for level in LEVELS:
            name, value, thresholds = tuned[level]
            lines.append(row(f"Refusal detection {label}, tuned {name} = {value}, `--fail-on {level}`",
                             metrics_at(reports, unsupported, thresholds, level)))
    return lines + [""]


def warning_table(datasets: dict[str, Dataset]) -> list[str]:
    lines = ["| Warning | Dataset | Fired on | Share of those really unsupported | Base rate |",
             "|---|---|---:|---:|---:|"]
    for prefix, label in (("Numbers", "Number not in context"), ("Negation", "Negation mismatch")):
        for name, (reports, unsupported) in datasets.items():
            fired = [u for r, u in zip(reports, unsupported, strict=True)
                     if any(w.startswith(prefix) for w in r.warnings)]
            share = sum(fired) / len(fired) if fired else 0.0
            base = sum(unsupported) / len(unsupported)
            lines.append(f"| {label} | {name} | {len(fired):,} ({len(fired) / len(reports):.0%}) "
                         f"| {share:.3f} | {base:.3f} |")
    return lines


def main() -> None:
    train, test, halu = (load(f) for f in ("ragtruth_qa_train.jsonl", "ragtruth_qa_test.jsonl",
                                           "halueval_qa.jsonl"))
    tuned = tune(train)

    # The same split with refusal detection switched off, for the comparison table.
    train_off = load("ragtruth_qa_train.jsonl", detect_abstentions=False)
    test_off = load("ragtruth_qa_test.jsonl", detect_abstentions=False)
    tuned_off = tune(train_off)

    out = [
        "# RAGLens benchmark results",
        "",
        f"Generated by `benchmarks/run_benchmarks.py` on {dt.date.today().isoformat()} with "
        f"raglens {raglens.__version__}, Python {platform.python_version()}. "
        "Data: see `benchmarks/prepare_data.py` (pinned commits, SHA-256 checked).",
        "",
        "Positive class = **unsupported** answer. *Flagged* = RAGLens would fail the answer with "
        "the given `--fail-on` level. Thresholds were tuned on RAGTruth **train** only. "
        "Refusal detection is on (the default) unless a row says otherwise.",
        "",
    ]
    out += results_table("RAGTruth QA, held-out test split (human labels)", test, tuned)
    out += results_table("HaluEval QA (constructed right/wrong answer pairs)", halu, tuned)
    out += refusal_ablation(test, test_off, tuned, tuned_off)
    out += ["## What the warnings are worth", "",
            "How often each warning fired, and how often it was right. A useful warning is right "
            "more often than the base rate. RAGTruth numbers use the train split.", ""]
    out += warning_table({"RAGTruth train": train, "HaluEval": halu})
    out += ["", "## Notes on the data", "", *NOTES, ""]

    text = "\n".join(out)
    (HERE / "RESULTS.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nwrote {HERE / 'RESULTS.md'}", file=sys.stderr)


if __name__ == "__main__":
    main()
