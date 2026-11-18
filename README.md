# RAGLens

[![CI](https://github.com/chandan9502/raglens/actions/workflows/ci.yml/badge.svg)](https://github.com/chandan9502/raglens/actions/workflows/ci.yml)

**A lightweight, explainable baseline for checking RAG answers against the context they were given.**

RAGLens compares the *words* in a question, a retrieved context, and a generated answer. It reports
how much of the question the context covers, how much of the answer the context supports, which
answer terms are missing from the context, and a few warnings (changed numbers, negation mismatch).

It is a transparent first check you can run in seconds with no API key. It is **not** a
replacement for human review or an LLM-based judge (see [Known limitations](#known-limitations)).

## How good is it?

Measured on 900 held-out answers from [RAGTruth](https://github.com/ParticleMedia/RAGTruth), real LLM
answers that human annotators checked for hallucinations
(full tables and reproduction steps: [benchmarks/RESULTS.md](benchmarks/RESULTS.md)):

| Setting (RAGTruth QA test split) | Precision | Recall | F1 |
|---|---:|---:|---:|
| Flag every answer (reference) | 0.178 | 1.000 | 0.302 |
| Default thresholds, `--fail-on needs_review` | 0.496 | 0.350 | 0.410 |
| Threshold tuned on the train split, `--fail-on needs_review` | 0.309 | 0.831 | 0.450 |

- `answer_support` ranks unsupported answers below supported ones better than chance
  (ROC AUC 0.756, where 0.5 is a coin flip).
- With a tuned threshold it catches about 8 in 10 unsupported answers, but about 7 in 10 of its
  flags are false alarms. It is a cheap first filter, not a judge.
- Recognising refusals ("Unable to answer based on given passages.") helps: with refusal detection
  switched off, the same tuned setup scores F1 0.428. None of the 27 test answers recognised as
  refusals had been labelled unsupported.
- On HaluEval's constructed wrong answers it does much better (F1 0.798 with default thresholds and
  `--fail-on partially_supported`), and neither threshold tuned on RAGTruth beats that there
  (0.743 and 0.663). The best setting for one dataset is not the best for another: calibrate on
  labelled examples from your own system (see [Calibrate thresholds](#calibrate-thresholds)).

## Install

Requires Python 3.10+. RAGLens has no runtime dependencies.

```bash
git clone https://github.com/chandan9502/raglens.git
cd raglens
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Quick start

```bash
raglens evaluate \
  --question "What database is used?" \
  --context "The service stores records in a PostgreSQL database." \
  --answer "It uses PostgreSQL." \
  --pretty
```

```json
{
  "question": "What database is used?",
  "context_relevance": 1.0,
  "answer_support": 1.0,
  "unsupported_terms": [],
  "verdict": "likely_supported",
  "explanations": [
    "Some meaningful question terms appear in the context.",
    "All meaningful answer terms appear in the context under this lexical check."
  ],
  "limitations": ["..."],
  "warnings": [],
  "abstained": false
}
```

(`limitations` is shortened here; the real output lists three entries.)

`python -m raglens ...` works too.

### Context or answer from a file

```bash
raglens evaluate --question "..." --context-file retrieved.txt --answer-file answer.txt
```

### Evaluate a dataset (JSONL)

Each line is a JSON object with `question`, `context` and `answer`:

```bash
raglens batch examples/sample.jsonl --pretty
```

The output has a `summary` (verdict counts and mean scores) plus one report per line.

### Use it in CI

`--fail-on` makes the command exit with code 1 when a verdict is that bad or worse:

```bash
raglens batch my_regression_set.jsonl --fail-on needs_review
```

### Measure it against your own labels

Add `"label": "supported"` or `"label": "unsupported"` to JSONL lines and `batch` adds
`label_metrics` to the summary: a confusion matrix (`tp`, `fp`, `fn`, `tn`), precision, recall,
F1 and accuracy for each `--fail-on` level, plus the ROC AUC of `answer_support`.
"Unsupported" is the positive class: recall is the share of unsupported answers that get flagged.

```bash
raglens batch examples/labelled_sample.jsonl --pretty
```

### Calibrate thresholds

The default thresholds are hand-picked starting points. `calibrate` tries every value on a 0.01
grid and reports the one with the best F1 for the `--fail-on` level you plan to use. With
`--fail-on partially_supported` it tunes the support threshold; with `--fail-on needs_review`
it tunes the review threshold.

```bash
raglens calibrate labelled_train.jsonl --fail-on needs_review --no-curve --pretty
```

Abridged output on the RAGTruth QA train split:

```json
{
  "fail_on": "needs_review",
  "tuned_threshold": "review",
  "examples": 5034,
  "current": {"threshold": 0.5, "precision": 0.6406, "recall": 0.3088, "f1": 0.4167},
  "best": {"threshold": 0.71, "precision": 0.5091, "recall": 0.7896, "f1": 0.619},
  "answer_support_auc": 0.734
}
```

Then check the threshold on **different** labelled data before you rely on it:

```bash
raglens batch labelled_test.jsonl --review-threshold 0.71 --fail-on needs_review
```

All three thresholds can be set on `evaluate`, `batch` and `calibrate`: `--support-threshold`,
`--review-threshold` and `--relevance-threshold`.

### Refusals

An answer that only declines to answer, such as "Unable to answer based on the given passages." or
"I don't know.", makes no claims, so RAGLens gives it the verdict `abstained` (and
`"abstained": true`) instead of flagging it. `--fail-on` never fails an abstention, and `batch`
counts them, so you can track your system's refusal rate.

A refusal may restate the question ("The passages do not say when the cafe opened."), but if it
adds anything else, such as "...but it opened in Lyon", the answer is scored as usual. Whether
refusing was the *right* call is not checked.

If your system declines with its own wording, add it (repeatable):

```bash
raglens batch answers.jsonl --abstention-phrase "not in our help centre"
```

`--no-abstentions` turns the check off. In Python, use `abstention_phrases=[...]` and
`detect_abstentions=False`.

## Python API

```python
from raglens import Thresholds, evaluate

question = "Is the API rate limited?"
context = "The public API allows 100 requests per minute for each key."
answer = "Yes, each key can make 100 requests per minute."

report = evaluate(question, context, answer)
print(report.answer_support, report.verdict)   # 0.75 partially_supported
print(report.to_dict())

# Thresholds change the verdict only, never the scores.
relaxed = evaluate(question, context, answer, thresholds=Thresholds(support=0.7))
print(relaxed.answer_support, relaxed.verdict)  # 0.75 likely_supported
```

`raglens.calibrate(reports, unsupported, fail_on=...)` runs the same calibration as the CLI, and
`raglens.metrics` has `binary_metrics` and `roc_auc` with no extra dependencies.

## How scoring works

- Text is lowercased and split into words; common stop words are dropped.
- A small suffix stripper lets `store`, `stores` and `stored` match.
- `context_relevance` = share of question terms found in the context.
- `answer_support` = share of answer terms found in the context.
- Verdict, with the default thresholds: `likely_supported` (support ≥ 0.8 and relevance ≥ 0.25),
  `needs_review` (support < 0.5 or relevance < 0.25), otherwise `partially_supported`.
- `warnings` flag numbers in the answer that are not in the context, and answer/context
  disagreeing on whether negation words (`not`, `no`, `never`...) appear. Any warning
  stops a result from being `likely_supported`.
- An answer whose every sentence is a refusal, and which adds no new terms, gets `abstained`
  before any threshold is applied (see [Refusals](#refusals)). Its scores are computed as usual.

## Known limitations

These are properties of a word-overlap baseline, not bugs:

- **Synonyms and paraphrases are missed.** If the context says "PostgreSQL" but never says
  "database", then asking *"What database is used?"* gives `context_relevance` 0.0 and
  `needs_review`, even though the context answers the question.
- **A matching answer can still be wrong.** Word overlap is not entailment. The negation and number
  warnings are rough flags, not understanding. On RAGTruth the number warning was right 53% of the
  time (base rate 31%), but the negation warning only 35%, so treat it as a prompt to look, not as
  evidence.
- **Refusals are recognised by wording only.** The built-in patterns are English. A refusal phrased
  some other way is scored like a normal answer (add your wording with `--abstention-phrase`).
  RAGLens also cannot tell a justified refusal from one where the context did contain the answer.
- Scores are not probabilities. Calibration picks better cut-offs for your data; it does not turn
  word overlap into a confidence.
- English-oriented tokenization and stop words.

Always review important results yourself.

## Development

```bash
pip install -e ".[dev]"
ruff check .
python -m pytest
```

Reproduce the benchmark (downloads about 40 MB of MIT-licensed data, checked by SHA-256):

```bash
python benchmarks/prepare_data.py
python benchmarks/run_benchmarks.py   # rewrites benchmarks/RESULTS.md
```

## Roadmap

Ideas, not promises. Issues and discussion welcome.

- Spot refusals that were not justified (the context did contain the answer)
- Optional semantic-similarity backend, measured on the same benchmark
- Adapters for common RAG frameworks
- Optional model-based evaluation with documented judge prompts

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Changes are listed in [CHANGELOG.md](CHANGELOG.md).

## License

MIT. See [LICENSE](LICENSE).
