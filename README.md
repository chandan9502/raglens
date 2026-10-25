# RAGLens

[![CI](https://github.com/chandan9502/raglens/actions/workflows/ci.yml/badge.svg)](https://github.com/chandan9502/raglens/actions/workflows/ci.yml)

**A lightweight, explainable baseline for checking RAG answers against the context they were given.**

RAGLens compares the *words* in a question, a retrieved context, and a generated answer. It reports
how much of the question the context covers, how much of the answer the context supports, which
answer terms are missing from the context, and a few warnings (changed numbers, negation mismatch).

It is a transparent first check you can run in seconds with no API key. It is **not** a
replacement for human review or an LLM-based judge (see [Known limitations](#known-limitations)).

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
  "warnings": []
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

## Python API

```python
from raglens import evaluate

report = evaluate(
    question="What database is used?",
    context="The service stores records in a PostgreSQL database.",
    answer="It uses PostgreSQL.",
)
print(report.verdict)      # likely_supported
print(report.to_dict())
```

## How scoring works

- Text is lowercased and split into words; common stop words are dropped.
- A small suffix stripper lets `store`, `stores` and `stored` match.
- `context_relevance` = share of question terms found in the context.
- `answer_support` = share of answer terms found in the context.
- Verdict: `likely_supported` (support ≥ 0.8 and relevance ≥ 0.25), `needs_review`
  (support < 0.5 or relevance < 0.25), otherwise `partially_supported`.
- `warnings` flag numbers in the answer that are not in the context, and answer/context
  disagreeing on whether negation words (`not`, `no`, `never`...) appear. Any warning
  stops a result from being `likely_supported`.

## Known limitations

These are properties of a word-overlap baseline, not bugs:

- **Synonyms and paraphrases are missed.** If the context says "PostgreSQL" but never says
  "database", then asking *"What database is used?"* gives `context_relevance` 0.0 and
  `needs_review`, even though the context answers the question.
- **A matching answer can still be wrong.** Word overlap is not entailment. The negation and number
  warnings are rough flags, not understanding.
- Scores are not probabilities and are not calibrated against human judgments.
- English-oriented tokenization and stop words.

Always review important results yourself.

## Development

```bash
pip install -e ".[dev]"
ruff check .
python -m pytest
```

## Roadmap

Ideas, not promises. Issues and discussion welcome.

- Optional semantic-similarity backend
- Adapters for common RAG frameworks
- Optional model-based evaluation with documented judge prompts
- Calibration against a small human-labelled set

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Changes are listed in [CHANGELOG.md](CHANGELOG.md).

## License

MIT. See [LICENSE](LICENSE).
