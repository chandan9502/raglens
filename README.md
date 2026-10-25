# RAGLens

**A lightweight, explainable evaluator for retrieval-augmented generation (RAG) answers.**

RAGLens checks whether an answer is supported by the context supplied to it, whether the context contains terms relevant to the question, and whether the answer makes claims that are absent from the context. It is a transparent baseline—not a replacement for human evaluation or an LLM judge.

## Why this exists

RAG demos often look convincing even when retrieval misses the relevant passage or an answer adds unsupported details. RAGLens gives developers a small, repeatable local check they can run before adding a model-based evaluator.

## Features

- No API key or external service required
- Simple CLI and Python API
- Transparent heuristic scores with explanations
- JSON output for CI pipelines and experiments
- Tests and sample examples

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e .
raglens evaluate --question "What database is used?" --context "The service stores records in PostgreSQL." --answer "It uses PostgreSQL."
```

Or run directly from the repository:

```bash
python -m raglens evaluate \
  --question "What database is used?" \
  --context "The service stores records in PostgreSQL." \
  --answer "It uses PostgreSQL."
```

## Example JSON output

```json
{
  "question": "What database is used?",
  "context_relevance": 0.5,
  "answer_support": 1.0,
  "unsupported_terms": [],
  "verdict": "likely_supported",
  "limitations": ["Heuristic baseline; scores are not calibrated probabilities."]
}
```

Scores are lexical heuristics, not semantic probabilities. In particular, synonyms, paraphrases, negation, numbers, and domain-specific language can cause false positives or negatives. Always review important results.

## Python API

```python
from raglens import evaluate

report = evaluate(
    question="What database is used?",
    context="The service stores records in PostgreSQL.",
    answer="It uses PostgreSQL.",
)
print(report.to_dict())
```

## Roadmap

- [ ] Optional semantic similarity backend
- [ ] Dataset-level evaluation and aggregate metrics
- [ ] Adapters for common RAG frameworks
- [ ] Optional model-based evaluation with clearly documented judge prompts

## Contributing

Issues and pull requests are welcome. Please include tests for behavior changes and describe limitations honestly. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).
