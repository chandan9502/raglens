"""Explainable lexical baseline checks for retrieval-augmented answers.

These metrics are deliberately simple heuristics, not calibrated probabilities.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "for", "from", "had", "has", "have", "he", "her", "hers", "him", "his",
    "i", "in", "is", "it", "its", "me", "my", "of", "on", "or", "our",
    "ours", "she", "that", "the", "their", "them", "there", "they", "this",
    "to", "was", "we", "were", "what", "when", "where", "which", "who",
    "why", "will", "with", "you", "your", "yours",
}
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+(?:['’-][a-zA-Z0-9]+)?")


def _tokens(text: str) -> list[str]:
    return [t.lower().replace("’", "'") for t in _TOKEN_RE.findall(text)]


def _content_tokens(text: str) -> set[str]:
    return {t for t in _tokens(text) if t not in _STOP_WORDS and len(t) > 1}


@dataclass(frozen=True)
class EvaluationReport:
    question: str
    context_relevance: float
    answer_support: float
    unsupported_terms: list[str]
    verdict: str
    explanations: list[str]
    limitations: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this report."""
        return asdict(self)


def evaluate(question: str, context: str, answer: str) -> EvaluationReport:
    """Evaluate an answer against supplied context using lexical overlap.

    ``context_relevance`` measures the fraction of meaningful question terms
    appearing in the context. ``answer_support`` measures the fraction of
    meaningful answer terms appearing in the context. These are intentionally
    basic lexical indicators; they do not understand meaning or entailment.
    """
    for name, value in (("question", question), ("context", context), ("answer", answer)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")

    q_terms = _content_tokens(question)
    c_terms = _content_tokens(context)
    a_terms = _content_tokens(answer)

    relevance = len(q_terms & c_terms) / len(q_terms) if q_terms else 0.0
    support = len(a_terms & c_terms) / len(a_terms) if a_terms else 0.0
    unsupported = sorted(a_terms - c_terms)
    explanations: list[str] = []

    if not q_terms:
        explanations.append("The question has no meaningful terms after basic stop-word filtering.")
    elif relevance < 0.25:
        explanations.append("Few meaningful question terms appear in the context; retrieval may be off-topic.")
    else:
        explanations.append("Some meaningful question terms appear in the context.")

    if unsupported:
        explanations.append("Some answer terms do not appear in the context; review whether they are unsupported claims or valid paraphrases.")
    else:
        explanations.append("All meaningful answer terms appear in the context under this lexical check.")

    if support >= 0.8 and relevance >= 0.25:
        verdict = "likely_supported"
    elif support < 0.5 or relevance < 0.25:
        verdict = "needs_review"
    else:
        verdict = "partially_supported"

    return EvaluationReport(
        question=question,
        context_relevance=round(relevance, 4),
        answer_support=round(support, 4),
        unsupported_terms=unsupported,
        verdict=verdict,
        explanations=explanations,
        limitations=[
            "Heuristic baseline; scores are not calibrated probabilities.",
            "Lexical overlap does not establish factual entailment and may miss valid paraphrases.",
            "Negation, numbers, and context contradictions require human review.",
        ],
    )
