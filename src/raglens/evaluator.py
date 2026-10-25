"""Explainable lexical baseline checks for retrieval-augmented answers.

These metrics are deliberately simple heuristics, not calibrated probabilities.
They compare *words*, not *meaning*.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

# Words that carry little meaning for overlap checks. Generic question words
# ("used", "many", "does") are included because they rarely appear in a
# context passage even when the context answers the question.
_STOP_WORDS = frozenset(
    """
    a an and are as at be been being by do does did for from had has have he her
    hers him his how i in into is it its me my many much of on or our ours she
    that the their them there they this to use used uses using was we were what
    when where which who why will with you your yours can could would should
    about than then so if but also
    """.split()
)

_NEGATIONS = frozenset(
    {"not", "no", "never", "none", "cannot", "without", "neither", "nor"}
)

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+(?:['’-][a-zA-Z0-9]+)*")
_NUMBER_RE = re.compile(r"^\d+(?:[.,]\d+)*$")


def _tokens(text: str) -> list[str]:
    """Split text into lowercase tokens (words and numbers)."""
    return [t.lower().replace("’", "'") for t in _TOKEN_RE.findall(text)]


def _stem(word: str) -> str:
    """Very small suffix stripper so 'stores', 'stored' and 'store' match.

    This is intentionally basic and not a full stemmer. Numbers are left alone.
    """
    if _NUMBER_RE.match(word):
        return word
    if word.endswith("'s"):
        word = word[:-2]
    if len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif len(word) > 4 and word.endswith("ing"):
        word = word[:-3]
    elif len(word) > 3 and word.endswith("ed"):
        word = word[:-2]
    elif len(word) > 3 and re.search(r"(s|x|z|ch|sh)es$", word):
        word = word[:-2]
    elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    if len(word) > 3 and word.endswith("e"):
        word = word[:-1]
    return word


def _content_terms(text: str) -> dict[str, str]:
    """Map each stemmed content term to the first surface word that produced it."""
    terms: dict[str, str] = {}
    for token in _tokens(text):
        if token in _STOP_WORDS or token in _NEGATIONS:
            continue
        if len(token) < 2 and not token.isdigit():
            continue
        terms.setdefault(_stem(token), token)
    return terms


def _numbers(text: str) -> set[str]:
    return {t.replace(",", "") for t in _tokens(text) if _NUMBER_RE.match(t)}


def _has_negation(text: str) -> bool:
    return any(t in _NEGATIONS or t.endswith("n't") for t in _tokens(text))


@dataclass(frozen=True)
class EvaluationReport:
    """Result of :func:`evaluate`. Convert to a plain dict with ``to_dict()``."""

    question: str
    context_relevance: float
    answer_support: float
    unsupported_terms: list[str]
    verdict: str
    explanations: list[str]
    limitations: list[str]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this report."""
        return asdict(self)


VERDICTS = ("likely_supported", "partially_supported", "needs_review")


def evaluate(question: str, context: str, answer: str) -> EvaluationReport:
    """Evaluate an answer against supplied context using lexical overlap.

    ``context_relevance``: fraction of meaningful question terms found in the
    context. ``answer_support``: fraction of meaningful answer terms found in
    the context. Neither understands meaning or entailment.

    ``warnings`` flags two cases the overlap scores cannot judge: numbers in the
    answer that are absent from the context, and a negation mismatch between
    answer and context. Any warning prevents a ``likely_supported`` verdict.

    Raises:
        ValueError: if any input is not a non-empty string.
    """
    for name, value in (("question", question), ("context", context), ("answer", answer)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")

    q_terms = _content_terms(question)
    c_terms = _content_terms(context)
    a_terms = _content_terms(answer)

    relevance = len(q_terms.keys() & c_terms.keys()) / len(q_terms) if q_terms else 0.0
    support = len(a_terms.keys() & c_terms.keys()) / len(a_terms) if a_terms else 0.0
    unsupported = sorted(a_terms[s] for s in a_terms.keys() - c_terms.keys())

    explanations: list[str] = []
    if not q_terms:
        explanations.append("The question has no meaningful terms after basic stop-word filtering.")
    elif relevance < 0.25:
        explanations.append(
            "Few meaningful question terms appear in the context; retrieval may be off-topic."
        )
    else:
        explanations.append("Some meaningful question terms appear in the context.")

    if not a_terms:
        explanations.append("The answer has no meaningful terms after basic stop-word filtering.")
    elif unsupported:
        explanations.append(
            "Some answer terms do not appear in the context; review whether they are "
            "unsupported claims or valid paraphrases."
        )
    else:
        explanations.append("All meaningful answer terms appear in the context under this lexical check.")

    warnings: list[str] = []
    missing_numbers = sorted(_numbers(answer) - _numbers(context))
    if missing_numbers:
        warnings.append(
            "Numbers in the answer not found in the context: " + ", ".join(missing_numbers) + "."
        )
    if _has_negation(answer) != _has_negation(context):
        warnings.append(
            "Negation differs between answer and context (e.g. 'not', 'no'); check the meaning."
        )

    if support >= 0.8 and relevance >= 0.25:
        verdict = "likely_supported"
    elif support < 0.5 or relevance < 0.25:
        verdict = "needs_review"
    else:
        verdict = "partially_supported"
    if warnings and verdict == "likely_supported":
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
        warnings=warnings,
    )
