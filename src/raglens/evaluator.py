"""Explainable lexical baseline checks for retrieval-augmented answers.

These metrics are deliberately simple heuristics, not calibrated probabilities.
They compare *words*, not *meaning*.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
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


# --- Abstentions ---------------------------------------------------------------
# An answer that only says it cannot answer ("Unable to answer based on the given
# passages.") makes no claims, so marking it unsupported would be a false alarm.
_ABSTENTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern)
    for pattern in (
        r"\b(?:unable|not able) to (?:answer|determine|find|provide|say|tell|identify|confirm)\b",
        r"\b(?:cannot|can't|can not|could not|couldn't) "
        r"(?:answer|determine|find|provide|say|tell|identify|confirm)\b",
        r"\b(?:i|we) (?:do not|don't) know\b",
        r"\b(?:not enough|insufficient|no relevant|no specific|no) (?:information|details|data)\b",
        r"\b(?:passages?|context|documents?|sources?|texts?|information) (?:provided |given |above )?"
        r"(?:does|do|did) not (?:contain|provide|mention|include|specify|say|state|answer|give|offer)\b",
        r"\bthere (?:is|are) no (?:mention|information|indication|details?)\b",
        # Only verbs about the text itself: "was not included in the album" is a claim.
        r"\b(?:is|are|was|were) not (?:mentioned|specified|stated)\b",
    )
)
# Words that often surround a refusal, e.g. "Sorry, I am unable to answer based on
# the given passages." They are allowed in a refusal sentence; any other new word
# is treated as a possible claim.
_ABSTENTION_FILLER = frozenset(
    _stem(word)
    for word in """
    based given provided provide passage passages context information document documents source sources
    text texts question answer answers available mention mentioned contain contains sufficient enough
    specific relevant detail details according data regarding unable able determine find say tell
    identify confirm sorry unfortunately apologies apologize apologise afraid am i'm we're currently
    exact date time year price cost number amount name location reason
    """.split()
)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _normalize(text: str) -> str:
    return text.lower().replace("’", "'")


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    """Compile a user's refusal phrase: case-insensitive, any whitespace between words."""
    if not isinstance(phrase, str):
        raise TypeError(f"abstention phrases must be strings, got {phrase!r}")
    words = _normalize(phrase).strip(" \t\n.!?,;:\"'").split()
    if not words:
        raise ValueError("abstention phrases must contain at least one word")
    return re.compile(r"\b" + r"\s+".join(re.escape(word) for word in words) + r"\b")


def _is_abstention(answer: str, allowed: set[str], patterns: Sequence[re.Pattern[str]]) -> bool:
    """True if the answer only declines to answer.

    Every sentence that has content must match a refusal pattern, and once the
    matched words are removed it may only repeat terms from ``allowed`` (the
    question and context) or refusal filler words. "The passages do not say when
    the cafe opened" restates the question, so it still counts. A sentence that
    adds anything else might be a claim, so the whole answer is then scored
    normally. That keeps partial answers ("No date is given, but it is in Paris")
    under review.
    """
    saw_refusal = False
    for sentence in _SENTENCE_RE.split(answer):
        if not _content_terms(sentence):
            continue
        text = _normalize(sentence)
        spans = [match.span() for pattern in patterns for match in pattern.finditer(text)]
        if not spans:
            return False
        remainder = list(text)
        for start, end in spans:
            remainder[start:end] = " " * (end - start)
        if _content_terms("".join(remainder)).keys() - allowed - _ABSTENTION_FILLER:
            return False
        saw_refusal = True
    return saw_refusal


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
    abstained: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this report."""
        return asdict(self)


# Ordered from least to most in need of review. The CLI's --fail-on and calibration
# rely on this order. "abstained" means the answer declined to answer, so it makes
# no claims to check; it is never flagged.
VERDICTS = ("abstained", "likely_supported", "partially_supported", "needs_review")


@dataclass(frozen=True)
class Thresholds:
    """Cut-offs that turn the two scores into a verdict.

    The defaults are hand-picked starting points, not values fitted to data.
    ``raglens calibrate`` can suggest better ones from labelled examples.

    - ``support``: ``answer_support`` needed for ``likely_supported``.
    - ``review``: ``answer_support`` below this gives ``needs_review``.
    - ``relevance``: ``context_relevance`` below this gives ``needs_review``.

    Raises:
        ValueError: if a value is outside 0-1 or ``review`` is above ``support``.
    """

    support: float = 0.8
    review: float = 0.5
    relevance: float = 0.25

    def __post_init__(self) -> None:
        for name in ("support", "review", "relevance"):
            value = getattr(self, name)
            # bool is a subclass of int in Python, so reject it explicitly.
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} threshold must be a number from 0 to 1, got {value!r}")
        if self.review > self.support:
            raise ValueError(
                f"review threshold ({self.review}) cannot be higher than support threshold ({self.support})"
            )


DEFAULT_THRESHOLDS = Thresholds()


def decide_verdict(
    answer_support: float,
    context_relevance: float,
    has_warnings: bool,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    abstained: bool = False,
) -> str:
    """Turn scores into a verdict.

    :func:`evaluate` and calibration both call this function, so a threshold
    suggested by calibration produces exactly the verdicts it was measured with.
    An abstention is always ``abstained``, whatever the scores. Otherwise any
    warning stops a result from being ``likely_supported``.
    """
    if abstained:
        return "abstained"
    if answer_support >= thresholds.support and context_relevance >= thresholds.relevance:
        verdict = "likely_supported"
    elif answer_support < thresholds.review or context_relevance < thresholds.relevance:
        verdict = "needs_review"
    else:
        verdict = "partially_supported"
    if has_warnings and verdict == "likely_supported":
        verdict = "partially_supported"
    return verdict


def evaluate(
    question: str,
    context: str,
    answer: str,
    *,
    thresholds: Thresholds | None = None,
    abstention_phrases: Sequence[str] = (),
    detect_abstentions: bool = True,
) -> EvaluationReport:
    """Evaluate an answer against supplied context using lexical overlap.

    ``context_relevance``: fraction of meaningful question terms found in the
    context. ``answer_support``: fraction of meaningful answer terms found in
    the context. Neither understands meaning or entailment.

    ``warnings`` flags two cases the overlap scores cannot judge: numbers in the
    answer that are absent from the context, and a negation mismatch between
    answer and context. Any warning prevents a ``likely_supported`` verdict.

    An answer that only declines to answer ("Unable to answer based on the given
    passages.") gets the verdict ``abstained`` and ``abstained=True``: it makes no
    claims, so it is not flagged. Add your own refusal wording with
    ``abstention_phrases``, or turn the check off with ``detect_abstentions=False``.
    Scores are computed the same way either way.

    ``thresholds`` changes only the verdict, never the scores or warnings.
    Leave it out to use :data:`DEFAULT_THRESHOLDS`.

    Raises:
        ValueError: if any input is not a non-empty string, or a phrase is empty.
        TypeError: if ``abstention_phrases`` is a single string, not a list of them.
    """
    for name, value in (("question", question), ("context", context), ("answer", answer)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    if isinstance(abstention_phrases, str):
        raise TypeError("abstention_phrases must be a list of strings, not a single string")
    patterns = _ABSTENTION_PATTERNS + tuple(_phrase_pattern(p) for p in abstention_phrases)

    q_terms = _content_terms(question)
    c_terms = _content_terms(context)
    a_terms = _content_terms(answer)
    abstained = detect_abstentions and _is_abstention(answer, q_terms.keys() | c_terms.keys(), patterns)

    # Round once, then use the rounded values everywhere, so the verdict always
    # agrees with the scores shown in the report.
    relevance = round(len(q_terms.keys() & c_terms.keys()) / len(q_terms), 4) if q_terms else 0.0
    support = round(len(a_terms.keys() & c_terms.keys()) / len(a_terms), 4) if a_terms else 0.0
    unsupported = sorted(a_terms[s] for s in a_terms.keys() - c_terms.keys())

    explanations: list[str] = []
    if not q_terms:
        explanations.append("The question has no meaningful terms after basic stop-word filtering.")
    elif relevance < thresholds.relevance:
        explanations.append(
            "Few meaningful question terms appear in the context; retrieval may be off-topic."
        )
    else:
        explanations.append("Some meaningful question terms appear in the context.")

    if abstained:
        explanations.append(
            "The answer declines to answer and adds nothing beyond the question and context, so it "
            "makes no claims to check. Whether declining was right is not checked."
        )
    elif not a_terms:
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
    # A refusal is worded negatively ("cannot answer"), so a negation mismatch
    # says nothing about it.
    if not abstained and _has_negation(answer) != _has_negation(context):
        warnings.append(
            "Negation differs between answer and context (e.g. 'not', 'no'); check the meaning."
        )

    verdict = decide_verdict(support, relevance, bool(warnings), thresholds, abstained)

    return EvaluationReport(
        question=question,
        context_relevance=relevance,
        answer_support=support,
        unsupported_terms=unsupported,
        verdict=verdict,
        explanations=explanations,
        limitations=[
            "Heuristic baseline; scores are not calibrated probabilities.",
            "Lexical overlap does not establish factual entailment and may miss valid paraphrases.",
            "Negation, numbers, and context contradictions require human review.",
        ],
        warnings=warnings,
        abstained=abstained,
    )
