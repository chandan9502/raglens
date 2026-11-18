"""RAGLens: transparent baseline checks for RAG answers."""
from .calibration import calibrate
from .evaluator import DEFAULT_THRESHOLDS, VERDICTS, EvaluationReport, Thresholds, evaluate

__version__ = "0.3.0"
__all__ = [
    "DEFAULT_THRESHOLDS",
    "VERDICTS",
    "EvaluationReport",
    "Thresholds",
    "__version__",
    "calibrate",
    "evaluate",
]
