"""RAGLens: transparent baseline checks for RAG answers."""
from .evaluator import VERDICTS, EvaluationReport, evaluate

__version__ = "0.2.0"
__all__ = ["VERDICTS", "EvaluationReport", "__version__", "evaluate"]
