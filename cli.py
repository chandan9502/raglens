"""Command-line interface for RAGLens."""
import argparse
import json
import sys
from .evaluator import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run explainable lexical checks on a RAG answer.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ev = subparsers.add_parser("evaluate", help="Evaluate one question/context/answer triple")
    ev.add_argument("--question", required=True, help="User question")
    ev.add_argument("--context", required=True, help="Retrieved context supplied to the model")
    ev.add_argument("--answer", required=True, help="Generated answer to inspect")
    ev.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    args = parser.parse_args()
    try:
        report = evaluate(args.question, args.context, args.answer)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report.to_dict(), indent=2 if args.pretty else None, ensure_ascii=False))


if __name__ == "__main__":
    main()
