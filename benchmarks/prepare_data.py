"""Download two public hallucination datasets and convert them to RAGLens JSONL.

Usage (from the repository root):

    python benchmarks/prepare_data.py

Writes to benchmarks/data/ (git-ignored):

- ragtruth_qa_train.jsonl, ragtruth_qa_test.jsonl
    QA part of RAGTruth (Niu et al., ACL 2024): real answers from six LLMs to
    MS MARCO questions, each checked by human annotators. A response is
    labelled "unsupported" if annotators marked at least one hallucinated span
    in it, otherwise "supported". The dataset's own train/test split is kept.
- halueval_qa.jsonl
    QA part of HaluEval (Li et al., EMNLP 2023): HotpotQA questions with the
    correct answer ("supported") and a ChatGPT-written plausible wrong answer
    ("unsupported"), both judged against the same Wikipedia knowledge text.

Both datasets are MIT licensed. Files are downloaded from a fixed commit and
checked against a SHA-256 hash, so results can be reproduced exactly. Only the
Python standard library is used.
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
RAW = DATA / "raw"

RAGTRUTH = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/c103204b9ce28d6bbad859304bf30de72b8ed8fe"
HALUEVAL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/b7253db3cdaa0ab2c382f92b26b390109174f77e"

FILES = {
    "ragtruth_response.jsonl": (
        f"{RAGTRUTH}/dataset/response.jsonl",
        "e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073",
    ),
    "ragtruth_source_info.jsonl": (
        f"{RAGTRUTH}/dataset/source_info.jsonl",
        "0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b",
    ),
    "halueval_qa_data.json": (
        f"{HALUEVAL}/data/qa_data.json",
        "89ed139ec5e3a3169a0b30e45569ac1283846f76f27f7bb5e908ee6deed57e88",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(name: str) -> Path:
    """Download one file (unless already present) and verify its hash."""
    url, expected = FILES[name]
    path = RAW / name
    if not path.exists():
        print(f"downloading {name} ...", file=sys.stderr)
        RAW.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as out:
            out.write(response.read())
        tmp.rename(path)
    actual = _sha256(path)
    if actual != expected:
        raise SystemExit(f"{name}: SHA-256 mismatch (got {actual}); delete {path} and try again")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {path.relative_to(DATA.parent.parent)} ({len(rows)} rows)", file=sys.stderr)


def prepare_ragtruth() -> None:
    sources = {}
    with download("ragtruth_source_info.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            sources[row["source_id"]] = row

    splits: dict[str, list[dict]] = {"train": [], "test": []}
    with download("ragtruth_response.jsonl").open(encoding="utf-8") as f:
        for line in f:
            response = json.loads(line)
            source = sources[response["source_id"]]
            if source["task_type"] != "QA":
                continue
            splits[response["split"]].append({
                "question": source["source_info"]["question"],
                "context": source["source_info"]["passages"],
                "answer": response["response"],
                "label": "unsupported" if response["labels"] else "supported",
                "id": response["id"],
                "model": response["model"],
            })
    for split, rows in splits.items():
        _write_jsonl(DATA / f"ragtruth_qa_{split}.jsonl", rows)


def prepare_halueval() -> None:
    rows = []
    with download("halueval_qa_data.json").open(encoding="utf-8") as f:
        for number, line in enumerate(f):
            item = json.loads(line)
            for key, label in (("right_answer", "supported"), ("hallucinated_answer", "unsupported")):
                rows.append({
                    "question": item["question"],
                    "context": item["knowledge"],
                    "answer": item[key],
                    "label": label,
                    "id": f"{number}-{label}",
                })
    _write_jsonl(DATA / "halueval_qa.jsonl", rows)


if __name__ == "__main__":
    prepare_ragtruth()
    prepare_halueval()
