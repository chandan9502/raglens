# Changelog

## 0.3.0

### Added
- `batch` reads an optional `label` field (`supported` / `unsupported`) per line and adds
  `label_metrics` to the summary: confusion matrix, precision, recall, F1 and accuracy for each
  `--fail-on` level, and the ROC AUC of `answer_support`.
- `raglens calibrate`: suggests the support or review threshold with the best F1 on labelled data.
- `--support-threshold`, `--review-threshold` and `--relevance-threshold` on every command, and a
  `Thresholds` object for `evaluate()` in Python. Defaults are unchanged (0.8 / 0.5 / 0.25).
- `raglens.metrics` (`binary_metrics`, `roc_auc`) and `raglens.calibrate`, with no new dependencies.
- `benchmarks/`: reproducible evaluation on RAGTruth (human labels) and HaluEval QA, with results
  in `benchmarks/RESULTS.md` and a summary in the README.
- `examples/labelled_sample.jsonl`.

### Changed
- The verdict is now decided from the rounded scores shown in the report, so the two always agree.
  Before, a score shown as 0.8 could in rare cases come from 0.79996 and miss `likely_supported`.

### Fixed
- Packaging metadata uses an SPDX licence expression (`license = "MIT"`). setuptools had warned
  that the old form stops being supported in February 2027. Building now needs setuptools 77+.

### Documented
- Measured limitations: the negation warning barely beats the base rate on RAGTruth, and refusals
  such as "Unable to answer..." are usually flagged `needs_review`.

## 0.2.0

### Fixed
- Packaging: the package now lives in `src/raglens/`, so `pip install` works and the
  `raglens` command can import its modules. Previously the modules sat outside any package,
  the installed command failed with `ModuleNotFoundError`, and `pytest` could not import `raglens`.
- `python -m raglens` now works (added `__main__.py`).
- README example output now matches real output; the old example claimed a result the code did not produce.
- `LICENSE` text was truncated (missing the liability sentence); now the full MIT text.
- Removed committed `.pyc` files and added `.gitignore`.

### Added
- Light word-form matching (`store`/`stores`/`stored`) and extra generic question stop words.
- `warnings` field: numbers missing from the context, negation mismatch.
- CLI: `--version`, `--context-file`, `--answer-file`, `--fail-on`, and a `batch` command for JSONL files.
- Tests for the evaluator and CLI, `ruff` linting, and GitHub Actions CI (Python 3.10-3.13 + wheel build).

## 0.1.0

- Initial prototype.
