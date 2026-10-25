# Changelog

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
