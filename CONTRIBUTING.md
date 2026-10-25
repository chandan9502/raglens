# Contributing

Thanks for considering a contribution! Small fixes and new test cases are very welcome.

## Set up

```bash
git clone https://github.com/chandan9502/raglens.git
cd raglens
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Before opening a pull request

1. For anything bigger than a small fix, open an issue first to discuss it.
2. Add or update tests for behavior changes (`tests/`).
3. Run `ruff check .` and `python -m pytest`; CI runs both on Python 3.10-3.13.
4. If behavior or output changes, update `README.md` and `CHANGELOG.md`.
5. Do not include private, confidential, or personal data in examples.

## Project principles

- Keep the baseline explainable and dependency-light.
- Document limitations honestly.
- Do not present heuristic scores as proof that an answer is factually correct.

## Good first contributions

- Add a tricky question/context/answer case to `tests/` that shows a false positive or negative.
- Add realistic (non-private) rows to `examples/`.
- Improve error messages or documentation.
