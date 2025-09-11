# Contributing

For contributor guidance, see `AGENTS.md` at the repository root. It covers project structure, commands, coding style, testing, and PR expectations.

Quick checks before opening a PR:
- Lint/style: follow PEP 8 and use 4‑space indentation; prefer Google‑style docstrings for new/updated code.
- Tests: add `dsge/tests/test_*.py` and run `uv run -m pytest -q dsge/tests`.
- No artifacts: avoid committing `build/`, `dist/`, `*.egg-info/`, `__pycache__/`, `_tmp_cpp_export/`.

Docs development:
```bash
uv pip install mkdocs mkdocs-material "mkdocstrings[python]"
uv run mkdocs serve
```
