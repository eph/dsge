# Repository Guidelines
Preferred tooling: uv (no conda-specific paths)
## Project Structure & Module Organization
- `dsge/`: Core Python package (models, parsing, validation). Key modules: `DSGE.py`, `StateSpaceModel.py`, `symbols.py`, `validation.py`.
- `dsge/examples/`: Example YAML models and templates.
- `dsge/tests/`: Unit tests (`test_*.py`); some ad-hoc scripts live at repo root (e.g., `test_symbols.py`).
- `docs/`: Documentation drafts and notes.
- Build artifacts and temp folders: `build/`, `dist/`, `dsge.egg-info/`, `_tmp_cpp_export/` (do not commit changes here).

## Build, Test, and Development Commands
- Install (editable): `uv pip install -e .` or `uv run python dev_install.py`.
- Run tests (pytest): `uv run -m pytest -q dsge/tests`.
- Run tests (unittest): `uv run python -m unittest discover -s dsge/tests -p 'test_*.py'`.
- Quick import check: `uv run python -c "import dsge; print(dsge.__file__)"`.

## Coding Style & Naming Conventions
- Python 3, PEP 8, 4-space indentation. Prefer explicit, absolute imports (e.g., `from dsge.validation import ...`).
- Naming: modules/functions/variables use `snake_case`; classes use `CamelCase`; constants `UPPER_SNAKE_CASE`.
- Docstrings with triple quotes; include short type hints where practical.
- Keep YAML examples and schema consistent with `dsge/schema/`. Avoid hardcoded absolute paths in code and tests.

## Testing Guidelines
- Framework: tests are `unittest`-style and pytest-compatible. Place new tests under `dsge/tests/` named `test_*.py`.
- Aim to cover parsing, linearization, validation, and exported matrices. Use small, deterministic fixtures from `dsge/examples/`.
- Run locally with both commands above before opening a PR.

## Commit & Pull Request Guidelines
- Commits: prefer conventional prefixes when meaningful (`feat:`, `fix:`, `refactor:`, `test:`). Group small related changes.
- PRs: include a clear description, linked issues, reproduction steps, and updated tests. Paste key command outputs when relevant.
- Hygiene: do not commit generated artifacts (`build/`, `dist/`, `*.egg-info/`, `__pycache__/`, `_tmp_cpp_export/`, `*.pickle`). Keep changes focused and incremental.

## Security & Configuration Tips
- Configuration lives in YAML models under `dsge/examples/`. Validate with small inputs; avoid embedding credentials or absolute user paths.
- Large data or binaries should be referenced, not committed.
