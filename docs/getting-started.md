# Getting Started

## Installation
- Source (editable): `uv pip install -e .`

Requirements: Python 3, NumPy/SciPy, SymPy, PyYAML, Numba.

Verify your install:
```bash
uv run python - << 'PY'
import dsge, sys
print("dsge loaded from:", dsge.__file__)
PY
```

## Documentation locally
Install tools and serve:
```bash
uv pip install mkdocs mkdocs-material "mkdocstrings[python]"
uv run mkdocs serve
```
Open the local URL to view the docs.
