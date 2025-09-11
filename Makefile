.PHONY: docs-install docs-serve docs-build

docs-install:
	uv pip install mkdocs mkdocs-material "mkdocstrings[python]"

docs-serve:
	uv run mkdocs serve

docs-build:
	uv run mkdocs build --strict
