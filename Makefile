.PHONY: docs-install docs-serve docs-build

docs-install:
	uv pip install mkdocs mkdocs-material mkdocs-macros-plugin "mkdocstrings[python]"

docs-serve:
	uv run mkdocs serve

docs-build:
	uv run mkdocs build --strict
