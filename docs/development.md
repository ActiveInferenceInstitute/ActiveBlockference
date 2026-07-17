# Development

Use the committed lockfile for the exact environment:

```bash
uv sync --locked --extra dev
uv run ruff check blockference tests GRTs scripts
uv run pytest
uv run python scripts/execute_notebooks.py
uv run python scripts/validate_manuscript.py
uv run python scripts/build_manuscript.py
uv run python scripts/build_manuscript.py --check
```

The development extra includes pytest, Ruff, notebook parsing/execution, and a
Python kernel. The runtime package includes both radCAD and cadCAD so backend
integration tests run rather than being omitted.

Ruff uses a 100-column line length and import sorting. New public functions
need docstrings and tests. Do not commit generated output trees, caches, or
virtual environments; `uv.lock` is intentionally tracked.

The manuscript source is under `docs/manuscript/`. Its ordered Markdown files,
`config.yaml`, `references.bib`, and `figures/` are the publication record.
`scripts/build_manuscript.py` composes a numbered Markdown manuscript and can
be passed to Pandoc for PDF or HTML rendering; `docs/_build/` is disposable.
