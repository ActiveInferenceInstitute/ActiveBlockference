# Development

Use the committed lockfile for the exact environment:

```bash
uv sync --locked --extra dev
uv run ruff check blockference tests GRTs scripts
uv run python scripts/check_docs_links.py
uv run pytest
uv run python scripts/execute_notebooks.py
uv run python scripts/validate_manuscript.py
uv run python scripts/build_manuscript.py
uv run python scripts/build_manuscript.py --check
uv run python scripts/export_gnn_spec.py --check
```

`uv run python scripts/release_check.py` runs the whole sequence above plus
mypy, the pipeline smoke runs, and the wheel import check. The pinned GNN
parser conformance check (`scripts/validate_gnn_spec.py`) is part of that gate
only when `GNN_ROOT` or `GNN_RUNTIME_PYTHON` is set, because it needs an
external runtime; CI runs it as its own required job. See [`gnn.md`](gnn.md).

Markdown links and anchors across all tracked documentation are validated by
`scripts/check_docs_links.py`, which is also part of the canonical release
gate.

The development extra includes pytest, Ruff, notebook parsing/execution, and a
Python kernel. The runtime package includes both radCAD and cadCAD so backend
integration tests run rather than being omitted.

Dependency bounds are kept to what the code and its runtime requirements
actually need. The `numpy` bound `>=1.25,<1.26.4` mirrors the requirement of
`radcad` 0.14.0, which is the current release of the simulation backend: a
`numpy` 2.x resolution fails with "no solution found" while `radcad` is a
required dependency, so the bound is a required pin rather than a habit.
Remove it only when `radcad` widens its own requirement.

Ruff uses a 100-column line length and import sorting. New public functions
need docstrings and tests. Do not commit generated output trees, caches, or
virtual environments; `uv.lock` is intentionally tracked.

The manuscript source is under `docs/manuscript/`. Its ordered Markdown files,
`config.yaml`, `references.bib`, and `figures/` are the publication record.
`scripts/build_manuscript.py` composes a numbered Markdown manuscript and can
be passed to Pandoc for PDF or HTML rendering; `docs/_build/` is disposable.

Figure SVG and PNG output is written by the host toolchain, so its bytes
legitimately differ between the maintainer host and CI: matplotlib changes
glyph identifiers, text layout, coordinate formatting, and rasterizer output
between releases, and `uv.lock` resolves different matplotlib versions for
different Python versions. Committed figures are checked structurally (SVG:
well-formed document, positive viewBox, version-free Dublin Core title, text
drawn as paths; PNG: signature, canvas, 8-bit RGB/RGBA), their bytes are
excluded from the release gate's generated-drift diff, and byte equality is
asserted within one toolchain by `tests/test_publication.py`. Regenerate both
formats with `uv run python scripts/build_manuscript.py`.
