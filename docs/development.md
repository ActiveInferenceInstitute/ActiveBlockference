# Development

Use the committed lockfile for the exact environment:

```bash
uv sync --locked --extra dev
uv run python scripts/release_check.py            # fast arms (default)
uv run python scripts/release_check.py --arm all --gnn-runtime /path/to/gnn-runtime/bin/python
```

[`gate.md`](gate.md) lists every arm and CI job; `--list` prints the exact
commands. The fast default runs lint, types, the fast test selection, and docs.
The release gate adds the full suite with the 90% coverage floor, manuscript
validation/build/drift checks, the GNN pin/pointer reader, drift and conformance
checks, pipeline smoke runs, and the wheel import check.

mypy targets the `requires-python` floor (`python_version = "3.11"` in
`pyproject.toml`), so one run covers the minimum supported stubs. Notebook
execution, real-pipeline integration, CLI smoke, and figure-byte checks carry
the `slow` marker and are excluded from `--arm test`. The recorded before/after
arm times, including the render arm, are in
[`_generated/gate-metrics.md`](_generated/gate-metrics.md).

The `gnn` arm always runs the shared `python -m gnn.importers check --repo .`
pin and pointer reader plus pinned parser conformance
(`scripts/validate_gnn_spec.py`); it fails without the external runtime rather
than skipping validation. Replace the runtime path above with an installed
pinned interpreter, or configure the source-checkout route in [`gnn.md`](gnn.md).
CI's `gnn-conformance` job runs `all` on Python 3.12 with that separate runtime;
the Python 3.11 `test` job runs `test-full`. There is no separate `gate` job.

Markdown links and anchors across all tracked documentation are validated by
`scripts/check_docs_links.py`, which is also part of the canonical release
gate.

The development extra includes pytest, Ruff, notebook parsing/execution, and a
Python kernel. The runtime package includes both radCAD and cadCAD so backend
integration tests run rather than being omitted.

Dependency bounds are kept to what the code and its runtime requirements
actually need. `pyproject.toml` declares `numpy>=1.25,<2.2.7`; the committed
`uv.lock` currently selects NumPy `1.26.3` with radCAD `0.14.0`. The declared
range does not itself exclude all NumPy 2.x versions, and the locked resolution
must not be confused with that range. Pinned GNN requires NumPy 2.x, so use its
separate runtime rather than changing the library's locked environment.

Ruff uses a 100-column line length and import sorting. New public functions
need docstrings and tests. Do not commit generated output trees, caches, or
virtual environments; `uv.lock` is intentionally tracked.

The manuscript source is under `docs/manuscript/`. Its ordered Markdown files,
`config.yaml`, `references.bib`, and `figures/` are the publication record.
[`figure-registry.json`](manuscript/figure-registry.json) records the figure
sources and artifacts. `scripts/build_manuscript.py` composes a numbered
Markdown manuscript; its authoritative PDF command is
`uv run python scripts/build_manuscript.py --render pdf`, also selected by the
explicit local gate arm `--arm render`. `docs/_build/` is disposable.

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
