# CI guidance

The workflow installs `uv.lock` with `uv sync --locked --extra dev`, runs Ruff
and the full pytest suite, executes the complete pipeline smoke check, and
executes every tracked notebook. A separate `gnn conformance` job provisions a
Python 3.12 runtime from the GNN revision pinned in `specs/gnn-pin.json` and
runs `scripts/validate_gnn_spec.py`; it is a required check and must not be
converted into a skip when the runtime cannot be provisioned.

Manuscript figure PNG bytes are rasterized by the host toolchain, so the
release gate excludes `docs/manuscript/figures/*.png` from its generated-drift
diff while `scripts/build_manuscript.py --check` keeps the structural check and
byte-identity for every rasterizer-independent artifact. Do not widen that
exclusion.
