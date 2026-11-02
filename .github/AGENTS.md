# CI guidance

The workflow installs `uv.lock` with `uv sync --locked --extra dev`, runs Ruff
and the full pytest suite, executes the complete pipeline smoke check, and
executes every tracked notebook. A separate `gnn conformance` job provisions a
Python 3.12 runtime from the GNN revision pinned in `specs/gnn-pin.json` and
runs `scripts/validate_gnn_spec.py`; it is a required check and must not be
converted into a skip when the runtime cannot be provisioned.

Manuscript figure bytes are written by the host toolchain, and `uv.lock`
resolves different matplotlib versions for different Python versions, so the
release gate excludes `docs/manuscript/figures/*.svg` and `*.png` from its
generated-drift diff. `scripts/build_manuscript.py --check` keeps the
structural checks for those two artefact kinds, and every other generated
artifact keeps byte-identity. Do not widen that exclusion.
