# CI guidance

The workflow installs `uv.lock`, then calls the arms of
`scripts/release_check.py` instead of repeating commands. The `test` matrix job
runs `--arm test-full`; the `gate` job runs lint, types, docs, manuscript,
smoke, and package once on 3.12; the `gnn-conformance` job provisions a Python
3.12 runtime from the GNN revision pinned in `specs/gnn-pin.json` and runs
`--arm gnn`. It is a required check and must not be converted into a skip when
the runtime cannot be provisioned.

Manuscript figure bytes are written by the host toolchain, and `uv.lock`
resolves different matplotlib versions for different Python versions, so the
release gate excludes `docs/manuscript/figures/*.svg` and `*.png` from its
generated-drift diff. `scripts/build_manuscript.py --check` keeps the
structural checks for those two artefact kinds, and every other generated
artifact keeps byte-identity. Do not widen that exclusion.
