# CI guidance

The workflow installs `uv.lock`, then calls the arms of
`scripts/release_check.py` instead of repeating commands. The `test` matrix
contains only Python 3.11 and runs `--arm test-full`. The `gnn-conformance` job
(display name `gnn conformance (pinned)`) runs `--arm all`
on Python 3.12 with `GNN_RUNTIME_PYTHON` set to the separate runtime provisioned from `specs/gnn-pin.json`.
There is no separate `gate` job. Pinned conformance always runs in the `gnn`
arm, where the shared `python -m gnn.importers check --repo .` reads the pin and
the pointer; missing runtime dependencies must fail, never skip. PDF rendering remains
an explicit local `--arm render` selection. The workflow uses pinned
`astral-sh/setup-uv` directly, not the Institute composite; see
[`../docs/gate.md`](../docs/gate.md) for the exact action and command map.

Manuscript figure bytes are written by the host toolchain, and `uv.lock`
resolves different matplotlib versions for different Python versions, so the
release gate excludes `docs/manuscript/figures/*.svg` and `*.png` from its
generated-drift diff. `scripts/build_manuscript.py --check` keeps the
structural checks for those two artefact kinds, and every other generated
artifact keeps byte-identity. Do not widen that exclusion.
