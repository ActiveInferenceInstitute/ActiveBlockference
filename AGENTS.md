# ActiveBlockference contributor guide

ActiveBlockference is a Python 3.11+ package for discrete Active Inference
models in radCAD and cadCAD. The canonical loop is observation, inference,
expected-free-energy evaluation, action sampling, and prior propagation.

## Development

```bash
uv sync --locked --extra dev
uv run python scripts/release_check.py            # fast arms (default)
uv run python scripts/release_check.py --arm all  # release gate
```

Arms are listed in [`docs/development/gate.md`](docs/development/gate.md). Use absolute
`blockference.*` imports, NumPy-style docstrings, and 100-column Ruff
formatting. Runtime dependencies belong in `pyproject.toml` and the
generated `uv.lock`. Do not commit generated result files, caches, virtual
environments, or notebook checkpoints.

## Generative-model specification

The grid-world model is declared in `specs/gridworld_3x3.gnn.md` (Generalized
Notation Notation) and generated from the implementation by
`blockference.gnn.render_spec`; `uv run python scripts/export_gnn_spec.py
--check` fails when the committed document drifts. The upstream parser is
pinned in `specs/gnn-pin.json` (GNN v5.1.0, commit `75c834bd8`).
`scripts/validate_gnn_spec.py` owns both deep conformance and the shared
pin/pointer/workflow/job check through `blockference/gnn_source.py`.
Require the selected separate interpreter plus independently reviewed
`GNN_BOOTSTRAP` and `GNN_SOURCE_MANIFEST` resources. Complete source/module
admission must remain active through verdict exit; metadata inspection alone
is not qualification. Never add a local pin reader or an unchecked direct
`-m gnn.importers` path. Installed provenance and source-artifact proof remain
distinct. Hosted resource delivery is a release prerequisite, not an assumed
published artifact. The required `gnn-conformance` job runs the `gnn` arm.
Do not hand-edit `specs/gridworld_3x3.gnn.md`: change the model or fixture in
`blockference/` and regenerate with `scripts/export_gnn_spec.py`. See
`docs/reference/gnn.md`.

## Contracts

`GridWorld` and `blockference.gridference._move` use `(y, x)` coordinates and
the configured affordance order. All probability arrays are finite,
non-negative, shape-checked, and normalised where required. Configuration
loaders reject unknown keys and invalid paths. Persistence receives an already
prepared `RunPaths` tree. `ValidationReport.ok` and `PipelineResult.ok` are the
single release verdicts.

Notebook cells must begin with a short teaching summary, import the current
package API, contain no absolute paths, and execute successfully in a clean
workspace. Tests are deterministic and never use network services.
