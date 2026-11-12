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

Arms are listed in [`docs/gate.md`](docs/gate.md). Use absolute
`blockference.*` imports, NumPy-style docstrings, and 100-column Ruff
formatting. Runtime dependencies belong in `pyproject.toml` and the
generated `uv.lock`. Do not commit generated result files, caches, virtual
environments, or notebook checkpoints.

## Generative-model specification

The grid-world model is declared in `specs/gridworld_3x3.gnn.md` (Generalized
Notation Notation) and generated from the implementation by
`blockference.gnn.render_spec`; `uv run python scripts/export_gnn_spec.py
--check` fails when the committed document drifts. The upstream parser is
pinned in `specs/gnn-pin.json` (GNN v5.0.3, commit `8f293d9ba`), and
`scripts/validate_gnn_spec.py` parses the document with that revision, builds
the runtime model from the declared parameters, and requires the declared
matrices to match. That check needs an external runtime (`GNN_ROOT` or
`GNN_RUNTIME_PYTHON`); the required `gnn-conformance` CI job provisions one from
the pin and runs `scripts/release_check.py --arm gnn`. Do not
hand-edit `specs/gridworld_3x3.gnn.md`: change the model or fixture in
`blockference/` and regenerate with `scripts/export_gnn_spec.py`. See
`docs/gnn.md`.

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
