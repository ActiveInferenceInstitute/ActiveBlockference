# ActiveBlockference contributor guide

ActiveBlockference is a Python 3.10+ package for discrete Active Inference
models in radCAD and cadCAD. The canonical loop is observation, inference,
expected-free-energy evaluation, action sampling, and prior propagation.

## Development

```bash
uv sync --locked --extra dev
uv run ruff check blockference tests
uv run pytest
```

Use absolute `blockference.*` imports, NumPy-style docstrings, and 100-column
Ruff formatting. Runtime dependencies belong in `pyproject.toml` and the
generated `uv.lock`. Do not commit generated result files, caches, virtual
environments, or notebook checkpoints.

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
