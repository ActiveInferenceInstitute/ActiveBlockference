# Gate and CI

One entry point runs every local gate:
[`../scripts/release_check.py`](../scripts/release_check.py). Each arm is a
separately runnable selection, and CI invokes the same arms instead of keeping a
second copy of any command.

```bash
uv run python scripts/release_check.py              # fast arms (default)
uv run python scripts/release_check.py --arm all    # release gate
uv run python scripts/release_check.py --list       # arms and their commands
```

Measured wall times and test counts live in the generated
[`_generated/gate-metrics.md`](_generated/gate-metrics.md). Do not hand-edit
those numbers: refresh them with the commands the generated page records.

## Arms

| Arm | Command | In `all` |
| --- | --- | --- |
| `lint` | `uv run ruff check blockference tests GRTs scripts` | yes |
| `types` | `uv run mypy blockference` (configured at the `requires-python` floor) | yes |
| `test` | `uv run pytest -m "not slow"` | no |
| `test-full` | `uv run pytest --cov=blockference --cov-report=term --cov-report=xml --cov-fail-under=90` | yes |
| `docs` | `scripts/check_docs_links.py`, `scripts/gate_metrics.py --check` | yes |
| `manuscript` | `scripts/manuscript_variables.py --check`, `scripts/validate_manuscript.py`, `scripts/build_manuscript.py`, `scripts/build_manuscript.py --check` | yes |
| `gnn` | `scripts/export_gnn_spec.py --check`, pinned `scripts/validate_gnn_spec.py` when a runtime is configured | yes |
| `smoke` | pipeline, validation, and standalone `run` JSON smoke checks in a temporary directory | yes |
| `package` | `uv build` and a clean virtualenv import of the wheel | yes |
| `render` | Pandoc + XeLaTeX PDF render of the manuscript | no |

`--arm` is repeatable and arms always run in the canonical order above. `fast`
is the default because it is the selection used while editing; `all` is the
release gate. The `slow` marker (declared in `pyproject.toml`) keeps notebook
execution, real-pipeline integration, CLI smoke, and figure-byte checks out of
the fast arm. They stay in `test-full`.

## GNN conformance

The canonical pin is [`../specs/gnn-pin.json`](../specs/gnn-pin.json) with schema
`gnn-importer-pin/v1`. It pins
`ActiveInferenceInstitute/Generalized_Notation_Notation` at `v5.0.3`
(`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`) and records SHA-256 digests for the
five parser files under `src/gnn/`. The pointer
[`../.github/GNN.md`](../.github/GNN.md) records exactly what the conformance
job parses.

The conformance job's key is `gnn-conformance` and its display name is
`gnn conformance (pinned)`.

The `gnn` arm always runs the offline drift check. Pinned-parser conformance
runs when `GNN_ROOT` or `GNN_RUNTIME_PYTHON` is set, or when `--gnn-runtime` is
passed; otherwise the arm reports the run as not run, never as a pass. CI
provisions a runtime and passes `--require-gnn`.

## GitHub Actions

[`../.github/workflows/ci.yml`](../.github/workflows/ci.yml) calls these arms:

| Job | Arms |
| --- | --- |
| `test` (matrix 3.11, 3.12) | `--arm test-full` |
| `gate` (3.12) | `--arm lint --arm types --arm docs --arm manuscript --arm smoke --arm package` |
| `gnn-conformance` | `--arm gnn --gnn-runtime <runtime> --require-gnn` |

The matrix job therefore repeats only the test suite across supported Python
versions; lint, types, docs, manuscript, smoke, and package work run once. The
workflow installs uv with
`astral-sh/setup-uv@68559e99cc807c1eb7e9238fd43a15bba460eec2` (`v10.3.3`,
pinned by commit because setup-uv publishes no floating `v10` tag). It does not
use the composite action
`ActiveInferenceInstitute/.github/.github/actions/setup-python-uv@7159e67315330aac8e414aebae0ad1e999a41448`;
the workflow installs the locked environment directly.
