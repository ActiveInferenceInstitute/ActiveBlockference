# `scripts/`

Repository gates and generators, all run from the repository root:

| Script | Purpose |
|---|---|
| `release_check.py` | The canonical gate: composable arms for lint, types, fast/full tests, docs, manuscript, GNN, smoke, package, and render. |
| `gate_metrics.py` | Imports disjoint baseline receipts with repeatable `--record-baseline PATH` and one current receipt with `--record-timings PATH`, writes `docs/_generated/gate-metrics.md`, and checks drift with `--check`. |
| `manuscript_variables.py` | Derives `docs/_generated/manuscript-variables.json` from `pyproject.toml`, the GNN pin, and CI; `--check` gates drift. |
| `check_docs_links.py` | Validates internal Markdown links and anchors across tracked documentation. |
| `build_manuscript.py` | Builds the manuscript and its figures, injects declared variables, and renders with Pandoc (`--render pdf\|html`). |
| `validate_manuscript.py` | Fail-closed manuscript validation, including declared variables. |
| `execute_notebooks.py` | Parses and executes every tracked notebook from the repository root. |
| `validate_notebooks.py` | Notebook hygiene checks before execution. |
| `export_gnn_spec.py` | Writes or verifies `specs/gridworld_3x3.gnn.md` from the implementation. |
| `validate_gnn_spec.py` | Validates the committed specification with the pinned parser, using a separate interpreter selected by `--gnn-runtime` or the documented environment variables. |
| `gnn_parser_probe.py` | The only module that imports the upstream `gnn` package; runs as a subprocess under the pin. |

Run the release selection with a pinned GNN runtime configured before
publishing; the no-argument default is the fast selection. The `gnn` arm always
runs real pinned conformance and fails if its runtime is unavailable.
[`../docs/gate.md`](../docs/gate.md) owns the arm commands, CI mapping,
`--timings PATH` recording and metrics-import workflow.

The gate reuses the already-synced locked project environment by setting
`UV_NO_SYNC=1` for its commands. Its package check constrains clean-wheel
dependencies using a locked `uv export` and imports the installed wheel from
outside the source checkout. PDF rendering remains an explicit local
`--arm render` selection of `build_manuscript.py --render pdf`; it is not part
of `all` or CI.
