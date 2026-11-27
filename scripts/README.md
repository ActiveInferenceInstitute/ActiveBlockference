# `scripts/`

Repository gates and generators, all run from the repository root:

| Script | Purpose |
|---|---|
| `release_check.py` | The canonical gate: composable arms for lint, types, fast/full tests, docs, manuscript, GNN, smoke, package, and render. |
| `gate_metrics.py` | Imports disjoint baseline receipts with repeatable `--record-baseline PATH` and one current receipt with `--record-timings PATH`, writes `docs/_generated/gate-metrics.md`, and checks drift with `--check`. |
| `manuscript_variables.py` | Derives variables from project/CI metadata, the metadata-selected experiment, and checked producer pin inspection; `--check` gates drift. Inspection is not GNN execution qualification. |
| `check_docs_links.py` | Validates internal Markdown links and anchors across tracked documentation. |
| `build_manuscript.py` | Builds the manuscript and its figures, injects declared variables, and renders with Pandoc (`--render pdf\|html`). |
| `validate_manuscript.py` | Fail-closed manuscript validation, including declared variables. |
| `execute_notebooks.py` | Executes the frozen tracked selection owned by `blockference.notebook_inventory` in a temporary workspace. |
| `validate_notebooks.py` | Uses the same tracked inventory and hygiene producer; successful empty selection remains explicitly empty. |
| `export_gnn_spec.py` | Writes or verifies `specs/gridworld_3x3.gnn.md` from the implementation. |
| `validate_gnn_spec.py` | Owns deep model conformance and shared importer checks through isolated, fully admitted external workers, with separate source/installed provenance and optional remote tag verification. |

The external worker is `blockference/gnn_source.py`, invoked directly with
`-I -B` rather than importing the project into the GNN NumPy environment.
`gnn_parser_probe.py` and `gnn_importer_report.py` are retired. Select the
prepared interpreter and reviewed bootstrap/full manifest per
[`../docs/gnn.md`](../docs/reference/gnn.md) before release. Missing inputs fail.
[`../docs/gate.md`](../docs/development/gate.md) owns the arm commands, CI mapping,
`--timings PATH` recording and metrics-import workflow.

The gate reuses the already-synced locked project environment by setting
`UV_NO_SYNC=1` for its commands. Its package check builds into an invocation-owned
directory, requires exactly one fresh regular wheel, constrains dependencies
with locked `uv export`, and imports outside the source checkout. Repository
`dist` is neither installed from nor cleaned. PDF rendering remains an explicit local
`--arm render` selection of `build_manuscript.py --render pdf`; it is not part
of `all` or CI.

Manuscript configuration comes from the confined `experiment.config_path`,
not a hardcoded fallback. Composition rebases source-relative links into the
selected output directory and renders using copied bibliography, metadata,
and registered figures; missing ordinary resources fail.
