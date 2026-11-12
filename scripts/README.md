# `scripts/`

Repository gates and generators, all run from the repository root:

| Script | Purpose |
|---|---|
| `release_check.py` | The canonical gate: composable arms for lint, types, fast/full tests, docs, manuscript, GNN, smoke, package, and render. |
| `gate_metrics.py` | Writes `docs/_generated/gate-metrics.md` from a recorded gate receipt; `--check` gates drift. |
| `manuscript_variables.py` | Derives `docs/_generated/manuscript-variables.json` from `pyproject.toml`, the GNN pin, and CI; `--check` gates drift. |
| `check_docs_links.py` | Validates internal Markdown links and anchors across tracked documentation. |
| `build_manuscript.py` | Builds the manuscript and its figures, injects declared variables, and renders with Pandoc (`--render pdf\|html`). |
| `validate_manuscript.py` | Fail-closed manuscript validation, including declared variables. |
| `execute_notebooks.py` | Parses and executes every tracked notebook from the repository root. |
| `validate_notebooks.py` | Notebook hygiene checks before execution. |
| `export_gnn_spec.py` | Writes or verifies `specs/gridworld_3x3.gnn.md` from the implementation. |
| `validate_gnn_spec.py` | Validates the committed specification with the pinned parser. |
| `gnn_parser_probe.py` | The only module that imports the upstream `gnn` package; runs as a subprocess under the pin. |

Run `uv run python scripts/release_check.py --arm all` before publishing; the
no-argument default is the fast selection. See
[`../docs/gate.md`](../docs/gate.md) for the arm table and measured numbers.
