# `scripts/`

Repository gates and generators, all run from the repository root:

| Script | Purpose |
|---|---|
| `release_check.py` | The canonical local release gate: lint, tests, docs links, manuscripts, notebooks, GNN conformance. |
| `check_docs_links.py` | Validates internal Markdown links and anchors across tracked documentation. |
| `build_manuscript.py` | Builds the manuscript and its publication figures; `_check_figures_against_committed` compares a fresh build with the committed figures. |
| `validate_manuscript.py` | Fail-closed manuscript validation. |
| `execute_notebooks.py` | Parses and executes every tracked notebook from the repository root. |
| `validate_notebooks.py` | Notebook hygiene checks before execution. |
| `export_gnn_spec.py` | Writes or verifies `specs/gridworld_3x3.gnn.md` from the implementation. |
| `validate_gnn_spec.py` | Validates the committed specification with the pinned parser. |
| `gnn_parser_probe.py` | The only module that imports the upstream `gnn` package; runs as a subprocess under the pin. |

Run `uv run python scripts/release_check.py` before publishing.
[`../docs/manuscript/README.md`](../docs/manuscript/README.md) covers the
manuscript pipeline.
