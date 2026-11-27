# `specs/`

- [`gridworld_3x3.gnn.md`](gridworld_3x3.gnn.md) — the GNN declaration of the
  3x3 grid-world model, generated from the implementation.
- [`gnn-pin.json`](gnn-pin.json) — the upstream repository, tag `v5.1.0`, commit
  `75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`, and the SHA-256 of each parser
  file, under schema `gnn-importer-pin/v1`. The pin is read by the upstream
  `gnn.importers` reader, not by a local copy.

Verify exporter drift with `uv run python scripts/export_gnn_spec.py --check`,
then run `uv run python scripts/validate_gnn_spec.py` from the project runtime
with the separate GNN interpreter and independently reviewed bootstrap/full
manifest selected per [`../docs/gnn.md`](../docs/reference/gnn.md). The validator owns
both admitted parser and shared importer verdicts; direct unchecked imports
are not an alternative. The renderer is
[`../blockference/gnn.py`](../blockference/gnn.py).
