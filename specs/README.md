# `specs/`

- [`gridworld_3x3.gnn.md`](gridworld_3x3.gnn.md) — the GNN declaration of the
  3x3 grid-world model, generated from the implementation.
- [`gnn-pin.json`](gnn-pin.json) — the upstream repository, tag `v5.1.0`, commit
  `75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`, and the SHA-256 of each parser
  file, under schema `gnn-importer-pin/v1`. The pin is read by the upstream
  `gnn.importers` reader, not by a local copy.

Verify: `uv run python scripts/export_gnn_spec.py --check`, then
`python -m gnn.importers check --repo .` and
`uv run python scripts/validate_gnn_spec.py` in the pinned runtime. The spec
renderer is [`../blockference/gnn.py`](../blockference/gnn.py).
