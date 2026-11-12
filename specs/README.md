# `specs/`

- [`gridworld_3x3.gnn.md`](gridworld_3x3.gnn.md) — the GNN declaration of the
  3x3 grid-world model, generated from the implementation.
- [`gnn-pin.json`](gnn-pin.json) — the upstream repository, tag `v5.0.3`, commit
  `8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`, and the SHA-256 of each parser
  file, under schema `gnn-importer-pin/v1`.

Verify: `uv run python scripts/export_gnn_spec.py --check`, then
`uv run python scripts/validate_gnn_spec.py`. The spec renderer is
[`../blockference/gnn.py`](../blockference/gnn.py).
