# AGENTS.md — `specs/`

- `gridworld_3x3.gnn.md` is generated. Do not hand-edit it; change the model or
  fixture in `blockference/` and run `uv run python scripts/export_gnn_spec.py`.
- Validate against the revision in `gnn-pin.json`, never a live GNN checkout. A
  re-pin updates the tag, commit and every parser digest.
- The declared matrices must match the runtime model; a specification-only edit
  is a drift.
- Parent rules: [../AGENTS.md](../AGENTS.md).
