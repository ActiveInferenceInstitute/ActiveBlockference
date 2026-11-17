# AGENTS.md — `scripts/`

- These are thin entry points over `blockference.*`; new computation belongs in
  the package and gets a test under `tests/`.
- Target GNN imports belong only inside the fully qualified external
  `blockference/gnn_source.py` worker, launched directly with `-I -B`.
  Project orchestration and checked metadata inspection do not bypass
  independent bootstrap/full-manifest admission or merge NumPy environments.
- Do not add a file to `docs/manuscript/figures/`. `build_manuscript.py`
  requires every file there to be produced by a fresh build, so a guide in that
  directory fails the release gate.
- `export_gnn_spec.py --check` is a drift gate: change the model or fixture in
  `blockference/`, then regenerate.
- The pin file is the canonical record (`specs/gnn-pin.json`, schema
  `gnn-importer-pin/v1`); bump it by re-recording the upstream commit and parser
  digests, never by editing the generated document.
- Parent rules: [../AGENTS.md](../AGENTS.md).
