# AGENTS.md — `scripts/`

- These are thin entry points over `blockference.*`; new computation belongs in
  the package and gets a test under `tests/`.
- Keep `gnn_parser_probe.py` the only importer of the upstream `gnn` package.
  The validator runs it in a subprocess with the pinned tree first on
  `PYTHONPATH`.
- Do not add a file to `docs/manuscript/figures/`. `build_manuscript.py`
  requires every file there to be produced by a fresh build, so a guide in that
  directory fails the release gate.
- `export_gnn_spec.py --check` is a drift gate: change the model or fixture in
  `blockference/`, then regenerate.
- The pin file is the canonical record (`specs/gnn-pin.json`, schema
  `gnn-importer-pin/v1`); bump it by re-recording the upstream commit and parser
  digests, never by editing the generated document.
- Parent rules: [../AGENTS.md](../AGENTS.md).
