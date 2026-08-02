# Full pipeline

This tutorial runs the configured pipeline, writes the complete artefact tree,
and validates it again from disk. Run it from the repository root after
installing the locked development environment:

```bash
uv sync --locked --extra dev
uv run blockference pipeline \
  --config configs/example.yml \
  --output-root /tmp/activeblockference-tutorial \
  --run-name tutorial
```

Validate the resulting run independently:

```bash
uv run blockference validation \
  --run-dir /tmp/activeblockference-tutorial/tutorial
```

Both commands print JSON. The second command parses the CSV, JSON, model
archive, diagnostics, PNGs, GIF, and configuration before returning the
aggregate verdict. A successful result has `"ok": true`; this means the
software and persisted artefacts satisfy their integrity contracts, not that a
scientific hypothesis has been empirically established.

The example uses radCAD by default. Set `engine: cadcad` in a copied
configuration to exercise the alternative backend under the same state-update
contract. See [`pipeline.md`](../pipeline.md) for the required run tree and
[`development.md`](../development.md) for the complete release gate.

What to try next: replace `target: random` with a bounded `[y, x]` target, or
run `configs/smoke.yml` as the smallest configured example.
