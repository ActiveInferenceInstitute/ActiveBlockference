# Full pipeline

```bash
uv run blockference pipeline --config configs/example.yml --output-root output
uv run blockference validation --run-dir output/gridworld_example
```

The second command parses the CSV, JSON, model archive, diagnostics, PNGs, GIF,
and configuration before returning the aggregate verdict.
