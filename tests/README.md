# Tests

The suite covers strict numerical primitives, closed configuration, canonical
grid dynamics, local RNG ownership, realized multi-agent transitions, real
radCAD/cadCAD runs, persisted artefacts, rendering, reproducible publication
assets, notebooks, and the credential-free research workflow.

```bash
uv run pytest -m "not slow"   # fast selection used by the default gate arm
uv run pytest                 # everything
```

The release-equivalent command is `uv run python scripts/release_check.py --arm
all`; it adds mypy, the 90% coverage gate, Markdown link/anchor checks, clean
notebook execution, CLI JSON smoke, manuscript drift checks, and a clean wheel
install. Notebook execution, real-pipeline integration, CLI smoke, and
figure-byte checks carry the `slow` marker; they run in `--arm test-full`, never
in the fast default.
