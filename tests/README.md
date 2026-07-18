# Tests

The suite covers strict numerical primitives, closed configuration, canonical
grid dynamics, local RNG ownership, realized multi-agent transitions, real
radCAD/cadCAD runs, persisted artefacts, rendering, reproducible publication
assets, notebooks, and the credential-free research workflow. Run it with
`uv run pytest` after `uv sync --locked --extra dev`.
The release-equivalent command is `uv run python scripts/release_check.py`;
it adds mypy, the 90% coverage gate, clean notebook execution, CLI JSON smoke,
manuscript drift checks, and a clean wheel install.
