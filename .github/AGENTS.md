# CI guidance

The workflow installs `uv.lock` with `uv sync --locked --extra dev`, runs Ruff
and the full pytest suite, executes the complete pipeline smoke check, and
executes every tracked notebook.
