# Development

Conventions and tooling for working on `blockference` itself.

## Environment (uv)

We standardise on [`uv`](https://docs.astral.sh/uv/). Install once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # or: pipx install uv
```

Bootstrap the project:

```bash
uv venv --python 3.11
uv pip install -e ".[dev,docs]"
```

The `[dev]` extra installs `pytest`, `pytest-cov`, `ruff`, `ipykernel`,
`jupyterlab`. The `[docs]` extra installs `mkdocs` + `mkdocs-material`
for optional doc-site builds.

A fully-pinned snapshot of the verified environment is checked in as
`requirements.lock` (generated via `uv pip freeze`). Recreate exactly:

```bash
uv venv --python 3.11
uv pip install -r requirements.lock
```

## Testing

```bash
uv run pytest                                         # full suite
uv run pytest tests/test_gridference.py               # one file
uv run pytest -k "not simulation"                     # skip the slow ones
uv run pytest --cov=blockference --cov-report=html    # HTML coverage report
```

* Unit tests should run in **<2 seconds** each.
* Smoke tests (`tests/test_simulations.py`) may run a tiny cadCAD
  experiment but must complete in <10 seconds total.

## Linting and formatting

```bash
uv run ruff check blockference tests
uv run ruff format blockference tests
```

Configuration lives in `pyproject.toml [tool.ruff]`. Line length is 100;
the import-sort rule (`I`) is enabled.

## Type checking

Not enforced today, but type annotations are encouraged on new code.
`mypy --ignore-missing-imports blockference` is the recommended local
sanity check.

## Notebook hygiene

* Strip outputs before commit when they exceed ~100 KB.
  `nbstripout` is the easiest tool:
  ```bash
  pipx install nbstripout
  nbstripout --install                # one-time, project-local
  ```
* Never commit `<<<<<<< HEAD` markers in notebooks. The previous merge
  conflict in `notebooks/simple_gridworld/multi_agent_experimental.ipynb`
  is a case study.

## Versioning

We follow loose [SemVer](https://semver.org/):

* `0.x` series — minor versions can break public surface; document in PR.
* When stable, bump to `1.0.0`. Breaking changes will then require a
  major bump and at least one minor with a `DeprecationWarning`.

`__version__` lives in `blockference/__init__.py` and is mirrored in
`pyproject.toml`.

## Releasing (future)

Not automated yet. The intended flow:

```bash
# 1. bump versions
# 2. tag
git tag -a v0.x.y -m "release: 0.x.y"
git push --tags
# 3. build + upload
python -m build
twine upload dist/*
```
