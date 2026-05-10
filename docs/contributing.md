# Contributing

Welcome — this project is steered by the
[Active Inference Institute](https://activeinference.org/) and welcomes
contributions from researchers, engineers, and AI agents alike.

## TL;DR

1. Open or comment on an issue describing what you want to do.
2. Fork → branch → code → test → PR.
3. Run `pytest` and `ruff check blockference tests` before pushing.
4. Keep PRs scoped: one logical change per PR.

## Detailed flow

### 1. Set up (uv)

```bash
git clone https://github.com/<you>/ActiveBlockference.git
cd ActiveBlockference
uv venv --python 3.11
uv pip install -e ".[dev,docs]"
uv run pytest                   # baseline; should be green
```

### 2. Branch

```bash
git checkout -b feat/<short-description>
# or fix/<...>, docs/<...>, refactor/<...>
```

### 3. Write code + tests together

* Library code lives in `blockference/`.
* Tests live in `tests/` and mirror the package layout.
* Notebooks in `notebooks/` are for exploration; promote stable code into
  the package.

### 4. Format and lint

```bash
uv run ruff format blockference tests
uv run ruff check  blockference tests
```

### 5. Document

If you change a public surface, update:

* The relevant module docstring.
* `docs/api.md`.
* `docs/architecture.md` if you change layering.
* The folder-local `README.md` table.

### 6. Open the PR

Fill in the PR template:

* **What changed** — code summary.
* **Why** — motivating issue or use case.
* **How verified** — pytest output, screenshots if visual.

A reviewer will look at correctness, scope, and doc/test parity.

## Folder ownership

See the table in the root [`AGENTS.md`](../AGENTS.md). Each folder also
ships its own `AGENTS.md` with local rules.

## Reporting issues

Open at
<https://github.com/ActiveInferenceInstitute/ActiveBlockference/issues>
with:

* Reproducer (minimal Python snippet or notebook cell).
* Expected vs. observed behaviour.
* Versions (`python -V`, `pip show inferactively-pymdp cadCAD radcad`).

## Code of conduct

Be kind. Default to assuming good faith. This community values rigour
*and* warmth — both are required.
