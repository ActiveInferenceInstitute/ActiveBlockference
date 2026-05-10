# `tests/` — pytest test suite

The test layout mirrors the package layout:

```
tests/
├── conftest.py              — shared fixtures (RNG seed, standard grids)
├── test_package.py          — public-surface sanity tests
├── test_agent.py            — BlockferenceAgent extension tests
├── test_gridference.py      — generative-model + per-step tests
├── test_simulations.py      — cadCAD/radCAD smoke tests
├── envs/
│   └── test_grid_env_multi.py
└── utils/
    ├── test_utils_math.py
    └── test_mutual_info.py
```

## Running

```bash
uv run pytest                                         # full suite
uv run pytest tests/test_gridference.py               # one file
uv run pytest -k "not simulation"                     # skip cadCAD smoke tests
uv run pytest --cov=blockference --cov-report=term    # with coverage
```

## Conventions

* One file per source module (mirror `blockference/...` → `tests/...`).
* Per-test runtime budget: <2 s (smoke tests <10 s).
* Use `pytest.fixture` for shared setup; `_seed_rng` (in `conftest.py`)
  pre-seeds NumPy + `random` for reproducibility.
* Heavy or environment-dependent imports go behind
  `pytest.importorskip(...)` so the suite degrades gracefully.

## Adding a test

1. Find or create the matching file under `tests/` (mirror the source
   path).
2. Write the smallest test that proves the behaviour you care about.
3. Run `pytest -k <name>` until it passes.
4. Add it to the table above if it covers a new source module.

See [`AGENTS.md`](AGENTS.md) for editing rules.
