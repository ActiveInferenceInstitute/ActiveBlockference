# AGENTS.md — `configs/`

## Hard rules

* **Schema source of truth** is
  `blockference.config.experiment.ExperimentConfig`. If you add a key
  here, also add it there (with a default + validation) and write a
  test in `tests/test_config.py`.
* **No secrets** in config files. They are committed to git.
* **Validate before commit**: every committed config must round-trip
  through `load_experiment_config` without error.

## Soft rules

* Prefer YAML for human-edited configs, TOML for declaration-style
  configs (sweeps, grids of parameters).
* Keep filenames short and lowercase: `smoke.yml`, `sweep.toml`,
  `paper_fig3.yml`.
* Comment non-obvious values inline.

## Adding a config

1. Copy `example.yml`.
2. Edit values; keep keys sorted under each section.
3. Run `uv run blockference pipeline --config configs/<your>.yml`
   and confirm a CSV is produced.
4. Add a row to `README.md`.

## Things to avoid

* Putting cadCAD-specific kwargs (e.g. `engine.deepcopy`) into config
  files yet — the loader is intentionally narrow. Promote into the
  schema first.
* Relative paths to data outside the repo. Use `results/` (gitignored).
