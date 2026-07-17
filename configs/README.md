# Configurations

YAML and TOML files are loaded by `blockference.config.load_experiment_config`.
The accepted keys are `name`, `seed`, `engine`, `grid`, `simulation`, and
`output`; each nested section also has a closed key set.

`grid.affordances` is an ordered subset of `UP`, `DOWN`, `LEFT`, `RIGHT`, and
`STAY`. `target` is `random` or a bounded `[y, x]` coordinate. `initial_state`
is always a bounded `[y, x]` coordinate. Use `configs/smoke.yml` for a small
locked-backend run and `configs/sweep.toml` for a TOML round-trip example.
