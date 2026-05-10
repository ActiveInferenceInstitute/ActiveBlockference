# AGENTS.md — `blockference/config/`

This subpackage owns the **schema** for experiment configs.

## Hard rules

* Adding a top-level config key requires:
  1. A new field on the relevant dataclass with a sensible default.
  2. Validation in `__post_init__` (raise `ValueError` with a clear message).
  3. A test in `tests/test_config.py`.
  4. An update to `configs/README.md` (user-facing schema).
  5. An update to `configs/example.yml` showing the new key.

* Never silently ignore unknown keys. The loader currently uses
  ``GridConfig(**grid_kwargs)`` etc., which raises `TypeError` on
  unknown keys — keep it that way.

## Soft rules

* Defaults should yield a runnable smoke experiment (`dimension=3`,
  `n_agents=1`, `timesteps=10`).
* Prefer Python primitives (`int`, `str`, `tuple`) over custom types
  unless coercion is needed.

## Things to avoid

* Pulling YAML/TOML content into `__init__.py` at import time. The
  loader is the only entry point.
* Embedding `numpy.ndarray` or other non-serialisable types in the
  schema. Configs must be diffable text.
