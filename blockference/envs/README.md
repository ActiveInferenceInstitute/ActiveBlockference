# `blockference/envs/` — environments

Holds *environment* classes: state spaces and step dynamics that the
agents act in. Inference itself lives in `blockference.gridference` and
`blockference.utils.policy`.

## Modules

| Module                  | Class                | Purpose                                    |
|-------------------------|----------------------|--------------------------------------------|
| `grid_env.py`           | `GridAgent`          | Single- or multi-agent grid, simple step.  |
| `grid_env_multi.py`     | `TwoMultiGridAgent`  | Two-agent grid with collision + relpos.    |

## Conventions

* Coordinates are `(y, x)` tuples to match `numpy` row-major convention.
* The 5-action affordance set is fixed: `["UP","DOWN","LEFT","RIGHT","STAY"]`.
  3-D extensions add `IN`, `OUT`.
* `border` is the maximum valid index (i.e. `grid_len - 1`).
* The `step` method must accept actions for *all* agents and return a list
  of new agent observations / states.

## Adding a new environment

1. Create `your_env.py` next to the existing files.
2. Implement `__init__`, `step`, and any helpers (collision, observation).
3. Add the class to `blockference/envs/__init__.py`.
4. Write tests under `tests/envs/`.
5. Add a markdown table row to this README.

See [`AGENTS.md`](AGENTS.md) for editing rules.
