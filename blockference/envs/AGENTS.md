# AGENTS.md — `blockference/envs/`

Environments encode **world dynamics**, not agent inference.

## Hard rules

* **No** generative-model construction (A/B/C/D/E) inside this folder —
  that belongs to `gridference.py` / future agent-side modules.
* **No** policy or expected-free-energy logic here. Environments are
  passive responders to actions.
* `step(actions)` must be pure with respect to its arguments: the only
  hidden state mutated is `self.current_state` / `self.current_obs`. No
  printing, no plotting.

## Coordinate system

* `(y, x)` everywhere, matching `numpy` row-major.
* Boundaries are *clamped* (no wrap-around) — that is the existing
  contract and notebooks rely on it.

## Collisions

* If two agents would land on the same cell, the *moving* agent stays
  put. Document the policy in the docstring of `step`.

## Adding new environments

* New file, new class, register in `__init__.py`, add a `tests/envs/test_*.py`.
* Add a markdown row to `README.md`.

## Things to avoid

* Mutable default arguments (`agents=[]`) — use `None` and initialise
  inside the body.
* `print(...)` for diagnostics — prefer `logging.getLogger(__name__)`.
