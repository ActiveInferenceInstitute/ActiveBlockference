# AGENTS.md — `blockference/` package

Library code only. Notebook code, ad-hoc scripts, and research
explorations belong elsewhere.

## Stability contract

* Anything importable from `blockference.<module>` is **public** unless its
  name begins with `_`.
* Public symbols may not be removed or renamed without a deprecation cycle:
  add a thin alias for at least one minor version and warn via
  `warnings.warn(..., DeprecationWarning, stacklevel=2)`.
* New public symbols must be added to the corresponding subpackage
  `__init__.py`'s `__all__`.

## File-by-file rules

### `agent.py`
* Subclasses `pymdp.agent.Agent` (pymdp 1.x). Do **not** re-shadow the
  parent name inside the file (a previous bug was `class Agent(Agent)`
  with both pointing at the same identifier — instant infinite-recursion
  hazard).
* All extra constructor arguments must be forwarded with `**kwargs`.
* The 1.x ``Agent`` expects per-modality lists of JAX arrays for ``A``
  and ``B``; the grid pipeline does **not** use it (we run the math
  directly via :mod:`blockference.maths` to keep cadCAD state in NumPy).

### `maths.py`
* Pure NumPy. **Never** import JAX, pymdp, or any other framework here.
* Every function must be deterministic given a seeded
  ``numpy.random.seed`` (see ``sample`` for the contract).
* New primitives must be unit-tested in ``tests/test_maths.py``.

### `gridference.py`
* Owns the `_move(action_id, env_state, border)` private helper. Every
  per-step / per-policy function in the package must use it; do not
  re-inline the if/elif ladder elsewhere.
* `ActiveGridference.border` must be an `int` so `range(border)` works.

### `envs/`
* Environment classes are responsible for **state transitions in the world**,
  not for inference. Keep `pymdp` imports at module top.
* Collision logic must be deterministic and unit-tested.

### `simulations/`
* Each file should expose a `run_*` function and a `__main__` guard that
  forwards `sys.argv`. No top-level side effects (`print`, file IO) at
  import time.

### `utils/`
* `utils.py` is a thin wrapper over pymdp math; do not duplicate pymdp
  internals — re-export instead.
* `policy.py` houses cadCAD policy functions. Each one must accept the
  cadCAD signature exactly (`params, substep, state_history,
  previous_state, ...`).
* `mutual_info.py` must keep CLI side-effects inside `if __name__ ==
  "__main__":`. Never run `parser.parse_args()` at module scope.

## Imports

* Use absolute imports rooted at `blockference.*`.
* Do **not** add `sys.path.insert(...)` to library code. (Notebooks may.)

## Tests

Add a unit test in `tests/` for every new public function/class. Smoke
tests for cadCAD experiments live under `tests/test_simulations.py`.
