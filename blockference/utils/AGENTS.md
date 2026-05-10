# AGENTS.md — `blockference/utils/`

## File-by-file rules

### `utils.py`
* Numerics first: every function should be deterministic, type-annotated,
  and tested.
* Plotting helpers stay in the same file but must be import-light: defer
  `matplotlib`/`seaborn` imports to function bodies if they are heavy.
* When you need new pymdp math, prefer **importing** from `pymdp.maths`
  rather than copying.

### `policy.py`
* cadCAD policy functions only. Each must accept the canonical signature
  exactly: `(params, substep, state_history, previous_state, ...)`.
* Reuse `_move` from `gridference.py` — never reimplement the action
  if/elif ladder.
* No file IO, no plotting, no print.

### `mutual_info.py`
* `argparse` lives inside `_cli()` — never at module scope.
* The library entry point is `calculate_mi(X, y)`. The CLI calls it.
* Don't add new CLI flags without documenting them in `README.md`.

## Things to avoid

* Recreating cadCAD machinery inside utility functions.
* Mutating inputs — return new arrays/lists.
* Adding a fourth concern to this folder without splitting it into a
  proper subpackage.
