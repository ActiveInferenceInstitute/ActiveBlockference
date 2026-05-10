# AGENTS.md — `blockference/simulations/`

Runnable experiments. Heavy enough to warrant CLI wiring; light enough to
keep import-time pure.

## Hard rules

* Module import must be **silent** — no prints, plots, file writes.
* Every experiment exposes a `run_*` function returning a `pandas.DataFrame`.
* `__main__` guard parses `sys.argv` and forwards to `run_*`.
* Default output paths are relative (`result.csv`) but always overridable
  via the function signature.

## Why

Tests need to import these modules without triggering simulations; CI
needs to invoke them with tiny grid sizes; users may want to script
parameter sweeps.

## Adding an experiment

* New file → new `run_*` function → CLI guard → register in
  `__init__.py` → add a smoke test that runs in <2 seconds.

## Don't

* Don't reach into `blockference.envs` to *modify* environments — compose,
  don't mutate.
* Don't import from `notebooks/` or copy notebook code verbatim. Promote
  it into proper modules first.
