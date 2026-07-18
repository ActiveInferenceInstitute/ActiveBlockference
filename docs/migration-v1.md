# v1 migration guide

This guide describes the current v1 contract for callers moving from the
older permissive grid helpers.

## Configuration

Continue using `simulation.initial_state` for the compatibility broadcast, or
provide one unique coordinate per agent with `simulation.initial_states`.
Unknown keys, duplicate explicit starts, invalid coordinates, and malformed
paths now fail during configuration loading. `grid.max_policies` defaults to
100,000 and prevents accidental exponential policy allocation; raise it only
when the run has an explicit resource budget.

## Randomness and backends

Pass a numeric `seed` for reproducibility. The seed is split into independent
child streams per configured run, so adding later runs does not alter earlier
trajectories. `radcad` and `cadcad` share the same transition and policy
boundary and must produce the same normalized trajectory for equivalent seeded
configurations. Callers must not rely on process-global NumPy or Python random
state.

## Outputs and verdicts

Use `run_pipeline()` for a complete run. A non-empty run directory is refused
unless reuse is explicitly requested. Successful runs contain a manifest with
hashes and sizes for stable artifacts plus a validation report. Consumers
should branch only on `PipelineResult.ok` or `ValidationReport.ok`; individual
files and scientific interpretation are not release verdicts.

## Optional integrations

The grid implementation does not require `pymdp`. Install
`active-blockference[pymdp]` for `BlockferenceAgent`; install
`active-blockference[research]` only for the explicit OpenAI GRTs provider.
Offline tests, notebooks, and release gates do not require either optional
integration.
