# Implementation architecture {#sec:implementation}

## Package boundaries

The package is divided into small contracts:

- `blockference.actions` defines the action vocabulary and affordance checks.
- `blockference.gridference` owns the generative model and canonical movement.
- `blockference.envs.grid_world` owns mutable positions and simultaneous
  collision resolution.
- `blockference.maths` and `blockference.utils.utils` implement checked NumPy
  primitives.
- `blockference.simulations.grid_sim` adapts the same state updates to radCAD
  and cadCAD.
- `blockference.io` serializes and validates the run tree.
- `blockference.viz` consumes persisted trajectory values and renders figures.
- `blockference.pipeline` composes the stages and returns `PipelineResult`.

The dependency direction is deliberate: rendering reads validated values and
does not redefine model semantics; persistence serializes already computed
values and does not invent defaults; validation parses artefacts rather than
trusting the process that created them.

## Configuration

Configuration is a typed tree with explicit keys at every level. A valid
configuration names an engine (`radcad` or `cadcad`), chooses a grid dimension
and planning horizon, supplies coordinates within the grid, and selects an
ordered affordance vocabulary. YAML and TOML loaders both pass through the
same dataclass constructor, so file format does not change validation.

The output boundary rejects unsafe run names and requires callers to prepare a
complete `RunPaths` directory tree before persistence. Direct simulation CSVs
are written with `index=False`, which makes the CSV a real row-wise artefact
rather than a DataFrame debugging dump.

The simulation boundary accepts a legacy `initial_state` coordinate and
broadcasts it to all agents. A caller can instead provide
`simulation.initial_states` with one coordinate per agent. Random targets and
sampled actions use one generator owned by the run; collision correction is
applied before the next prior is persisted.

## Persistence contract

One canonical run has these required files:

```text
run/
|-- config.yml
|-- run.log
|-- manifest.json
|-- validation_report.json
|-- data/
|   |-- trajectory.csv
|   |-- summary.json
|   |-- generative_model.json
|   |-- generative_model.npz
|   |-- policies.json
|   `-- per_step.csv
|-- viz/
|   |-- trajectory.png
|   |-- action_distribution.png
|   |-- efe.png
|   `-- belief_heatmap_*.png
`-- animations/trajectory.gif
```

Nested values in CSV files are encoded as JSON-compatible values. A reader can
recover coordinates, mappings, arrays, and scalar fields without relying on
Python object representations. Parquet is an optional convenience and never
substitutes for the required CSV and JSON artefacts.

## Backend equivalence

cadCAD provides a general Python framework for designing and validating
complex systems [@cadCAD2024]. radCAD supplies a compatible local simulation
path [@radCAD2024]. ActiveBlockference constructs one state-update contract and
drives either backend from it. The release tests compare seeded schemas and
row-index fields across both paths; they do not treat backend availability as a
reason to omit required coverage.
