# `blockference`

The installable package contains the strict core model, configuration,
simulation, persistence, visualisation, and command-line interfaces.

Public entry points are `GridWorld`, `ActiveGridference`,
`BlockferenceAgent`, `ExperimentConfig`, `run_pipeline`, and `plot_efe`.
The grid core has no `pymdp` dependency; `BlockferenceAgent` is available
through the optional `active-blockference[pymdp]` extra.
