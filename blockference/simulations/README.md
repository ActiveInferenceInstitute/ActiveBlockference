# Simulations

Use `run_experiment(config)` for a configured radCAD/cadCAD run or
`run_grid(dimension, n_agents, timesteps, ...)` for typed programmatic setup.
Both routes use the same configured affordances, canonical transitions, an
experiment-local seeded random stream, and diagnostic schema. `initial_state`
is broadcast to all agents for compatibility; pass `initial_states` when
distinct multi-agent starts are required. Independent child streams mean that
adding later configured runs cannot change earlier trajectories. The default
`grid.max_policies` guard keeps exponential policy enumeration bounded.
