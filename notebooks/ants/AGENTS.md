# AGENTS.md — `notebooks/ants/`

Same rules as `notebooks/AGENTS.md`, plus:

* This notebook drives toward a future `blockference.envs.AntColonyEnv`.
  When patterns repeat (food sources, pheromone trails, neighbour
  observations), promote them into proper modules + tests.
* Keep grids small (≤ 10×10) so the notebook runs end-to-end in CI-time.
