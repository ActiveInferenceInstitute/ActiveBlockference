# `notebooks/simple_gridworld/` — gridworld notebooks

Progressive tour from one agent → many agents → graph of agents on a
small discrete grid.

## Suggested reading order

1. `agent_api_single.ipynb` — start with the high-level pymdp `Agent` API.
2. `gridference_single.ipynb` — same problem rebuilt from primitives.
3. `modular_single.ipynb` — refactor into `Agent` + `Grid` modules.
4. `mutliple_agents_dict.ipynb` — many agents, dict-based registry.
5. `multiple_agents_network.ipynb` — many agents, `networkx` graph.
6. `two_multi_agent.ipynb` — two-agent gridworld with collisions.
7. `multi_agent_experimental.ipynb` — exploratory sandbox (no cadCAD).

`Untitled.ipynb` is a scratch notebook drafting `TwoMultiGridAgent`; it
should be renamed or removed in a future cleanup pass.

See [`../AGENTS.md`](../AGENTS.md) for editing rules.
