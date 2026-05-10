# `notebooks/` — exploratory and pedagogical Jupyter notebooks

Notebooks are research artefacts: they may be more experimental than the
core library, but they should still be runnable and well-described.

## Index

| Notebook                                                | Topic                                                    |
|---------------------------------------------------------|----------------------------------------------------------|
| [`simple_gridworld/agent_api_single.ipynb`](simple_gridworld/agent_api_single.ipynb)         | Single agent via the high-level pymdp `Agent` API.       |
| [`simple_gridworld/gridference_single.ipynb`](simple_gridworld/gridference_single.ipynb)     | Single agent built from primitives + cadCAD.             |
| [`simple_gridworld/modular_single.ipynb`](simple_gridworld/modular_single.ipynb)             | Modular split between `Agent` and `Grid` environment.    |
| [`simple_gridworld/multiple_agents_network.ipynb`](simple_gridworld/multiple_agents_network.ipynb) | Multi-agent on a `networkx` graph.                       |
| [`simple_gridworld/mutliple_agents_dict.ipynb`](simple_gridworld/mutliple_agents_dict.ipynb) | Multi-agent registry as a dict.                          |
| [`simple_gridworld/two_multi_agent.ipynb`](simple_gridworld/two_multi_agent.ipynb)           | Two-agent gridworld with collisions and relative pos.    |
| [`simple_gridworld/multi_agent_experimental.ipynb`](simple_gridworld/multi_agent_experimental.ipynb) | Multi-agent experimental sandbox (no cadCAD).            |
| [`simple_gridworld/Untitled.ipynb`](simple_gridworld/Untitled.ipynb)                         | Scratch notebook (TwoMultiGridAgent draft). Rename me.   |
| [`ants/blockferants.ipynb`](ants/blockferants.ipynb)                                         | Active BlockferAnts — collective foraging exploration.   |

## Running

```bash
pip install -e .[dev]                  # installs jupyterlab via dev extra
jupyter lab notebooks/
```

Each notebook starts with a markdown cell describing its purpose. If you
add a new notebook, do the same.

## Conventions

* Strip large outputs before commit (>100 KB).
* Prefer `from blockference import ...` over copy-pasting cells from the
  library.
* If you discover a stable, useful pattern, **promote** it into
  `blockference/` and add a regression test.

See [`AGENTS.md`](AGENTS.md) for editing rules.
