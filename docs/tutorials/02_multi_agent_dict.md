# 02 — Many agents, dict-registered

**Prerequisites:**
[`01_single_agent.md`](01_single_agent.md) and a basic feel for cadCAD's
"policies + state-update functions" mental model.

You will simulate `N` Active Inference agents on a shared grid. Each
agent has its own random preferred location.

## 1. Use the bundled experiment

```python
from blockference.simulations import run_grid

df = run_grid(dimension=3, no_agents=4, no_timesteps=15, output_path="out.csv")
print(df.tail())
```

The DataFrame contains, per timestep × substep × run:

* `agents`       — `dict[int, ActiveGridference]`
* `priors`       — `dict[int, np.ndarray]`
* `env_states`   — `dict[int, tuple[int, int]]`
* `actions`      — `dict[int, int]`
* `inferences`   — `dict[int, np.ndarray]`

## 2. Build the same experiment by hand

Useful when you want to customise initial conditions:

```python
import itertools
from radcad import Model, Simulation
from blockference.gridference import ActiveGridference
from blockference.utils.policy import p_actinf_dict

grid    = list(itertools.product(range(3), repeat=2))
agents  = {i: ActiveGridference(grid) for i in range(2)}
for i, a in agents.items():
    a.get_C((2, 2))
    a.get_D((0, 0))

initial_state = {
    "agents": agents,
    "priors":     {i: a.D            for i, a in agents.items()},
    "env_states": {i: a.env_state    for i, a in agents.items()},
    "actions":    {i: ""             for i, a in agents.items()},
    "inferences": {i: ""             for i, a in agents.items()},
}

def policy(params, substep, state_history, previous_state):
    return p_actinf_dict(params, substep, state_history, previous_state, grid)

# state-update functions omitted for brevity; see grid_sim.py for the canonical pattern.
```

## What to try next

* Plot the per-agent trajectory using
  `blockference.utils.utils.plot_point_on_grid`.
* Compute mutual information between successive `actions` columns:
  ```bash
  python -m blockference.utils.mutual_info out.csv --target actions
  ```
* Move on to [`03_cadcad_pipeline.md`](03_cadcad_pipeline.md) for full
  cadCAD parameter sweeps.
