# 03 — A full cadCAD experiment with a parameter sweep

**Prerequisites:** [`02_multi_agent_dict.md`](02_multi_agent_dict.md).

You will sweep over (a) grid dimension and (b) number of agents to see
how time-to-target scales.

## 1. Wrap `run_grid` in a sweep

```python
import pandas as pd
from blockference.simulations import run_grid

records = []
for dim in (3, 4, 5):
    for n in (1, 2, 4):
        df = run_grid(
            dimension=dim,
            no_agents=n,
            no_timesteps=20,
            output_path=f"sweep_{dim}_{n}.csv",
        )
        records.append({"dim": dim, "n": n, "rows": len(df)})

summary = pd.DataFrame(records)
print(summary)
```

## 2. Where to plug in cadCAD-native sweeps

The `run_grid` helper is intentionally simple. For first-class cadCAD
sweeps, build the `Simulation(model=..., timesteps=..., runs=K)` object
directly with `params` containing `list[Any]` values; cadCAD will expand
each list into a sweep dimension.

```python
params = {
    "preferred_state": [grid],
    "initial_state":   [grid],
    "noise":           [0.0, 0.05, 0.10],   # cadCAD will sweep this
}
```

See `blockference/simulations/grid_sim.py` for the wiring pattern; copy
it and adjust the `params` dict.

## What to try next

* Add a third sweep axis (`planning_length`).
* Persist results to Parquet for faster reload:
  `df.to_parquet("sweep.parquet")`.
* Build a small dashboard (e.g. with Plotly) over the swept frames.
