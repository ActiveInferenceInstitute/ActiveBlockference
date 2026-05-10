# `blockference/simulations/` — runnable experiments

Each module here is a **complete cadCAD/radCAD experiment**. Importing the
module is side-effect free; running it requires either calling the `run_*`
function programmatically or invoking the module via `python -m`.

## Modules

| Module                  | Function       | What it does                                       |
|-------------------------|----------------|----------------------------------------------------|
| `grid_sim.py`           | `run_grid()`   | Multi-agent gridworld (random targets, 5 actions). |

## Running

```bash
# 3x3 grid, 2 agents, 25 timesteps, written to result.csv
python -m blockference.simulations.grid_sim 3 2 25
```

Programmatic equivalent:

```python
from blockference.simulations import run_grid
df = run_grid(dimension=3, no_agents=2, no_timesteps=25, output_path="out.csv")
```

The returned DataFrame contains, per timestep and substep, the dict-valued
columns `agents`, `priors`, `env_states`, `actions`, `inferences`. See
[`docs/architecture.md`](../../docs/architecture.md) for column semantics.

## Adding an experiment

1. Create `your_sim.py` exposing `run_your_sim(**kwargs)`.
2. Add a `__main__` guard parsing `sys.argv`.
3. Register the function in `blockference/simulations/__init__.py`.
4. Add a smoke test in `tests/test_simulations.py`.
5. Add a row to the table above.

See [`AGENTS.md`](AGENTS.md) for editing rules.
