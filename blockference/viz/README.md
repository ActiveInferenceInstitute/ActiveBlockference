# `blockference/viz/` — visualisation + animation

Renders cadCAD trajectory frames into PNGs and GIFs. Pure functions —
each renderer takes a DataFrame and an output path and returns the
path it wrote.

| Module          | Public symbols                                                       |
|-----------------|-----------------------------------------------------------------------|
| `plots.py`      | `plot_trajectory`, `plot_action_distribution`, `plot_belief_heatmap`, `plot_efe_proxy` |
| `animation.py`  | `animate_trajectory` (GIF; MP4 with ffmpeg)                          |
| `_common.py`    | Internal helpers: `extract_agent_positions`, `extract_belief_history`, `infer_grid_dimension` |

## Surface

```python
from blockference.viz import (
    plot_trajectory,
    plot_action_distribution,
    plot_belief_heatmap,
    plot_efe_proxy,
    animate_trajectory,
)
```

## Examples

```python
import pandas as pd
from blockference.viz import plot_trajectory, animate_trajectory

df = pd.read_csv("output/gridworld_example/data/trajectory.csv")
plot_trajectory(df, "trajectory.png")
animate_trajectory(df, "trajectory.gif", fps=4)
```

## Conventions

* Coordinates are `(y, x)` (matches the rest of the package). Plots
  invert the y-axis so `(0,0)` reads as top-left.
* Renderers force `matplotlib.use("Agg")` so they work in CI / headless
  shells.
* All renderers create the parent directory if missing.

See [`AGENTS.md`](AGENTS.md) for editing rules.
