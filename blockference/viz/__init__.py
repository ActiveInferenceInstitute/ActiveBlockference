"""Visualisation helpers for ActiveBlockference trajectories.

All renderers accept a cadCAD/radCAD result :class:`pandas.DataFrame`
(or a previously saved ``trajectory.csv``) and emit static PNGs or
animated GIFs into a :class:`blockference.io.RunPaths` directory.
"""

from blockference.viz.animation import animate_trajectory
from blockference.viz.plots import (
    plot_action_distribution,
    plot_belief_heatmap,
    plot_efe,
    plot_trajectory,
)

__all__ = [
    "animate_trajectory",
    "plot_action_distribution",
    "plot_belief_heatmap",
    "plot_efe",
    "plot_trajectory",
]
