"""Static renderers for validated ActiveBlockference trajectories."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from blockference.actions import DEFAULT_AFFORDANCES, validate_affordances
from blockference.viz._common import (
    atomic_savefig,
    extract_action_history,
    extract_agent_positions,
    extract_belief_history,
    extract_efe_history,
    infer_grid_dimension,
)

__all__ = [
    "plot_action_distribution",
    "plot_belief_heatmap",
    "plot_efe",
    "plot_trajectory",
]


def _target(out_path: str | Path, suffix: str = ".png") -> Path:
    path = Path(out_path)
    if path.suffix.lower() != suffix:
        raise ValueError(f"output must use the {suffix} extension")
    return path


def _grid_size(df: pd.DataFrame, grid_dim: int | None) -> int:
    inferred = infer_grid_dimension(df)
    if grid_dim is None:
        return inferred
    if isinstance(grid_dim, bool) or not isinstance(grid_dim, int) or grid_dim < 2:
        raise ValueError("grid_dim must be an integer >= 2")
    if grid_dim != inferred:
        raise ValueError(f"grid_dim {grid_dim} does not match trajectory dimension {inferred}")
    return grid_dim


def plot_trajectory(df: pd.DataFrame, out_path: str | Path, *, title: str = "Agent trajectories", grid_dim: int | None = None) -> Path:
    """Plot every agent path on a validated square grid."""

    path = _target(out_path)
    positions = extract_agent_positions(df)
    n = _grid_size(df, grid_dim)
    fig, ax = plt.subplots(figsize=(6, 6))
    try:
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_aspect("equal")
        ax.set_title(title)
        cmap = plt.get_cmap("tab10")
        for index, (agent_id, trajectory) in enumerate(positions.items()):
            ys, xs = zip(*trajectory, strict=True)
            ax.plot(xs, ys, "-o", color=cmap(index % 10), label=f"agent {agent_id}", alpha=0.8)
            ax.scatter([xs[0]], [ys[0]], c="green", s=80, marker="s", zorder=5)
            ax.scatter([xs[-1]], [ys[-1]], c="red", s=80, marker="*", zorder=5)
        ax.legend(loc="best")
        fig.tight_layout()
        atomic_savefig(fig, path, dpi=120)
    finally:
        plt.close(fig)
    return path


def plot_belief_heatmap(df: pd.DataFrame, out_path: str | Path, *, agent_id: object | None = None, timestep: int = -1, grid_dim: int | None = None, title: str | None = None) -> Path:
    """Render one persisted posterior vector as a grid heatmap."""

    path = _target(out_path)
    beliefs = extract_belief_history(df)
    if not beliefs:
        raise ValueError("trajectory contains no posterior vectors")
    selected = next(iter(beliefs)) if agent_id is None else agent_id
    if selected not in beliefs or not beliefs[selected]:
        raise ValueError(f"agent {selected!r} has no posterior history")
    vector = beliefs[selected][timestep]
    n = _grid_size(df, grid_dim)
    if vector.size != n * n:
        raise ValueError("posterior vector size does not match the grid")
    fig, ax = plt.subplots(figsize=(5, 5))
    try:
        image = ax.imshow(vector.reshape(n, n), cmap="viridis", vmin=0.0, vmax=max(float(vector.max()), 1.0))
        ax.set_title(title or f"q(s) for agent {selected} @ step {timestep}")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        atomic_savefig(fig, path, dpi=120)
    finally:
        plt.close(fig)
    return path


def plot_action_distribution(df: pd.DataFrame, out_path: str | Path, *, affordances=None, title: str = "Action frequency per agent") -> Path:
    """Plot action frequencies using the configured action vocabulary."""

    path = _target(out_path)
    labels = validate_affordances(DEFAULT_AFFORDANCES if affordances is None else affordances)
    actions = extract_action_history(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    try:
        width = 0.8 / max(len(actions), 1)
        cmap = plt.get_cmap("tab10")
        for index, (agent_id, history) in enumerate(actions.items()):
            integers = [int(value) for value in history if isinstance(value, (int, np.integer))]
            counts = Counter(integers)
            offsets = np.arange(len(labels)) + index * width
            ax.bar(offsets, [counts.get(action, 0) for action in range(len(labels))], width=width, label=f"agent {agent_id}", color=cmap(index % 10))
        ax.set_xticks(np.arange(len(labels)) + (len(actions) - 1) * width / 2)
        ax.set_xticklabels(labels)
        ax.set_ylabel("count")
        ax.set_title(title)
        ax.legend(loc="best")
        fig.tight_layout()
        atomic_savefig(fig, path, dpi=120)
    finally:
        plt.close(fig)
    return path


def plot_efe(df: pd.DataFrame, out_path: str | Path, *, title: str = "Expected free energy over time") -> Path:
    """Plot the minimum persisted EFE value for each agent and timestep."""

    path = _target(out_path)
    efe_history = extract_efe_history(df)
    if not efe_history:
        raise ValueError("trajectory contains no persisted EFE vectors")
    fig, ax = plt.subplots(figsize=(7, 4))
    try:
        cmap = plt.get_cmap("tab10")
        for index, (agent_id, history) in enumerate(efe_history.items()):
            values = [float(vector.min()) for vector in history]
            ax.plot(range(len(values)), values, "-o", color=cmap(index % 10), label=f"agent {agent_id}")
        ax.set_xlabel("step")
        ax.set_ylabel("min G(π) (nats)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        atomic_savefig(fig, path, dpi=120)
    finally:
        plt.close(fig)
    return path
