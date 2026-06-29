"""Static plots for ActiveBlockference trajectories."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402  — headless rendering for CI / scripts

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from blockference.viz._common import (  # noqa: E402
    extract_action_history,
    extract_agent_positions,
    extract_belief_history,
    infer_grid_dimension,
)

__all__ = [
    "plot_action_distribution",
    "plot_belief_heatmap",
    "plot_efe_proxy",
    "plot_trajectory",
]


def plot_trajectory(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    title: str = "Agent trajectories",
    grid_dim: int | None = None,
) -> Path:
    """Plot every agent's path on a square grid, save as PNG."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    positions = extract_agent_positions(df)
    n = grid_dim or infer_grid_dimension(df)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)  # invert so (0,0) is top-left like a grid
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_aspect("equal")
    ax.set_title(title)

    cmap = plt.get_cmap("tab10")
    for i, (agent_id, traj) in enumerate(positions.items()):
        if not traj:
            continue
        ys, xs = zip(*traj, strict=False)
        ax.plot(xs, ys, "-o", color=cmap(i % 10), label=f"agent {agent_id}", alpha=0.8)
        ax.scatter([xs[0]], [ys[0]], c="green", s=80, marker="s", zorder=5)
        ax.scatter([xs[-1]], [ys[-1]], c="red", s=80, marker="*", zorder=5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_belief_heatmap(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    agent_id: object | None = None,
    timestep: int = -1,
    grid_dim: int | None = None,
    title: str | None = None,
) -> Path:
    """Render the belief vector at ``timestep`` as an n×n heatmap."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    beliefs = extract_belief_history(df)
    if not beliefs:
        raise ValueError("no belief history found in trajectory frame")
    if agent_id is None:
        agent_id = next(iter(beliefs))
    history = beliefs[agent_id]
    if not history:
        raise ValueError(f"agent {agent_id!r} has no belief history")
    qs_flat = np.asarray(history[timestep])
    # Prefer the belief vector's own size (always n*n for square grids); fall
    # back to data-driven inference if the caller passed something exotic.
    n = grid_dim or int(round(qs_flat.size**0.5))
    if n * n != qs_flat.size:
        n = grid_dim or infer_grid_dimension(df)
    qs = qs_flat.reshape(n, n)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(qs, cmap="viridis", vmin=0.0, vmax=qs.max())
    ax.set_title(title or f"q(s) for agent {agent_id} @ step {timestep}")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_action_distribution(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    affordances: list[str] | None = None,
    title: str = "Action frequency per agent",
) -> Path:
    """Bar-chart action frequencies, one bar group per agent."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    actions = extract_action_history(df)
    affordances = affordances or ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
    n_actions = len(affordances)
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.8 / max(len(actions), 1)
    cmap = plt.get_cmap("tab10")
    for i, (agent_id, history) in enumerate(actions.items()):
        # Discard cadCAD's initial empty-string action at t=0.
        ints = [
            int(a)
            for a in history
            if isinstance(a, (int, np.integer)) or (isinstance(a, str) and a.lstrip("-").isdigit())
        ]
        counts = Counter(ints)
        bar_y = [counts.get(a, 0) for a in range(n_actions)]
        offsets = np.arange(n_actions) + i * width
        ax.bar(offsets, bar_y, width=width, label=f"agent {agent_id}", color=cmap(i % 10))
    ax.set_xticks(np.arange(n_actions) + (len(actions) - 1) * width / 2)
    ax.set_xticklabels(affordances)
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_efe_proxy(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    title: str = "Belief entropy over time (EFE proxy)",
) -> Path:
    """Plot per-agent posterior entropy at every step.

    True expected-free-energy traces are not stored on cadCAD frames by
    default; we plot the entropy of the inferred posterior ``q(s)`` as
    a model-free proxy that decreases as the agent becomes confident.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    beliefs = extract_belief_history(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.get_cmap("tab10")
    for i, (agent_id, history) in enumerate(beliefs.items()):
        if not history:
            continue
        ents = [-_safe_xlogx(np.asarray(qs)).sum() for qs in history]
        ax.plot(range(len(ents)), ents, "-o", color=cmap(i % 10), label=f"agent {agent_id}")
    ax.set_xlabel("step")
    ax.set_ylabel("H[q(s)]  (nats)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _safe_xlogx(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-16, 1.0)
    return p * np.log(p)
