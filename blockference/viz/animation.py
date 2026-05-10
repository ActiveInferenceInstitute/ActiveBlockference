"""Animated trajectories — GIF (and optionally MP4) rendering."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from blockference.viz._common import extract_agent_positions, infer_grid_dimension  # noqa: E402

__all__ = ["animate_trajectory"]


def animate_trajectory(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    fps: int = 4,
    title: str = "Agent trajectories",
    grid_dim: int | None = None,
) -> Path:
    """Render a GIF of the agent paths unfolding over time.

    Parameters
    ----------
    df :
        cadCAD trajectory DataFrame.
    out_path :
        Output ``.gif`` (or ``.mp4`` if ffmpeg is installed) path.
    fps :
        Frames per second.
    title :
        Plot title.
    grid_dim :
        Grid side length; inferred from data when omitted.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    positions = extract_agent_positions(df)
    n = grid_dim or infer_grid_dimension(df)
    n_frames = max((len(traj) for traj in positions.values()), default=0)
    if n_frames == 0:
        raise ValueError("no agent trajectories to animate")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_aspect("equal")
    ax.set_title(title)

    cmap = plt.get_cmap("tab10")
    lines, dots, labels = {}, {}, {}
    for i, agent_id in enumerate(positions):
        color = cmap(i % 10)
        (line,) = ax.plot([], [], "-", color=color, alpha=0.6, label=f"agent {agent_id}")
        (dot,) = ax.plot([], [], "o", color=color, markersize=10)
        lines[agent_id] = line
        dots[agent_id] = dot
        labels[agent_id] = ax.text(0, 0, "", color=color, fontsize=8)
    ax.legend(loc="upper right")
    step_text = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=10)

    def init():
        for line in lines.values():
            line.set_data([], [])
        for dot in dots.values():
            dot.set_data([], [])
        step_text.set_text("")
        return list(lines.values()) + list(dots.values()) + [step_text]

    def update(frame: int):
        for agent_id, traj in positions.items():
            up_to = traj[: frame + 1]
            if up_to:
                ys, xs = zip(*up_to)
                lines[agent_id].set_data(xs, ys)
                dots[agent_id].set_data([xs[-1]], [ys[-1]])
                labels[agent_id].set_position((xs[-1] + 0.05, ys[-1] - 0.1))
                labels[agent_id].set_text(str(agent_id))
        step_text.set_text(f"step {frame}")
        return list(lines.values()) + list(dots.values()) + [step_text]

    anim = FuncAnimation(
        fig, update, frames=n_frames, init_func=init, blit=False, interval=1000 / fps
    )

    suffix = out_path.suffix.lower()
    if suffix == ".gif":
        writer = PillowWriter(fps=fps)
        anim.save(out_path, writer=writer)
    elif suffix == ".mp4":
        try:
            from matplotlib.animation import FFMpegWriter
            anim.save(out_path, writer=FFMpegWriter(fps=fps))
        except (ImportError, RuntimeError) as exc:  # pragma: no cover
            raise RuntimeError(
                "MP4 output requires ffmpeg; install it or use a .gif extension."
            ) from exc
    else:
        raise ValueError(f"unsupported animation extension {suffix!r}; use .gif or .mp4")

    plt.close(fig)
    return out_path
