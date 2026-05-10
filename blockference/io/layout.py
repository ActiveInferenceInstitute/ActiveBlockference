"""Canonical output directory layout for ActiveBlockference runs.

Every experiment produces an artefact tree of the form::

    <root>/<run_name>/
        config.yml
        data/
            trajectory.csv
            trajectory.parquet           # optional, written when pyarrow is present
            summary.json
        viz/
            trajectory.png
            efe_trace.png
            belief_heatmap_t<###>.png
        animations/
            trajectory.gif
        validation_report.json

The :class:`RunPaths` dataclass exposes typed accessors for each of these
locations and creates the directories on demand.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

__all__ = ["RunPaths", "build_run_paths"]


@dataclass(frozen=True)
class RunPaths:
    """Canonical output directory tree for a single experiment run."""

    root: Path
    run_name: str

    @property
    def run_dir(self) -> Path:
        return self.root / self.run_name

    @property
    def data_dir(self) -> Path:
        return self.run_dir / "data"

    @property
    def viz_dir(self) -> Path:
        return self.run_dir / "viz"

    @property
    def animations_dir(self) -> Path:
        return self.run_dir / "animations"

    @property
    def config_path(self) -> Path:
        return self.run_dir / "config.yml"

    @property
    def trajectory_csv(self) -> Path:
        return self.data_dir / "trajectory.csv"

    @property
    def trajectory_parquet(self) -> Path:
        return self.data_dir / "trajectory.parquet"

    @property
    def summary_json(self) -> Path:
        return self.data_dir / "summary.json"

    @property
    def validation_report(self) -> Path:
        return self.run_dir / "validation_report.json"

    @property
    def run_log(self) -> Path:
        return self.run_dir / "run.log"

    @property
    def matrices_json(self) -> Path:
        return self.data_dir / "generative_model.json"

    @property
    def matrices_npz(self) -> Path:
        return self.data_dir / "generative_model.npz"

    @property
    def per_step_csv(self) -> Path:
        return self.data_dir / "per_step.csv"

    @property
    def per_step_parquet(self) -> Path:
        return self.data_dir / "per_step.parquet"

    @property
    def policies_json(self) -> Path:
        return self.data_dir / "policies.json"

    def ensure(self) -> RunPaths:
        """Create every directory in the layout. Idempotent."""
        for d in (self.run_dir, self.data_dir, self.viz_dir, self.animations_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self


def build_run_paths(
    root: str | Path = "output",
    run_name: str | None = None,
    *,
    timestamped: bool = False,
) -> RunPaths:
    """Construct a :class:`RunPaths` tree.

    Parameters
    ----------
    root :
        The top-level output directory. Defaults to ``output/`` so all
        artefacts collect under one well-known location.
    run_name :
        Sub-directory under ``root``. Defaults to ``"run"`` if omitted.
    timestamped :
        If True, append a UTC timestamp suffix to make the directory unique.
    """
    name = run_name or "run"
    if timestamped:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        name = f"{name}-{ts}"
    return RunPaths(root=Path(root), run_name=name).ensure()
