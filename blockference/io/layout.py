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
        manifest.json
        run.log
        validation_report.json

The :class:`RunPaths` dataclass exposes typed accessors for each of these
locations and creates the directories on demand.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["RunPaths", "build_run_paths"]


@dataclass(frozen=True)
class RunPaths:
    """Canonical output directory tree for a single experiment run."""

    root: Path
    run_name: str

    def __post_init__(self) -> None:
        root = Path(self.root)
        if not isinstance(self.run_name, str) or not self.run_name.strip():
            raise ValueError("run_name must be a non-empty string")
        if self.run_name in {".", ".."} or Path(self.run_name).name != self.run_name or any(char in self.run_name for char in "/\\\x00"):
            raise ValueError("run_name must be a single safe path component")
        object.__setattr__(self, "root", root)

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
    def manifest_json(self) -> Path:
        return self.run_dir / "manifest.json"

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

    def ensure(self, *, allow_existing: bool = False) -> RunPaths:
        """Create every directory, refusing accidental non-empty reuse."""

        if self.run_dir.exists() and any(self.run_dir.iterdir()) and not allow_existing:
            raise FileExistsError(
                f"run directory is not empty: {self.run_dir}; "
                "choose a new run name or explicitly allow reuse"
            )
        for d in (self.run_dir, self.data_dir, self.viz_dir, self.animations_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def require_tree(self) -> RunPaths:
        """Require the complete directory tree before persistence begins."""

        missing = [str(path) for path in (self.run_dir, self.data_dir, self.viz_dir, self.animations_dir) if not path.is_dir()]
        if missing:
            raise FileNotFoundError(f"RunPaths tree is not prepared: {', '.join(missing)}")
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
    name = "run" if run_name is None else run_name
    if not isinstance(name, str) or not name.strip():
        raise ValueError("run_name must be a non-empty string")
    if name in {".", ".."} or Path(name).name != name or any(char in name for char in "/\\\x00"):
        raise ValueError("run_name must be a single safe path component")
    if not all(char.isprintable() for char in name):
        raise ValueError("run_name must contain printable characters only")
    if timestamped:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = f"{name}-{ts}"
    root_path = Path(root)
    if any(part == ".." for part in root_path.parts):
        raise ValueError("output root must not contain parent-directory traversal")
    return RunPaths(root=root_path, run_name=name)
