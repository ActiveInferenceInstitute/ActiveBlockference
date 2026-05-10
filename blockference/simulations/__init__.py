"""cadCAD/radCAD simulation entry points for ActiveBlockference.

The concrete experiments depend on ``radcad``; we import them lazily so
that ``import blockference.simulations`` always succeeds, even in slim
environments without a cadCAD-family runtime installed. Trying to actually
*use* an experiment without ``radcad`` will raise the original
``ModuleNotFoundError`` at call time.
"""

from importlib import import_module
from typing import Any

__all__ = ["run_grid", "run_experiment"]


def __getattr__(name: str) -> Any:  # pragma: no cover - thin shim
    if name in __all__:
        return getattr(import_module("blockference.simulations.grid_sim"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
