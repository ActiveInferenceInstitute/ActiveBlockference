"""ActiveBlockference — Active Inference agents in cadCAD.

Public surface re-exports the most useful classes/functions so callers can
do ``from blockference import ActiveGridference``.
"""

from blockference.agent import BlockferenceAgent
from blockference.config import ExperimentConfig, load_experiment_config
from blockference.gridference import (
    ActiveGridference,
    actinf_graph,
    actinf_planning_single,
    make_grid,
)
from blockference.io import (
    RunPaths,
    ValidationReport,
    build_run_paths,
    validate_generative_model,
    validate_per_step_records,
    validate_run_outputs,
    validate_trajectory_dataframe,
)
from blockference.pipeline import (
    PipelineResult,
    persist,
    render_visualisations,
    run_pipeline,
    simulate,
    validate,
)

__version__ = "0.5.0"

__all__ = [
    "ActiveGridference",
    "BlockferenceAgent",
    "ExperimentConfig",
    "PipelineResult",
    "RunPaths",
    "ValidationReport",
    "actinf_graph",
    "actinf_planning_single",
    "build_run_paths",
    "load_experiment_config",
    "make_grid",
    "persist",
    "render_visualisations",
    "run_pipeline",
    "simulate",
    "validate",
    "validate_generative_model",
    "validate_per_step_records",
    "validate_run_outputs",
    "validate_trajectory_dataframe",
    "__version__",
]
