"""Fail-closed validation for in-memory and persisted run artefacts."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zipfile import BadZipFile

import numpy as np
import pandas as pd
import yaml

from blockference.config import load_experiment_config
from blockference.io.layout import RunPaths
from blockference.io.persist import _model_digest
from blockference.maths import construct_policies

REQUIRED_TRAJECTORY_COLUMNS: tuple[str, ...] = (
    "agents",
    "priors",
    "env_states",
    "actions",
    "inferences",
    "efe",
    "efe_epistemic",
    "efe_pragmatic",
    "q_pi",
    "p_u",
    "obs_idx",
    "timestep",
    "substep",
    "run",
)
REQUIRED_STEP_FIELDS: tuple[str, ...] = (
    "agent_id",
    "env_state",
    "action",
    "obs_idx",
    "posterior",
    "prior",
    "efe",
    "efe_epistemic",
    "efe_pragmatic",
    "q_pi",
    "p_u",
)
REQUIRED_VIZ: tuple[str, ...] = ("trajectory.png", "action_distribution.png", "efe.png")
MANIFEST_VERSION = 1


@dataclass
class ValidationReport:
    """Result of validating one artefact boundary."""

    ok: bool = True
    checks: dict[str, bool] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    artefacts: dict[str, str] = field(default_factory=dict)

    def add(self, name: str, ok: bool, issue: str | None = None) -> None:
        passed = bool(ok)
        self.checks[name] = passed
        if not passed:
            self.ok = False
            self.issues.append(f"{name}: {issue or 'check failed'}")

    def merge(self, prefix: str, other: ValidationReport) -> None:
        """Merge another report while retaining its check namespace."""

        for name, passed in other.checks.items():
            self.add(f"{prefix}.{name}", passed, None if passed else "failed")
        self.issues.extend(f"{prefix}: {issue}" for issue in other.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": self.checks,
            "issues": self.issues,
            "artefacts": self.artefacts,
        }

    def write(self, paths: RunPaths) -> Path:
        paths.require_tree()
        temporary = paths.validation_report.with_name(f".{paths.validation_report.name}.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, indent=2, sort_keys=True)
            stream.write("\n")
        temporary.replace(paths.validation_report)
        return paths.validation_report


def _literal(value: Any) -> Any:
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value
    return value


def _vector(value: Any, name: str, *, non_negative: bool = True) -> np.ndarray | None:
    parsed = _literal(value)
    if parsed is None:
        return None
    try:
        array = np.asarray(parsed, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        return None
    if non_negative and (array < 0).any():
        return None
    return array


def _coordinate(value: Any) -> tuple[int, int] | None:
    parsed = _literal(value)
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 2:
        return None
    if any(
        isinstance(component, bool) or not isinstance(component, (int, np.integer))
        for component in parsed
    ):
        return None
    return int(parsed[0]), int(parsed[1])


def _index(value: Any, minimum: int = 0) -> int | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(number) or not number.is_integer() or number < minimum:
        return None
    return int(number)


def _agent_id(value: Any) -> str | None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (str, int, np.integer)):
        return None
    number = _index(value)
    return str(number) if number is not None and str(value) == str(number) else None


def _row_identity(row: Any) -> tuple[int, int, int] | None:
    run = _index(row.get("run"), 1)
    time = _index(row.get("timestep"))
    substep = _index(row.get("substep"))
    if run is None or time is None or substep is None:
        return None
    return run, time, substep


def _trajectory_rows(trajectory: Any) -> dict | None:
    if (
        not isinstance(trajectory, pd.DataFrame) or trajectory.empty
        or any(column not in trajectory for column in REQUIRED_TRAJECTORY_COLUMNS)
    ):
        return None
    rows = {}
    for _, row in trajectory.iterrows():
        key = _row_identity(row)
        if key is None or key in rows:
            return None
        rows[key] = row
    return rows


def _expected_model_keys(config: Any) -> set[tuple[int, str]]:
    return {
        (run, str(agent)) for run in range(1, config.simulation.runs + 1)
        for agent in range(config.simulation.n_agents)
    }


def _expected_schedule(config: Any) -> set[tuple[int, int, int]]:
    # Both adapters run the single canonical update block and retain its initial row.
    return {
        (run, time, int(time > 0))
        for run in range(1, config.simulation.runs + 1)
        for time in range(config.simulation.timesteps + 1)
    }


def _run_records(payload: Any) -> dict[tuple[int, str], Any]:
    if not isinstance(payload, dict) or payload.get("version") != 2:
        raise ValueError("expected a version-2 run/agent artifact")
    runs = payload.get("runs")
    if not isinstance(runs, dict) or not runs:
        raise ValueError("runs must be a non-empty mapping")
    records = {}
    for run_label, agents in runs.items():
        run = _index(run_label, 1)
        if run is None or str(run) != run_label or not isinstance(agents, dict) or not agents:
            raise ValueError("invalid run registry")
        for agent_label, value in agents.items():
            agent = _agent_id(agent_label)
            if agent is None:
                raise ValueError("invalid configured agent ID")
            records[(run, agent)] = value
    return records




def validate_trajectory_dataframe(df: pd.DataFrame) -> ValidationReport:
    """Validate trajectory columns, row values, and nested state structure."""

    report = ValidationReport()
    if not isinstance(df, pd.DataFrame):
        report.add("dataframe_type", False, f"got {type(df).__name__}")
        return report
    report.add("non_empty", not df.empty, f"frame has {len(df)} rows")
    missing = [column for column in REQUIRED_TRAJECTORY_COLUMNS if column not in df.columns]
    report.add("required_columns", not missing, f"missing columns: {missing}" if missing else None)
    if df.empty or missing:
        return report

    bad_maps = 0
    bad_row_indices = 0
    inconsistent_keys = 0
    bad_coordinates = 0
    agent_keys: set[Any] = set()
    registry_drift = 0
    canonical_registry: set[Any] | None = None
    for _, row in df.iterrows():
        maps: dict[str, Any] = {}
        row_key_sets: list[set[Any]] = []
        for column in REQUIRED_TRAJECTORY_COLUMNS[:11]:
            value = _literal(row[column])
            maps[column] = value
            if not isinstance(value, dict):
                bad_maps += 1
            else:
                row_key_sets.append(set(value))
        if row_key_sets and any(keys != row_key_sets[0] for keys in row_key_sets[1:]):
            inconsistent_keys += 1
        env = maps["env_states"]
        if isinstance(env, dict):
            row_registry = set(env)
            agent_keys.update(row_registry)
            if canonical_registry is None:
                canonical_registry = row_registry
            elif row_registry != canonical_registry:
                registry_drift += 1
            for position in env.values():
                if _coordinate(position) is None:
                    bad_coordinates += 1
        for scalar in ("timestep", "substep", "run"):
            value = row[scalar]
            try:
                numeric = float(value)
                finite = bool(np.isfinite(numeric))
            except (TypeError, ValueError):
                numeric = float("nan")
                finite = False
            if isinstance(value, bool) or not finite:
                bad_maps += 1
            elif numeric < 0 or not numeric.is_integer() or (scalar == "run" and numeric < 1):
                bad_row_indices += 1
    report.add("nested_state_maps", bad_maps == 0, f"{bad_maps} malformed cells")
    report.add("row_indices", bad_row_indices == 0, f"{bad_row_indices} invalid row indices")
    report.add(
        "agent_keys_consistent",
        inconsistent_keys == 0,
        f"{inconsistent_keys} rows have inconsistent agent registries",
    )
    report.add(
        "agent_registry_stable",
        registry_drift == 0,
        f"{registry_drift} rows changed the trajectory agent registry",
    )
    report.add("coordinates", bad_coordinates == 0, f"{bad_coordinates} malformed coordinates")
    report.add("agent_registry_non_empty", bool(agent_keys), "no agent IDs found")
    try:
        finite_indices = bool(
            np.isfinite(df[["timestep", "substep", "run"]].to_numpy(dtype=float)).all()
        )
    except (TypeError, ValueError):
        finite_indices = False
    report.add("finite_row_indices", finite_indices, "row indices are not numeric")
    report.add(
        "unique_row_identities", _trajectory_rows(df) is not None,
        "run/timestep/substep identities are invalid or duplicated",
    )
    return report


def validate_per_step_records(
    records: list[dict[str, Any]], *, n_states: int | None = None,
    n_actions: int | None = None, n_policies: int | None = None,
) -> ValidationReport:
    """Validate complete per-agent diagnostics for every simulation step.

    Optional model dimensions make the checker strict when validating a
    persisted run while preserving a useful shape-only mode for standalone
    records.
    """

    report = ValidationReport()
    report.add("non_empty", bool(records), f"{len(records)} records")
    if not records:
        return report
    non_mappings = sum(1 for record in records if not isinstance(record, dict))
    missing = sum(
        1
        for record in records
        if isinstance(record, dict)
        and any(record.get(field) is None for field in REQUIRED_STEP_FIELDS)
    )
    missing += non_mappings
    report.add("required_fields", missing == 0, f"{missing} records have missing diagnostics")

    bad_coordinates = bad_actions = bad_obs = 0
    bad_posterior = bad_prior = bad_efe = bad_qpi = bad_pu = bad_decomp = 0
    bad_shapes = bad_bounds = bad_dimensions = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        if _coordinate(record.get("env_state")) is None:
            bad_coordinates += 1
        action = record.get("action")
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)) or int(action) < 0:
            bad_actions += 1
        observation = record.get("obs_idx")
        if (
            isinstance(observation, bool)
            or not isinstance(observation, (int, np.integer))
            or int(observation) < 0
        ):
            bad_obs += 1
        posterior = _vector(record.get("posterior"), "posterior")
        prior = _vector(record.get("prior"), "prior")
        efe = _vector(record.get("efe"), "efe", non_negative=False)
        epistemic = _vector(record.get("efe_epistemic"), "efe_epistemic", non_negative=False)
        pragmatic = _vector(record.get("efe_pragmatic"), "efe_pragmatic", non_negative=False)
        q_pi = _vector(record.get("q_pi"), "q_pi")
        p_u = _vector(record.get("p_u"), "p_u")
        if posterior is None or not np.isclose(posterior.sum(), 1.0, atol=1e-6):
            bad_posterior += 1
        if prior is None or not np.isclose(prior.sum(), 1.0, atol=1e-6):
            bad_prior += 1
        if efe is None or epistemic is None or pragmatic is None or not np.isfinite(efe).all():
            bad_efe += 1
        if q_pi is None or not np.isclose(q_pi.sum(), 1.0, atol=1e-6):
            bad_qpi += 1
        if p_u is None or not np.isclose(p_u.sum(), 1.0, atol=1e-6):
            bad_pu += 1
        vectors = (posterior, prior, efe, epistemic, pragmatic, q_pi, p_u)
        if any(vector is None for vector in vectors):
            bad_shapes += 1
        else:
            assert posterior is not None
            assert prior is not None
            assert efe is not None
            assert epistemic is not None
            assert pragmatic is not None
            assert q_pi is not None
            assert p_u is not None
            if (
                posterior.shape != prior.shape
                or efe.shape != epistemic.shape
                or efe.shape != pragmatic.shape
                or efe.shape != q_pi.shape
            ):
                bad_shapes += 1
            if n_states is not None and (posterior.size != n_states or prior.size != n_states):
                bad_dimensions += 1
            if n_actions is not None and p_u.size != n_actions:
                bad_dimensions += 1
            if n_policies is not None and (efe.size != n_policies or q_pi.size != n_policies):
                bad_dimensions += 1
            valid_action = isinstance(action, (int, np.integer)) and not isinstance(action, bool)
            valid_observation = isinstance(observation, (int, np.integer)) and not isinstance(
                observation, bool
            )
            if valid_observation and int(observation) >= posterior.size:  # type: ignore[arg-type]
                bad_bounds += 1
            if valid_action and int(action) >= p_u.size:  # type: ignore[arg-type]
                bad_bounds += 1
        if (
            efe is None
            or epistemic is None
            or pragmatic is None
            or efe.shape != epistemic.shape
            or efe.shape != pragmatic.shape
            or not np.allclose(efe, epistemic + pragmatic, atol=1e-6)
        ):
            bad_decomp += 1
    report.add("coordinates", bad_coordinates == 0, f"{bad_coordinates} malformed coordinates")
    report.add("actions", bad_actions == 0, f"{bad_actions} malformed actions")
    report.add("observation_indices", bad_obs == 0, f"{bad_obs} malformed observation indices")
    report.add(
        "posterior_stochastic", bad_posterior == 0, f"{bad_posterior} invalid posterior vectors"
    )
    report.add("prior_stochastic", bad_prior == 0, f"{bad_prior} invalid prior vectors")
    report.add("efe_finite", bad_efe == 0, f"{bad_efe} invalid EFE vectors")
    report.add("q_pi_stochastic", bad_qpi == 0, f"{bad_qpi} invalid policy posterior vectors")
    report.add("p_u_stochastic", bad_pu == 0, f"{bad_pu} invalid action posterior vectors")
    report.add("vector_shapes", bad_shapes == 0, f"{bad_shapes} records have inconsistent vectors")
    report.add("index_bounds", bad_bounds == 0, f"{bad_bounds} records have out-of-range indices")
    report.add(
        "model_dimensions",
        bad_dimensions == 0,
        f"{bad_dimensions} records disagree with the model dimensions",
    )
    report.add("efe_decomposition", bad_decomp == 0, f"{bad_decomp} inconsistent decompositions")
    return report


def _probability_matrix(
    value: Any, shape: tuple[int, ...], name: str
) -> tuple[np.ndarray | None, str | None]:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        return None, str(exc)
    if array.shape != shape:
        return None, f"expected shape {shape}, got {array.shape}"
    if not np.isfinite(array).all() or (array < 0).any():
        return None, "contains non-finite or negative probabilities"
    return array, None


def validate_generative_model(agent: Any) -> ValidationReport:
    """Validate complete non-negative stochastic A/B/C/D matrices."""

    report = ValidationReport()
    n_states = getattr(agent, "n_states", None)
    n_obs = getattr(agent, "n_observations", None)
    actions = getattr(agent, "E", None)
    if (
        isinstance(n_states, (bool, np.bool_))
        or not isinstance(n_states, (int, np.integer))
        or int(n_states) < 1
    ):
        report.add("state_count", False, "n_states must be positive")
        return report
    if (
        isinstance(n_obs, (bool, np.bool_))
        or not isinstance(n_obs, (int, np.integer))
        or int(n_obs) < 1
    ):
        report.add("observation_count", False, "n_observations must be positive")
        return report
    if not isinstance(actions, (list, tuple)) or not actions:
        report.add("actions_present", False, "E must be a non-empty action sequence")
        return report
    n_states, n_obs, n_actions = int(n_states), int(n_obs), len(actions)
    action_labels_valid = all(isinstance(action, str) and bool(action) for action in actions)
    report.add("actions_labels", action_labels_valid, "E must contain non-empty strings")
    report.add(
        "actions_unique",
        action_labels_valid and len(set(actions)) == n_actions,
        "E contains duplicate labels",
    )
    matrices = {name: getattr(agent, name, None) for name in ("A", "B", "C", "D")}
    for name, value in matrices.items():
        report.add(f"{name}_present", value is not None, f"{name} is missing")
    if any(value is None for value in matrices.values()):
        return report
    A, issue = _probability_matrix(matrices["A"], (n_obs, n_states), "A")
    report.add("A_valid", issue is None, issue)
    B, issue = _probability_matrix(matrices["B"], (n_states, n_states, n_actions), "B")
    report.add("B_valid", issue is None, issue)
    C, issue = _probability_matrix(matrices["C"], (n_obs,), "C")
    report.add("C_valid", issue is None, issue)
    D, issue = _probability_matrix(matrices["D"], (n_states,), "D")
    report.add("D_valid", issue is None, issue)
    if A is not None:
        report.add("A_column_stochastic", bool(np.allclose(A.sum(axis=0), 1.0, atol=1e-8)))
    if B is not None:
        report.add("B_column_stochastic", bool(np.allclose(B.sum(axis=0), 1.0, atol=1e-8)))
    if C is not None:
        report.add("C_stochastic", bool(np.isclose(C.sum(), 1.0, atol=1e-8)))
    if D is not None:
        report.add("D_stochastic", bool(np.isclose(D.sum(), 1.0, atol=1e-8)))
    return report


def _model_from_payload(payload: dict[str, Any]) -> Any:
    return SimpleNamespace(
        n_states=payload.get("n_states"),
        n_observations=payload.get("n_observations"),
        E=payload.get("E"),
        A=payload.get("A"),
        B=payload.get("B"),
        C=payload.get("C"),
        D=payload.get("D"),
    )


def _file_nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _render_file_valid(path: Path) -> bool:
    if not _file_nonempty(path):
        return False
    signatures = {".png": b"\x89PNG\r\n\x1a\n", ".gif": (b"GIF87a", b"GIF89a")}
    expected = signatures.get(path.suffix.lower())
    if expected is None:
        return True
    prefix = path.read_bytes()[
        : max(len(item) for item in expected) if isinstance(expected, tuple) else len(expected)
    ]
    return prefix in expected if isinstance(expected, tuple) else prefix == expected


def _validate_manifest(paths: RunPaths) -> ValidationReport:
    """Validate stable artifact paths, sizes, and SHA-256 digests."""

    report = ValidationReport()
    try:
        payload = json.loads(paths.manifest_json.read_text(encoding="utf-8"))
        files = payload.get("files") if isinstance(payload, dict) else None
        report.add(
            "parses",
            isinstance(payload, dict)
            and payload.get("version") == MANIFEST_VERSION
            and payload.get("complete") is True,
            "manifest is not a completed version-1 boundary",
        )
        report.add("files_list", isinstance(files, list) and bool(files))
        if not isinstance(files, list):
            return report
        listed: set[str] = set()
        failures = 0
        for item in files:
            if not isinstance(item, dict):
                failures += 1
                continue
            relative = item.get("path")
            if (
                not isinstance(relative, str)
                or not relative
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
            ):
                failures += 1
                continue
            listed.add(relative)
            path = paths.run_dir / relative
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                valid = digest == item.get("sha256") and path.stat().st_size == item.get("size_bytes")
            except (OSError, TypeError, ValueError):
                valid = False
            failures += int(not valid)
        expected = {
            paths.config_path.relative_to(paths.run_dir).as_posix(),
            paths.trajectory_csv.relative_to(paths.run_dir).as_posix(),
            paths.summary_json.relative_to(paths.run_dir).as_posix(),
            paths.matrices_json.relative_to(paths.run_dir).as_posix(),
            paths.matrices_npz.relative_to(paths.run_dir).as_posix(),
            paths.per_step_csv.relative_to(paths.run_dir).as_posix(),
            paths.policies_json.relative_to(paths.run_dir).as_posix(),
        }
        expected.update(f"viz/{name}" for name in REQUIRED_VIZ)
        expected.add("animations/trajectory.gif")
        report.add("required_files_listed", expected <= listed, "manifest omits required artifacts")
        actual = {
            path.relative_to(paths.run_dir).as_posix()
            for path in paths.run_dir.rglob("*")
            if path.is_file() and path not in {paths.manifest_json, paths.validation_report, paths.run_log}
        }
        report.add(
            "artifact_set_complete",
            listed == actual,
            f"manifest set differs from run tree: missing={sorted(actual - listed)} "
            f"extra={sorted(listed - actual)}",
        )
        report.add("digests", failures == 0, f"{failures} manifest entries do not match")
    except (OSError, UnicodeError, json.JSONDecodeError, AttributeError, TypeError):
        report.add("parses", False, "manifest is not valid JSON")
    return report


def validate_tree_structure(paths: RunPaths) -> ValidationReport:
    """Validate that the canonical run-tree directories exist.

    This is the first, cheapest boundary: no point parsing artefacts whose
    directory tree was never created.
    """
    report = ValidationReport()
    report.artefacts = {"run_dir": str(paths.run_dir), "data_dir": str(paths.data_dir)}
    for name, path in {
        "run_dir": paths.run_dir,
        "data_dir": paths.data_dir,
        "viz_dir": paths.viz_dir,
        "animations_dir": paths.animations_dir,
    }.items():
        report.add(f"{name}_exists", path.is_dir())
    return report


def validate_artifact_presence(paths: RunPaths) -> ValidationReport:
    """Validate that every required artefact is present and non-empty."""
    report = ValidationReport()
    required_files = {
        "config": paths.config_path,
        "trajectory_csv": paths.trajectory_csv,
        "summary": paths.summary_json,
        "generative_model": paths.matrices_json,
        "generative_model_npz": paths.matrices_npz,
        "per_step_csv": paths.per_step_csv,
        "policies": paths.policies_json,
        "run_log": paths.run_log,
        "manifest": paths.manifest_json,
        "animation": paths.animations_dir / "trajectory.gif",
    }
    required_files.update({f"viz_{name}": paths.viz_dir / name for name in REQUIRED_VIZ})
    for name, path in required_files.items():
        valid = (
            _render_file_valid(path)
            if path.suffix.lower() in {".png", ".gif"}
            else _file_nonempty(path)
        )
        report.add(f"{name}_present", valid, str(path))
    return report


def validate_config_boundary(paths: RunPaths) -> tuple[ValidationReport, Any]:
    """Parse the persisted config file and return ``(report, config)``."""
    report = ValidationReport()
    config = None
    if _file_nonempty(paths.config_path):
        try:
            config = load_experiment_config(paths.config_path)
            report.add("config_parses", True)
        except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
            report.add("config_parses", False, str(exc))
    return report, config


def validate_trajectory_boundary(
    paths: RunPaths, config: Any
) -> tuple[ValidationReport, Any, dict[str, Any] | None]:
    """Parse the persisted trajectory and summary; return ``(report, trajectory, summary)``."""
    report = ValidationReport()
    trajectory = None
    summary_payload: dict[str, Any] | None = None
    if _file_nonempty(paths.trajectory_csv):
        try:
            trajectory = pd.read_csv(paths.trajectory_csv)
            report.add("trajectory_csv_parses", True)
            report.merge("trajectory", validate_trajectory_dataframe(trajectory))
        except (OSError, pd.errors.ParserError, UnicodeError, ValueError) as exc:
            report.add("trajectory_csv_parses", False, str(exc))

    if _file_nonempty(paths.summary_json):
        try:
            summary = json.loads(paths.summary_json.read_text(encoding="utf-8"))
            summary_payload = summary if isinstance(summary, dict) else None
            report.add("summary_parses", summary_payload is not None)
            report.add(
                "summary_has_contract",
                summary_payload is not None
                and {"n_rows", "n_agents", "grid_dimension"} <= set(summary_payload),
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            report.add("summary_parses", False, str(exc))
    return report, trajectory, summary_payload


def validate_model_boundary(
    paths: RunPaths, config: Any
) -> tuple[ValidationReport, dict[tuple[int, str], Any] | None]:
    """Validate every run/agent model and exact JSON/NPZ correspondence."""
    report = ValidationReport()
    models: dict[tuple[int, str], Any] | None = None
    try:
        models = _run_records(json.loads(paths.matrices_json.read_text(encoding="utf-8")))
        report.add("generative_model_parses", True)
        for (run, agent), payload in models.items():
            name = f"model.run_{run}.agent_{agent}"
            if not isinstance(payload, dict):
                report.add(f"{name}.payload", False, "agent model must be a mapping")
                continue
            report.merge(name, validate_generative_model(_model_from_payload(payload)))
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        report.add("generative_model_parses", False, str(exc))
        models = None
    try:
        with np.load(paths.matrices_npz, allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
        report.add("generative_model_npz_parses", True)
        expected = {
            f"run_{run}/agent_{agent}/{name}"
            for run, agent in (models or {}) for name in ("A", "B", "C", "D", "E")
        }
        report.add(
            "generative_model_npz_complete", bool(models) and set(arrays) == expected,
            "NPZ keys must match every and only the JSON run/agent matrices",
        )
        failures = 0
        for (run, agent), payload in (models or {}).items():
            if not isinstance(payload, dict):
                failures += 1
                continue
            for name in ("A", "B", "C", "D", "E"):
                try:
                    wanted = np.asarray(payload[name])
                    actual = arrays[f"run_{run}/agent_{agent}/{name}"]
                    equal = wanted.shape == actual.shape and (
                        np.array_equal(wanted, actual) if name == "E"
                        else np.allclose(wanted, actual, atol=0.0, rtol=0.0)
                    )
                except (KeyError, TypeError, ValueError):
                    equal = False
                failures += int(not equal)
        report.add(
            "generative_model_npz_values", bool(models) and failures == 0,
            f"{failures} JSON/NPZ matrix values differ",
        )
    except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile) as exc:
        report.add("generative_model_npz_parses", False, str(exc))
    return report, models




def validate_manifest_boundary(paths: RunPaths) -> ValidationReport:
    """Validate the manifest's version, digest integrity, and set completeness."""
    report = ValidationReport()
    if _file_nonempty(paths.manifest_json):
        report.merge("manifest", _validate_manifest(paths))
    return report


def validate_policies_boundary(
    paths: RunPaths, config: Any, models: dict | None
) -> ValidationReport:
    """Require every run/agent's complete, ordered, actually allowed policies."""
    report = ValidationReport()
    try:
        records = _run_records(json.loads(paths.policies_json.read_text(encoding="utf-8")))
        report.add("policies_parses", True)
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        report.add("policies_parses", False, str(exc))
        return report
    ready = config is not None and models is not None
    report.add("policies_prerequisites", ready)
    if not ready:
        return report
    assert models is not None
    report.add(
        "policies_run_agent_coverage", set(records) == _expected_model_keys(config) == set(models),
        "policy identities differ from configuration or models",
    )
    try:
        expected = construct_policies(
            [config.grid.dimension**2], [len(config.grid.affordances)],
            policy_len=config.grid.planning_length, max_policies=config.grid.max_policies,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        report.add("policies_complete_ordered", False, str(exc))
        return report
    failures = 0
    for record in records.values():
        if not isinstance(record, dict):
            failures += 1
            continue
        values = record.get("policies")
        if not isinstance(values, list) or not values:
            failures += 1
            continue
        valid = (
            type(record.get("n_policies")) is int
            and record["n_policies"] == len(values) == len(expected)
        )
        try:
            for value, policy in zip(values, expected, strict=True):
                array = np.asarray(value)
                valid = valid and array.dtype.kind in "iu" and np.array_equal(array, policy)
        except (TypeError, ValueError):
            valid = False
        failures += int(not valid)
    report.add("policies_complete_ordered", failures == 0, f"{failures} invalid policy sets")
    return report




def validate_per_step_boundary(
    paths: RunPaths, config: Any, trajectory: Any, models: dict | None
) -> ValidationReport:
    """Require one correctly associated diagnostic per configured run/time/agent."""
    report = ValidationReport()
    try:
        frame = pd.read_csv(paths.per_step_csv)
        records = [
            {column: _literal(value) for column, value in record.items()}
            for record in frame.to_dict(orient="records")
        ]
        report.add("per_step_csv_parses", True)
    except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as exc:
        report.add("per_step_csv_parses", False, str(exc))
        return report
    rows = _trajectory_rows(trajectory)
    ready = config is not None and rows is not None and models is not None
    report.add("per_step_prerequisites", ready, "config, keyed trajectory and models are required")
    if not ready:
        return report
    assert rows is not None and models is not None
    expected = {
        (*key, str(agent)) for key in _expected_schedule(config) if key[1] > 0
        for agent in range(config.simulation.n_agents)
    }
    seen = set()
    bad_keys = bad_values = 0
    columns = {
        "env_state": "env_states", "action": "actions", "obs_idx": "obs_idx",
        "posterior": "inferences", "prior": "priors", "efe": "efe",
        "efe_epistemic": "efe_epistemic", "efe_pragmatic": "efe_pragmatic",
        "q_pi": "q_pi", "p_u": "p_u",
    }
    for index, record in enumerate(records):
        identity = _row_identity(record)
        agent = _agent_id(record.get("agent_id"))
        key = (*identity, agent) if identity is not None and agent is not None else None
        if key is None or key in seen or key not in expected:
            bad_keys += 1
        if key is not None:
            seen.add(key)
        model = models.get((identity[0], agent)) if identity is not None else None
        if not isinstance(model, dict):
            report.add(f"per_step.{index}.model", False, "no model for this run/agent")
            continue
        report.merge(
            f"per_step.{index}", validate_per_step_records(
                [record], n_states=config.grid.dimension**2,
                n_actions=len(config.grid.affordances),
                n_policies=len(config.grid.affordances)**config.grid.planning_length,
            ),
        )
        row = rows.get(identity)
        if row is None:
            bad_values += 1
            continue
        for diagnostic, column in columns.items():
            mapping = _literal(row[column])
            try:
                value = next(
                    value for label, value in mapping.items() if _agent_id(label) == agent
                ) if isinstance(mapping, dict) else None
                actual = record.get(diagnostic)
                if diagnostic in ("action", "obs_idx"):
                    equal = (
                        isinstance(actual, (int, np.integer))
                        and not isinstance(actual, (bool, np.bool_))
                        and isinstance(value, (int, np.integer))
                        and not isinstance(value, (bool, np.bool_))
                        and actual == value
                    )
                else:
                    actual, wanted = np.asarray(actual, dtype=float), np.asarray(value, dtype=float)
                    equal = actual.shape == wanted.shape and np.allclose(
                        actual, wanted, atol=1e-12, rtol=1e-12,
                    )
            except (TypeError, ValueError, StopIteration):
                equal = False
            bad_values += int(not equal)
    report.add(
        "per_step_coverage", bad_keys == 0 and seen == expected and len(records) == len(expected),
        "diagnostics must have unique, complete configured run/time/agent identities",
    )
    report.add("per_step_trajectory_values", bad_values == 0, f"{bad_values} keyed diagnostics differ")
    return report




def validate_config_trajectory_agreement(
    config: Any, trajectory: Any, summary_payload: dict[str, Any] | None,
) -> ValidationReport:
    """Check exact backend schedules, every run's starts and run-scoped summaries."""
    report = ValidationReport()
    rows = _trajectory_rows(trajectory)
    ready = config is not None and rows is not None
    report.add("configured_trajectory_prerequisites", ready)
    if not ready:
        return report
    assert rows is not None
    report.add(
        "configured_schedule", set(rows) == _expected_schedule(config),
        "trajectory must retain every run's initial row and configured single-block steps",
    )
    expected_agents = {str(index) for index in range(config.simulation.n_agents)}
    bad_registries = out_of_bounds = 0
    for row in rows.values():
        states = _literal(row["env_states"])
        if not isinstance(states, dict):
            bad_registries += 1
            continue
        bad_registries += int({_agent_id(agent) for agent in states} != expected_agents)
        for value in states.values():
            coordinate = _coordinate(value)
            out_of_bounds += int(coordinate is None or not all(
                0 <= component < config.grid.dimension for component in coordinate
            ))
    report.add("configured_agent_count", bad_registries == 0, "agent registry differs from config")
    report.add("configured_coordinate_bounds", out_of_bounds == 0, "invalid configured coordinates")
    starts = {
        str(agent): coordinate
        for agent, coordinate in enumerate(config.simulation.resolved_initial_states)
    }
    bad_starts = 0
    for run in range(1, config.simulation.runs + 1):
        row = rows.get((run, 0, 0))
        states = _literal(row["env_states"]) if row is not None else None
        actual = {
            _agent_id(agent): _coordinate(value) for agent, value in states.items()
        } if isinstance(states, dict) else None
        bad_starts += int(actual != starts)
    report.add("configured_initial_states", bad_starts == 0, "one or more run starts differ")
    summary_ok = isinstance(summary_payload, dict) and summary_payload.get("version") == 2
    report.add("summary_run_contract", summary_ok, "version-2 run-scoped summary is required")
    if not summary_ok:
        return report
    assert summary_payload is not None
    report.add("summary_row_count", summary_payload.get("n_rows") == len(trajectory))
    report.add("summary_agent_count", summary_payload.get("n_agents") == len(expected_agents))
    report.add("summary_grid_dimension", summary_payload.get("grid_dimension") == config.grid.dimension)
    report.add("summary_run_count", summary_payload.get("n_runs") == config.simulation.runs)
    report.add("summary_max_timestep", summary_payload.get("max_timestep") == config.simulation.timesteps)
    report.add("summary_max_run", summary_payload.get("max_run") == config.simulation.runs)
    summaries = summary_payload.get("runs")
    valid_runs = isinstance(summaries, dict) and set(summaries) == {
        str(run) for run in range(1, config.simulation.runs + 1)
    }
    report.add("summary_run_coverage", valid_runs)
    if not valid_runs:
        return report
    assert isinstance(summaries, dict)
    for run in range(1, config.simulation.runs + 1):
        selected = [row for key, row in sorted(rows.items()) if key[0] == run]
        value = summaries[str(run)]
        valid = isinstance(value, dict) and bool(selected)
        if valid:
            final = _literal(selected[-1]["env_states"])
            finals = {
                str(agent): list(coordinate) if coordinate is not None else None
                for agent, coordinate in (
                    (agent, _coordinate(position)) for agent, position in final.items()
                )
            } if isinstance(final, dict) else None
            lengths = {agent: len(selected) for agent in expected_agents}
            beliefs = {agent: 0 for agent in expected_agents}
            for row in selected:
                mapping = _literal(row["inferences"])
                if isinstance(mapping, dict):
                    for agent, vector in mapping.items():
                        label = _agent_id(agent)
                        if label in beliefs and _vector(vector, "posterior") is not None:
                            beliefs[label] += 1
            valid = (
                value.get("n_rows") == len(selected)
                and value.get("n_agents") == len(expected_agents)
                and value.get("final_positions") == finals
                and value.get("trajectory_lengths") == lengths
                and value.get("belief_history_lengths") == beliefs
                and value.get("max_timestep") == max(key[1] for key in rows if key[0] == run)
            )
        report.add(f"summary.run_{run}", valid, "run-scoped summary differs from trajectory")
    return report




def validate_config_model_agreement(
    config: Any, model_payload: dict | None, trajectory: Any,
) -> ValidationReport:
    """Bind all run/agent models to config and immutable trajectory references."""
    report = ValidationReport()
    ready = config is not None and isinstance(model_payload, dict) and bool(model_payload)
    report.add("configured_model_prerequisites", ready)
    if not ready:
        return report
    assert isinstance(model_payload, dict)
    expected_keys = _expected_model_keys(config)
    report.add(
        "configured_model_agents", set(model_payload) == expected_keys,
        "models must contain every configured run/agent pair",
    )
    states = config.grid.dimension**2
    failures = 0
    digests = {}
    for key, payload in model_payload.items():
        valid = isinstance(payload, dict)
        if valid:
            valid = (
                key in expected_keys
                and all(type(payload.get(name)) is int for name in ("border", "policy_len", "max_policies"))
                and payload.get("n_states") == states and payload.get("n_observations") == states
                and payload.get("border") == config.grid.dimension - 1
                and payload.get("policy_len") == config.grid.planning_length
                and payload.get("max_policies") == config.grid.max_policies
                and payload.get("E") == list(config.grid.affordances)
            )
            try:
                digests[key] = _model_digest(payload)
            except (TypeError, ValueError):
                valid = False
            if key in expected_keys:
                start = config.simulation.resolved_initial_states[int(key[1])]
                prior = np.zeros(states)
                prior[start[0] * config.grid.dimension + start[1]] = 1.0
                try:
                    valid = valid and np.array_equal(np.asarray(payload.get("D"), dtype=float), prior)
                except (TypeError, ValueError):
                    valid = False
        failures += int(not valid)
    report.add("configured_model_dimensions", failures == 0, "model metadata/affordance order differs")
    rows = _trajectory_rows(trajectory)
    report.add("model_trajectory_prerequisites", rows is not None)
    if rows is None:
        return report
    failures = 0
    initial_failures = 0
    observed = set()
    for (run, _time, _substep), row in rows.items():
        references = _literal(row["agents"])
        if not isinstance(references, dict):
            failures += 1
            continue
        for label, reference in references.items():
            agent = _agent_id(label)
            key = (run, agent)
            observed.add(key)
            valid = (
                isinstance(reference, dict) and reference.get("agent_id") == agent
                and key in digests and reference.get("model_sha256") == digests[key]
            )
            failures += int(not valid)
            if _time == 0:
                priors = _literal(row["priors"])
                vector = _vector(priors.get(label), "prior") if isinstance(priors, dict) else None
                model = model_payload.get(key)
                valid_prior = vector is not None and vector.size == states and isinstance(model, dict)
                if valid_prior and vector is not None and isinstance(model, dict):
                    try:
                        wanted = np.asarray(model.get("D"), dtype=float)
                        valid_prior = vector.shape == wanted.shape and np.allclose(vector, wanted)
                    except (TypeError, ValueError):
                        valid_prior = False
                initial_failures += int(not valid_prior)
    report.add(
        "model_run_association", failures == 0 and observed == expected_keys,
        "trajectory references do not match their own run/agent model",
    )
    report.add("model_initial_priors", initial_failures == 0, "initial trajectory priors differ from D")
    return report


def _validate_render_coverage(paths: RunPaths, config: Any) -> ValidationReport:
    report = ValidationReport()
    report.add("render_coverage_prerequisites", config is not None)
    if config is None:
        return report
    for run, agent in sorted(_expected_model_keys(config)):
        for label in ("first", "last"):
            name = f"belief_heatmap_run_{run}_agent_{agent}_{label}.png"
            report.add(f"render.{name}", _render_file_valid(paths.viz_dir / name), name)
    return report




def validate_run_outputs(paths: RunPaths) -> ValidationReport:
    """Parse and validate every required artefact in a run tree.

    Thin aggregator over the focused per-boundary validators below. Each stage
    returns a ``ValidationReport`` and the aggregate merges them so that every
    failed check (tree, presence, config, trajectory, model, manifest,
    policies, per-step, and the config cross-checks) is recorded fail-closed.
    """
    aggregate = ValidationReport()
    aggregate.artefacts = {"run_dir": str(paths.run_dir), "data_dir": str(paths.data_dir)}
    stages = [
        validate_tree_structure(paths),
        validate_artifact_presence(paths),
    ]
    config_report, config = validate_config_boundary(paths)
    trajectory_report, trajectory, summary_payload = validate_trajectory_boundary(paths, config)
    model_report, model_payload = validate_model_boundary(paths, config)
    stages += [
        config_report,
        trajectory_report,
        model_report,
        validate_manifest_boundary(paths),
        validate_policies_boundary(paths, config, model_payload),
        validate_per_step_boundary(paths, config, trajectory, model_payload),
        validate_config_trajectory_agreement(config, trajectory, summary_payload),
        validate_config_model_agreement(config, model_payload, trajectory),
        _validate_render_coverage(paths, config),
    ]
    for stage in stages:
        aggregate.merge("artefacts", stage)
    return aggregate


__all__ = [
    "REQUIRED_TRAJECTORY_COLUMNS",
    "ValidationReport",
    "validate_artifact_presence",
    "validate_config_boundary",
    "validate_config_model_agreement",
    "validate_config_trajectory_agreement",
    "validate_generative_model",
    "validate_manifest_boundary",
    "validate_model_boundary",
    "validate_per_step_boundary",
    "validate_per_step_records",
    "validate_policies_boundary",
    "validate_run_outputs",
    "validate_trajectory_boundary",
    "validate_trajectory_dataframe",
    "validate_tree_structure",
]
