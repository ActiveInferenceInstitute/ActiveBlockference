"""Fail-closed validation for in-memory and persisted run artefacts."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import yaml

from blockference.config import load_experiment_config
from blockference.io.layout import RunPaths

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
    return report


def validate_per_step_records(
    records: list[dict[str, Any]], *, n_states: int | None = None, n_actions: int | None = None
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
            ):
                bad_shapes += 1
            if n_states is not None and (posterior.size != n_states or prior.size != n_states):
                bad_dimensions += 1
            if n_actions is not None and p_u.size != n_actions:
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
        return array, f"expected shape {shape}, got {array.shape}"
    if not np.isfinite(array).all() or (array < 0).any():
        return array, "contains non-finite or negative probabilities"
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
    except (OSError, json.JSONDecodeError, AttributeError, TypeError):
        report.add("parses", False, "manifest is not valid JSON")
    return report


def validate_run_outputs(paths: RunPaths) -> ValidationReport:
    """Parse and validate every required artefact in a run tree."""

    report = ValidationReport()
    report.artefacts = {"run_dir": str(paths.run_dir), "data_dir": str(paths.data_dir)}
    for name, path in {
        "run_dir": paths.run_dir,
        "data_dir": paths.data_dir,
        "viz_dir": paths.viz_dir,
        "animations_dir": paths.animations_dir,
    }.items():
        report.add(f"{name}_exists", path.is_dir())
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

    config = None
    if _file_nonempty(paths.config_path):
        try:
            config = load_experiment_config(paths.config_path)
            report.add("config_parses", True)
        except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
            report.add("config_parses", False, str(exc))

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
        except (OSError, json.JSONDecodeError) as exc:
            report.add("summary_parses", False, str(exc))

    model_payload = None
    if _file_nonempty(paths.matrices_json):
        try:
            model_payload = json.loads(paths.matrices_json.read_text(encoding="utf-8"))
            valid_payload = isinstance(model_payload, dict) and bool(model_payload)
            report.add("generative_model_parses", valid_payload)
            if isinstance(model_payload, dict) and model_payload:
                for agent_id, payload in model_payload.items():
                    if not isinstance(payload, dict):
                        report.add(
                            f"model.{agent_id}.payload",
                            False,
                            "agent model payload must be a mapping",
                        )
                        continue
                    report.merge(
                        f"model.{agent_id}",
                        validate_generative_model(_model_from_payload(payload)),
                    )
            if not valid_payload:
                model_payload = None
        except (OSError, AttributeError, json.JSONDecodeError, TypeError) as exc:
            report.add("generative_model_parses", False, str(exc))
            model_payload = None

    if _file_nonempty(paths.matrices_npz):
        try:
            with np.load(paths.matrices_npz, allow_pickle=False) as archive:
                keys = set(archive.files)
                npz_arrays = {key: np.asarray(archive[key]) for key in archive.files}
            expected = {
                f"agent_{agent_id}/{name}"
                for agent_id in (model_payload or {})
                for name in ("A", "B", "C", "D", "E")
            }
            report.add(
                "generative_model_npz_complete",
                expected <= keys,
                f"missing keys: {sorted(expected - keys)}",
            )
            value_failures = 0
            if isinstance(model_payload, dict):
                for agent_id, payload in model_payload.items():
                    if not isinstance(payload, dict):
                        continue
                    for name in ("A", "B", "C", "D", "E"):
                        key = f"agent_{agent_id}/{name}"
                        if key not in npz_arrays or name not in payload:
                            value_failures += 1
                            continue
                        try:
                            json_array = np.asarray(payload[name])
                            npz_array = npz_arrays[key]
                            equal = (
                                np.array_equal(json_array, npz_array)
                                if name == "E"
                                else json_array.shape == npz_array.shape
                                and np.allclose(json_array, npz_array, atol=0.0, rtol=0.0)
                            )
                            if not equal:
                                value_failures += 1
                        except (TypeError, ValueError):
                            value_failures += 1
            report.add(
                "generative_model_npz_values",
                value_failures == 0,
                f"{value_failures} JSON/NPZ matrix values differ",
            )
        except (OSError, ValueError, KeyError) as exc:
            report.add("generative_model_npz_parses", False, str(exc))

    if _file_nonempty(paths.manifest_json):
        report.merge("manifest", _validate_manifest(paths))

    if _file_nonempty(paths.policies_json):
        try:
            policy_payload = json.loads(paths.policies_json.read_text(encoding="utf-8"))
            policies = policy_payload.get("policies") if isinstance(policy_payload, dict) else None
            report.add("policies_parses", isinstance(policies, list) and bool(policies))
            report.add(
                "policies_count_consistent",
                isinstance(policy_payload, dict)
                and policy_payload.get("n_policies") == len(policies or []),
            )
        except (OSError, json.JSONDecodeError) as exc:
            report.add("policies_parses", False, str(exc))

    if _file_nonempty(paths.per_step_csv):
        try:
            per_step_frame = pd.read_csv(paths.per_step_csv)
            records = [
                {column: _literal(value) for column, value in row.items()}
                for row in per_step_frame.to_dict(orient="records")
            ]
            report.merge(
                "per_step",
                validate_per_step_records(
                    records,
                    n_states=config.grid.dimension**2 if config is not None else None,
                    n_actions=len(config.grid.affordances) if config is not None else None,
                ),
            )
            if config is not None and trajectory is not None:
                expected_records = sum(
                    int(float(value)) > 0 for value in trajectory["timestep"]
                ) * config.simulation.n_agents
                report.add(
                    "per_step_coverage",
                    len(records) == expected_records,
                    f"expected {expected_records} records, found {len(records)}",
                )
        except (OSError, pd.errors.ParserError, ValueError) as exc:
            report.add("per_step_csv_parses", False, str(exc))

    if config is not None and trajectory is not None and "env_states" in trajectory:
        first_env = _literal(trajectory.iloc[0]["env_states"]) if not trajectory.empty else None
        agent_ids = set(first_env) if isinstance(first_env, dict) else set()
        report.add(
            "configured_agent_count",
            len(agent_ids) == config.simulation.n_agents,
            f"trajectory has {len(agent_ids)} agents; config expects {config.simulation.n_agents}",
        )
        out_of_bounds = 0
        for value in trajectory["env_states"]:
            states = _literal(value)
            if not isinstance(states, dict):
                continue
            for coordinate in states.values():
                parsed = _coordinate(coordinate)
                if parsed is None or not all(
                    0 <= component < config.grid.dimension for component in parsed
                ):
                    out_of_bounds += 1
        report.add(
            "configured_coordinate_bounds",
            out_of_bounds == 0,
            f"{out_of_bounds} coordinates exceed the configured grid",
        )
        expected_starts = {
            index: coordinate
            for index, coordinate in enumerate(config.simulation.resolved_initial_states)
        }
        actual_starts = {
            int(agent_id): _coordinate(coordinate)
            for agent_id, coordinate in (first_env or {}).items()
            if isinstance(agent_id, (int, np.integer, str)) and str(agent_id).isdigit()
        }
        report.add(
            "configured_initial_states",
            actual_starts == expected_starts,
            "trajectory first-row positions do not match configured starts",
        )
        if summary_payload is not None:
            report.add(
                "summary_row_count",
                summary_payload.get("n_rows") == len(trajectory),
                "summary n_rows does not match trajectory rows",
            )
            report.add(
                "summary_agent_count",
                summary_payload.get("n_agents") == len(agent_ids),
                "summary n_agents does not match trajectory agents",
            )
            report.add(
                "summary_grid_dimension",
                summary_payload.get("grid_dimension") == config.grid.dimension,
                "summary grid_dimension does not match configuration",
            )

    if config is not None and isinstance(model_payload, dict):
        expected_agents = {str(index) for index in range(config.simulation.n_agents)}
        report.add(
            "configured_model_agents",
            set(model_payload) == expected_agents,
            "generative model agent IDs do not match configuration",
        )
        expected_states = config.grid.dimension**2
        bad_dimensions = 0
        for payload in model_payload.values():
            if not isinstance(payload, dict):
                continue
            try:
                matches = (
                    int(payload.get("n_states", -1)) == expected_states
                    and int(payload.get("n_observations", -1)) == expected_states
                    and len(payload.get("E", [])) == len(config.grid.affordances)
                )
            except (TypeError, ValueError):
                matches = False
            bad_dimensions += int(not matches)
        report.add(
            "configured_model_dimensions",
            bad_dimensions == 0,
            f"{bad_dimensions} model payloads disagree with configuration",
        )

    return report


__all__ = [
    "REQUIRED_TRAJECTORY_COLUMNS",
    "ValidationReport",
    "validate_generative_model",
    "validate_per_step_records",
    "validate_run_outputs",
    "validate_trajectory_dataframe",
]
