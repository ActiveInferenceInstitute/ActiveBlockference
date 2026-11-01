"""Tests for the GNN specification and its conformance with the runtime model."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import scripts.export_gnn_spec as exporter
from blockference.gnn import (
    GNN_SPEC_PATH,
    build_spec_model,
    declared_arrays,
    parameters_from_parsed,
    render_spec,
    spec_arrays,
    spec_parameters,
)
from blockference.gridference import _move

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_SECTIONS = (
    "GNNSection",
    "GNNVersionAndFlags",
    "ModelName",
    "ModelAnnotation",
    "StateSpaceBlock",
    "Connections",
    "InitialParameterization",
    "Equations",
    "Time",
    "ActInfOntologyAnnotation",
    "ModelParameters",
    "Footer",
    "Signature",
)


def _parsed_payload() -> dict:
    """Return a payload shaped like ``scripts/gnn_parser_probe.py`` output."""

    parameters = spec_parameters()
    arrays = spec_arrays(parameters)
    return {
        "parser": {
            "parameters": {
                "A": arrays["A"].tolist(),
                "B": arrays["B"].tolist(),
                "C": list(arrays["C"]),
                "D": list(arrays["D"]),
                "E": list(arrays["E"]),
                "grid_dimension": parameters.grid_dimension,
                "n_states": parameters.n_states,
                "n_actions": parameters.n_actions,
                "n_policies": parameters.n_policies,
                "planning_length": parameters.planning_length,
                "affordances": "UP, DOWN, LEFT, RIGHT, STAY",
                "preferred_state": list(parameters.preferred_state),
                "initial_state": list(parameters.initial_state),
            }
        }
    }


def test_committed_spec_is_generated_from_the_implementation():
    assert exporter.check() == []


def test_committed_spec_declares_the_required_sections():
    text = GNN_SPEC_PATH.read_text(encoding="utf-8")
    for section in REQUIRED_SECTIONS:
        assert f"## {section}" in text
    assert "GNN v1.1" in text
    assert "grid_dimension: 3" in text
    assert "affordances: UP, DOWN, LEFT, RIGHT, STAY" in text
    assert "b_tensor_order: next_state_previous_state_action" in text
    assert "B[next_state, previous_state, action]" in text


def test_export_check_detects_drift(tmp_path):
    path = tmp_path / "gridworld.gnn.md"
    assert exporter.check(path)
    path.write_text(render_spec(), encoding="utf-8")
    assert exporter.check(path) == []
    path.write_text(render_spec() + "extra\n", encoding="utf-8")
    assert exporter.check(path)


def test_declared_arrays_match_the_runtime_model():
    parameters = spec_parameters()
    model = build_spec_model(parameters)
    arrays = spec_arrays(parameters)

    assert arrays["A"].shape == (9, 9)
    assert arrays["B"].shape == (9, 9, 5)
    assert arrays["C"].shape == (9,)
    assert arrays["D"].shape == (9,)
    assert arrays["E"].shape == (5,)
    assert np.array_equal(arrays["A"], model.A)
    assert np.array_equal(arrays["B"], model.B)
    assert np.array_equal(arrays["C"], model.C)
    assert np.array_equal(arrays["D"], model.D)


def test_declared_matrices_are_valid_probability_arrays():
    arrays = spec_arrays()

    assert np.allclose(arrays["A"].sum(axis=0), 1.0)
    assert np.array_equal(arrays["A"], np.eye(9))
    for action in range(arrays["B"].shape[2]):
        assert np.allclose(arrays["B"][:, :, action].sum(axis=0), 1.0)
        assert set(np.unique(arrays["B"][:, :, action])) <= {0.0, 1.0}
    for name in ("C", "D", "E"):
        assert np.isclose(arrays[name].sum(), 1.0)
        assert (arrays[name] >= 0).all()
    assert np.array_equal(arrays["E"], np.full(5, 0.2))


def test_transition_tensor_matches_the_declared_movement_rules():
    parameters = spec_parameters()
    model = build_spec_model(parameters)
    arrays = spec_arrays(parameters)

    for action in range(parameters.n_actions):
        for current in model.grid:
            expected_next = _move(action, current, model.border, parameters.affordances)
            column = arrays["B"][:, model.grid.index(current), action]
            assert column[model.grid.index(expected_next)] == 1.0
            assert column.sum() == 1.0


def test_parameters_are_reconstructed_from_a_parsed_document():
    payload = _parsed_payload()

    assert parameters_from_parsed(payload["parser"]) == spec_parameters()
    declared = declared_arrays(payload["parser"])
    for name, array in spec_arrays().items():
        assert np.array_equal(declared[name], array)


def test_parameter_reconstruction_rejects_malformed_documents():
    payload = _parsed_payload()
    with pytest.raises(TypeError):
        parameters_from_parsed({})
    with pytest.raises(ValueError):
        parameters_from_parsed({"parameters": {"grid_dimension": 2.5, "planning_length": 2}})
    malformed = dict(payload["parser"]["parameters"], affordances="DIAGONAL")
    with pytest.raises(ValueError):
        parameters_from_parsed({"parameters": malformed})
    missing = dict(payload["parser"]["parameters"])
    missing.pop("D")
    with pytest.raises(KeyError):
        declared_arrays({"parameters": missing})


def _pinned_runtime_python() -> str | None:
    runtime = os.environ.get("GNN_RUNTIME_PYTHON")
    if runtime and Path(runtime).is_file():
        return runtime
    root = os.environ.get("GNN_ROOT")
    if root:
        candidate = Path(root) / ".venv" / "bin" / "python"
        if candidate.is_file():
            return str(candidate)
    return None


@pytest.mark.gnn
def test_pinned_parser_validates_the_committed_spec(tmp_path):
    runtime = _pinned_runtime_python()
    if runtime is None:
        pytest.skip(
            "no pinned GNN runtime: set GNN_RUNTIME_PYTHON to an interpreter with the pinned "
            "distribution, or GNN_ROOT to a checkout that has one"
        )
    receipt = tmp_path / "gnn-receipt.json"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "validate_gnn_spec.py"),
        "--receipt",
        str(receipt),
    ]
    if os.environ.get("GNN_ROOT"):
        command += ["--gnn-root", os.environ["GNN_ROOT"], "--python", runtime]
    else:
        command += ["--gnn-runtime", runtime]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["failures"] == []
    assert payload["gnn_version"] == "5.0.3"
    assert payload["orientation"] == ["canonical"]
    assert payload["consistency_score"] == 1.0
