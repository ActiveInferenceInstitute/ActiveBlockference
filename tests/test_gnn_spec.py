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
from blockference.gnn_runtime import resolve_gnn_runtime
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
    """Return the JSON data contract emitted by ``blockference/gnn_source.py``."""

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
    assert arrays["E"].shape == (25,)
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
    assert np.array_equal(arrays["E"], np.full(25, 0.04))


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
    return resolve_gnn_runtime(required=False)


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
    command += ["--gnn-runtime", runtime]
    if os.environ.get("GNN_ROOT"):
        command += ["--gnn-root", os.environ["GNN_ROOT"]]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["source_qualification"]["ok"] is True
    assert payload["source_qualification"]["imports_verified"] is True
    assert payload["importer_check"]["qualified"] is True
    assert payload["importer_check"]["source_qualification"]["imports_verified"] is True
    assert payload["failures"] == []
    assert payload["gnn_version"] == "5.1.0"
    assert payload["orientation"] == ["canonical"]
    assert payload["consistency_score"] == 1.0


@pytest.mark.parametrize(("affordances", "horizon", "count"), [
    (("UP", "DOWN", "LEFT", "RIGHT", "STAY"), 2, 25),
    (("RIGHT", "DOWN", "STAY"), 2, 9),
    (("STAY",), 3, 1),
])
def test_exported_policy_prior_round_trips_without_changing_action_vocabulary(affordances, horizon, count):
    import ast
    import re

    from blockference.gnn import GridWorldSpecParameters

    parameters = GridWorldSpecParameters(affordances=affordances, planning_length=horizon)
    model = build_spec_model(parameters)
    rendered = render_spec(parameters)
    declared = ast.literal_eval(re.search(r"^E=(\([^\n]+\))$", rendered, re.MULTILINE).group(1))
    assert np.array_equal(np.asarray(declared), np.full(count, 1.0 / count))
    for name in ("E", "pi", "G"):
        shape = re.search(rf"^{name}\[(\d+),type=float\]", rendered, re.MULTILINE)
        assert int(shape.group(1)) == count
    assert spec_arrays(parameters)["B"].shape == (9, 9, len(affordances))
    assert model.E == list(affordances)


@pytest.mark.parametrize("value", [1 / 3, 1 / 25, np.nextafter(0.0, 1.0), 1 - np.finfo(float).eps])
def test_probability_text_is_float_round_trip_exact(value):
    from blockference.gnn import _format_value

    assert float(_format_value(value)) == value


def _conformance_fixture():
    parameters = spec_parameters()
    arrays = spec_arrays()
    parser = _parsed_payload()["parser"]
    parser.update(
        success=True, errors=[], warnings=[], model_name=parameters.model_name,
        variables=[
            {"name": name, "dimensions": list(array.shape)} for name, array in arrays.items()
        ] + [
            {"name": "π", "dimensions": [parameters.n_policies]},
            {"name": "G", "dimensions": [parameters.n_policies]},
        ],
    )
    return {
        "gnn_version": "5.1.0", "gnn_module": "/fixture/gnn/__init__.py", "parser": parser,
        "semantic": {"is_valid": True, "warnings": []},
        "orientation": {"warnings": [], "tensors": []},
        "consistency": {"consistent": True},
    }


@pytest.mark.parametrize("name", ["π", "G"])
@pytest.mark.parametrize("dimensions", [[5], [25, 1], []])
def test_conformance_rejects_action_sized_or_missing_policy_vector_dimensions(name, dimensions):
    from scripts.validate_gnn_spec import conformance_failures

    pin = {"tag": "v5.1.0", "commit": "a" * 40}
    payload = _conformance_fixture()
    assert conformance_failures(payload, pin=pin, source=None, spec_text=render_spec()) == []
    variable = next(value for value in payload["parser"]["variables"] if value["name"] == name)
    variable["dimensions"] = dimensions
    assert conformance_failures(payload, pin=pin, source=None, spec_text=render_spec())


@pytest.mark.parametrize(("name", "value"), [("n_actions", 25), ("n_policies", 5), ("n_policies", None)])
def test_conformance_rejects_inconsistent_redundant_control_policy_counts(name, value):
    from scripts.validate_gnn_spec import conformance_failures

    pin = {"tag": "v5.1.0", "commit": "a" * 40}
    payload = _conformance_fixture()
    assert conformance_failures(payload, pin=pin, source=None, spec_text=render_spec()) == []
    payload["parser"]["parameters"][name] = value
    assert conformance_failures(payload, pin=pin, source=None, spec_text=render_spec())


@pytest.mark.gnn
@pytest.mark.parametrize(("affordances", "policy_count"), [
    (("RIGHT", "DOWN", "STAY"), 9),
    (("STAY",), 1),
])
def test_pinned_parser_round_trips_policy_cardinality_and_precision(tmp_path, affordances, policy_count):
    # Real parsing stays inside producer admission; only primitive data returns
    # to the project interpreter.
    from blockference.gnn import GridWorldSpecParameters
    from scripts.validate_gnn_spec import (
        materialize_pin,
        read_pin_report,
        run_probe,
    )

    runtime = _pinned_runtime_python()
    if runtime is None:
        pytest.skip("pinned GNN runtime unavailable; external parsing is not qualified")
    checkout = Path(os.environ["GNN_ROOT"]) if os.environ.get("GNN_ROOT") else None
    pin = read_pin_report(runtime)["pin"]
    if checkout is not None:
        source, _origin = materialize_pin(checkout, pin, tmp_path / "pinned-source")
    else:
        source = None
    parameters = GridWorldSpecParameters(affordances=affordances, planning_length=2)
    spec = tmp_path / "policy-case.gnn.md"
    spec.write_text(render_spec(parameters), encoding="utf-8")
    payload = run_probe(runtime, spec, tmp_path / "policy-case-probe.json", source=source)
    qualification = payload["source_qualification"]
    assert qualification["ok"] and qualification["imports_verified"]
    assert qualification["commit"] == pin["commit"]
    package = Path(qualification["package_root"])
    assert payload["gnn_version"] == pin["tag"].lstrip("v")
    assert Path(payload["gnn_module"]).resolve().is_relative_to(package.resolve())
    parser = payload["parser"]
    assert parser["success"] and parser["errors"] == []
    assert parameters_from_parsed(parser) == parameters
    arrays = declared_arrays(parser)
    assert arrays["B"].shape == (9, 9, len(affordances))
    assert np.array_equal(arrays["E"], np.full(policy_count, 1 / policy_count))
    dimensions = {variable["name"]: variable["dimensions"] for variable in parser["variables"]}
    assert {name: dimensions[name] for name in ("E", "π", "G")} == {
        "E": [policy_count], "π": [policy_count], "G": [policy_count],
    }
