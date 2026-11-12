"""Generalized Notation Notation (GNN) declaration of the grid-world model.

The generative model that :class:`blockference.gridference.ActiveGridference`
implements is also published as a GNN document under ``specs/``. This module owns
the single canonical construction shared by both directions:

* :func:`spec_parameters` returns the fixture parameters the committed GNN
  document declares;
* :func:`spec_arrays` builds the ``A``/``B``/``C``/``D``/``E`` arrays through
  :class:`ActiveGridference`, so the document and the runtime model cannot
  diverge;
* :func:`render_spec` renders the deterministic ``.gnn.md`` text that
  ``scripts/export_gnn_spec.py`` writes or verifies;
* :func:`parameters_from_parsed` reconstructs the parameters from a pinned GNN
  parser payload, which lets ``scripts/validate_gnn_spec.py`` build the runtime
  model *from* the parsed document and compare it with the declared matrices.

The pin record in ``specs/gnn-pin.json`` names the exact upstream revision whose
parser validates the document; see ``docs/gnn.md`` for the commands.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from blockference.actions import DEFAULT_AFFORDANCES, validate_affordances
from blockference.gridference import ActiveGridference, make_grid

__all__ = [
    "GNN_PIN_PATH",
    "GNN_SPEC_PATH",
    "SPEC_DIR",
    "GridWorldSpecParameters",
    "build_spec_model",
    "declared_arrays",
    "load_pin",
    "parameters_from_parsed",
    "render_spec",
    "spec_arrays",
    "spec_parameters",
]

ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = ROOT / "specs"
GNN_SPEC_PATH = SPEC_DIR / "gridworld_3x3.gnn.md"
GNN_PIN_PATH = SPEC_DIR / "gnn-pin.json"

#: GNN language revision declared by the document header.
GNN_LANGUAGE_VERSION = "GNN v1.1"

#: Declared matrix keys of the canonical fixture, in declaration order.
SPEC_MATRIX_KEYS = ("A", "B", "C", "D", "E")


@dataclass(frozen=True)
class GridWorldSpecParameters:
    """Fixture parameters declared by the committed GNN document."""

    grid_dimension: int = 3
    affordances: tuple[str, ...] = DEFAULT_AFFORDANCES
    planning_length: int = 2
    preferred_state: tuple[int, int] = (2, 2)
    initial_state: tuple[int, int] = (0, 0)
    model_name: str = "ActiveBlockference GridWorld 3x3"

    def __post_init__(self) -> None:
        if self.grid_dimension < 2:
            raise ValueError("grid_dimension must be an integer >= 2")
        if self.planning_length < 1:
            raise ValueError("planning_length must be a positive integer")
        affordances = validate_affordances(self.affordances)
        object.__setattr__(self, "affordances", tuple(affordances))
        for name, coordinate in (
            ("preferred_state", self.preferred_state),
            ("initial_state", self.initial_state),
        ):
            if len(coordinate) != 2 or any(
                isinstance(component, bool) or not isinstance(component, int)
                for component in coordinate
            ):
                raise ValueError(f"{name} must be a two-dimensional integer coordinate")
            if any(not 0 <= component < self.grid_dimension for component in coordinate):
                raise ValueError(f"{name} {coordinate} is outside the declared grid")

    @property
    def n_states(self) -> int:
        """Number of hidden states and observations of the square grid."""

        return self.grid_dimension**2

    @property
    def n_actions(self) -> int:
        """Number of declared affordances."""

        return len(self.affordances)

    @property
    def n_policies(self) -> int:
        """Number of enumerated policies for the declared planning length."""

        return self.n_actions**self.planning_length


def spec_parameters() -> GridWorldSpecParameters:
    """Return the canonical fixture parameters of the committed document."""

    return GridWorldSpecParameters()


def build_spec_model(parameters: GridWorldSpecParameters) -> ActiveGridference:
    """Build the runtime model for the declared fixture parameters.

    The model is constructed through :class:`ActiveGridference` only, so the
    document, the exporter, and the simulation pipeline share one
    implementation of the likelihood, transition, preference, and prior.
    """

    model = ActiveGridference(
        make_grid(parameters.grid_dimension),
        planning_length=parameters.planning_length,
        env_state=parameters.initial_state,
        affordances=parameters.affordances,
    )
    model.get_C(parameters.preferred_state)
    model.get_D(parameters.initial_state)
    return model


def spec_arrays(parameters: GridWorldSpecParameters | None = None) -> dict[str, np.ndarray]:
    """Return the declared ``A``/``B``/``C``/``D``/``E`` arrays.

    ``E`` is the uniform policy prior over the declared affordances, which is
    the GNN convention; the affordance *labels* are declared separately in the
    document's ``ModelParameters`` section.
    """

    resolved = parameters or spec_parameters()
    model = build_spec_model(resolved)
    return {
        "A": np.asarray(model.A, dtype=float),
        "B": np.asarray(model.B, dtype=float),
        "C": np.asarray(model.C, dtype=float),
        "D": np.asarray(model.D, dtype=float),
        "E": np.full(resolved.n_actions, 1.0 / resolved.n_actions, dtype=float),
    }


def load_pin(path: Path | None = None) -> dict[str, Any]:
    """Return the upstream pin, rejecting records outside the local pin contract."""

    pin = json.loads((path or GNN_PIN_PATH).read_text(encoding="utf-8"))
    if not isinstance(pin, dict):
        raise ValueError("invalid GNN pin: expected a JSON object")
    if pin.get("schema") != "gnn-importer-pin/v1":
        raise ValueError("invalid GNN pin schema: expected gnn-importer-pin/v1")
    if pin.get("tag") != "v5.0.3":
        raise ValueError("invalid GNN pin tag: expected v5.0.3")
    commit = pin.get("commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-fA-F]{40}", commit) is None:
        raise ValueError("invalid GNN pin commit: expected a full 40-hex revision")
    return pin


def _format_value(value: float) -> str:
    return f"{float(value):.1f}"


def _format_vector(values: Iterable[float]) -> str:
    return "(" + ", ".join(_format_value(value) for value in values) + ")"


def _format_matrix(matrix: np.ndarray) -> str:
    rows = [_format_vector(row) for row in np.asarray(matrix, dtype=float)]
    return "{\n  " + ",\n  ".join(rows) + "\n}"


def _format_transition(tensor: np.ndarray) -> str:
    """Render ``B[next_state, previous_state, action]`` as nested tuples."""

    slices: list[str] = []
    for next_state in np.asarray(tensor, dtype=float):
        rows = [f"    {_format_vector(row)}" for row in next_state]
        slices.append("  (\n" + ",\n".join(rows) + "\n  )")
    return "{\n" + ",\n".join(slices) + "\n}"


def render_spec(parameters: GridWorldSpecParameters | None = None) -> str:
    """Render the deterministic GNN document text for the declared model."""

    resolved = parameters or spec_parameters()
    arrays = spec_arrays(resolved)
    n = resolved.n_states
    horizon = resolved.planning_length
    action_labels = ", ".join(label.lower() for label in resolved.affordances)
    affordances = ", ".join(resolved.affordances)
    preferred = ", ".join(str(component) for component in resolved.preferred_state)
    initial = ", ".join(str(component) for component in resolved.initial_state)
    preferred_index = resolved.preferred_state[0] * resolved.grid_dimension + resolved.preferred_state[1]
    initial_index = resolved.initial_state[0] * resolved.grid_dimension + resolved.initial_state[1]

    return f"""# GNN Specification: {resolved.model_name}

## GNNSection

ActInfPOMDP

## GNNVersionAndFlags

{GNN_LANGUAGE_VERSION}

## ModelName

{resolved.model_name}

## ModelAnnotation

Discrete, deterministic, fully observed grid-world POMDP implemented by
`blockference.gridference.ActiveGridference` in ActiveBlockference.

- One hidden state factor over {n} cells of a {resolved.grid_dimension}x{resolved.grid_dimension} grid in
  row-major `(y, x)` order; observations are the same cells with an identity
  likelihood (`A`), the fully observed control condition of the design
  contract.
- One control factor with {resolved.n_actions} boundary-clamped affordances in the declared order
  `{affordances}`. `B[next_state, previous_state, action]` is
  deterministic and column-stochastic: every column selects exactly one next
  cell, and moves outside the grid clamp to the border.
- `C` is a one-hot preference over the preferred cell {preferred}; `D` is a one-hot prior over
  the initial cell {initial}; `E` is the uniform policy prior over the {resolved.n_actions} affordances.
- `G` is the expected free energy per policy and `pi` the policy posterior;
  the pipeline persists both components (`update_efe_epistemic`,
  `update_efe_pragmatic`) for every step.

## StateSpaceBlock

A[{n},{n},type=float]        # Likelihood: identity, fully observed
B[{n},{n},{resolved.n_actions},type=float]        # Transition: B[next_state,previous_state,action]
C[{n},type=float]        # Preference over observations
D[{n},type=float]        # Prior over hidden states
E[{resolved.n_actions},type=float]        # Policy prior over affordances

s[{n},1,type=float]      # Current hidden state distribution
s_prime[{n},1,type=float] # Next hidden state distribution
o[{n},1,type=int]        # Current observation
pi[{resolved.n_actions},type=float]       # Policy posterior over affordances
u[1,type=int]            # Sampled affordance index
G[{resolved.n_actions},type=float]        # Expected free energy per affordance
t[1,type=int]            # Discrete time step

## Connections

D>s
t>s
s-A
A-o
s-B
u>B
B>s_prime
C>G
E>pi
G>pi
pi>u
s>s_prime

## InitialParameterization

A={_format_matrix(arrays["A"])}

B={_format_transition(arrays["B"])}

C={_format_vector(arrays["C"])}

D={_format_vector(arrays["D"])}

E={_format_vector(arrays["E"])}

## Equations

Likelihood (identity, fully observed):
  A[o, s] = 1 if o == s else 0

Transition (deterministic, boundary clamped):
  B[s_prime, s, u] = 1 if s_prime == move(u, s) else 0
  move(UP, (y, x)) = (max(0, y - 1), x)
  move(DOWN, (y, x)) = (min(dimension - 1, y + 1), x)
  move(LEFT, (y, x)) = (y, max(0, x - 1))
  move(RIGHT, (y, x)) = (y, min(dimension - 1, x + 1))
  move(STAY, (y, x)) = (y, x)

State inference and policy evaluation (per step):
  q(s) = softmax(log A[o, :] + log prior)
  G(pi) = sum over horizon of ambiguity(H[A q(s)]) + risk(KL(A q(s) || C))
  Q(pi) = softmax(-G(pi))
  P(u) = sum over policies of pi[0] * Q(pi)
  prior_next = B[:, :, u] q(s)

## Time

Time=t
Dynamic
Discrete
ModelTimeHorizon={horizon}

## ActInfOntologyAnnotation

A=LikelihoodMatrix
B=TransitionMatrix
C=PreferenceVector
D=PriorOverHiddenStates
E=Habit
G=ExpectedFreeEnergy
s=HiddenState
s_prime=NextHiddenState
o=Observation
pi=PolicyVector
u=Action
t=Time

## ModelParameters

grid_dimension: {resolved.grid_dimension}
n_states: {n}
n_observations: {n}
n_actions: {resolved.n_actions}
n_policies: {resolved.n_policies}
planning_length: {horizon}
affordances: {affordances}
action_labels: {action_labels}
preferred_state: {preferred}
initial_state: {initial}
preferred_state_index: {preferred_index}
initial_state_index: {initial_index}
observation_likelihood: identity
transition_dynamics: deterministic_boundary_clamped
b_tensor_order: next_state_previous_state_action

## Footer

{resolved.model_name} — canonical GNN fixture of ActiveBlockference.
Generated by `blockference.gnn.render_spec`; verified with the pinned upstream
parser recorded in `specs/gnn-pin.json`.

## Signature

Generated from the implementation by `scripts/export_gnn_spec.py`. Not cryptographically signed.
"""


def _as_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"parsed {name} must be a mapping, got {type(value).__name__}")
    return value


def _scalar(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"parsed {name} must be a number, got {value!r}")
    if float(value) != int(value):
        raise ValueError(f"parsed {name} must be an integer, got {value!r}")
    return int(value)


def _coordinate(value: Any, *, name: str) -> tuple[int, int]:
    if isinstance(value, (str, bytes)):
        parts = str(value).split(",")
    elif isinstance(value, Sequence):
        parts = list(value)
    else:
        raise TypeError(f"parsed {name} must be a coordinate, got {value!r}")
    if len(parts) != 2:
        raise ValueError(f"parsed {name} must have two components, got {value!r}")
    return (_scalar(parts[0], name=f"{name}[0]"), _scalar(parts[1], name=f"{name}[1]"))


def _affordances(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        labels = [part.strip() for part in value.split(",")]
    elif isinstance(value, Sequence):
        labels = [str(part).strip() for part in value]
    else:
        raise TypeError(f"parsed affordances must be a sequence, got {value!r}")
    return validate_affordances([label.upper() for label in labels if label])


def parameters_from_parsed(payload: Mapping[str, Any]) -> GridWorldSpecParameters:
    """Reconstruct the fixture parameters from a pinned parser payload.

    ``payload`` is the JSON object emitted by ``scripts/validate_gnn_spec.py``
    from the pinned upstream parser: the model's ``ModelParameters`` entries are
    read back so the runtime model is built from the document, not from a
    duplicate copy of these values.
    """

    parameters = _as_mapping(payload.get("parameters"), name="parameters")
    return GridWorldSpecParameters(
        grid_dimension=_scalar(parameters.get("grid_dimension"), name="grid_dimension"),
        affordances=_affordances(parameters.get("affordances")),
        planning_length=_scalar(parameters.get("planning_length"), name="planning_length"),
        preferred_state=_coordinate(parameters.get("preferred_state"), name="preferred_state"),
        initial_state=_coordinate(parameters.get("initial_state"), name="initial_state"),
    )


def declared_arrays(payload: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return the matrices a pinned parser payload declared, keyed by name."""

    parameters = _as_mapping(payload.get("parameters"), name="parameters")
    arrays: dict[str, np.ndarray] = {}
    for key in SPEC_MATRIX_KEYS:
        value = parameters.get(key)
        if value is None:
            raise KeyError(f"parsed document does not declare matrix {key!r}")
        arrays[key] = np.asarray(value, dtype=float)
    return arrays
