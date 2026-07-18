# Design contract

This document records the scope that follows from the implementation's
fundamental requirements. It is the short decision boundary for contributors
who need to decide whether a proposed feature belongs in ActiveBlockference or
requires a new model contract.

## Function

Given a validated experiment configuration, the package must produce a
reproducible discrete Active Inference trajectory and a self-describing run
tree whose data, diagnostics, renders, and validation verdict agree.

The smallest complete computation is:

```text
observation → q(s) inference → policy EFE → Q(π) → P(u) → action
           → realized environment transition → next prior
```

## Fundamental requirements

These are the requirements the implementation cannot relax without changing
the model itself:

1. A state is one coordinate in a finite square grid and coordinates are
   represented as `(y, x)`.
2. An affordance index has one configured meaning everywhere: `B`, policy
   enumeration, action sampling, environment stepping, and plots.
3. Probability arrays are finite, non-negative, shape-compatible, and
   normalized wherever they represent a distribution.
4. A multi-agent step is simultaneous: every proposal is computed from the
   same pre-step occupancy map and resolved deterministically.
5. A persisted prior describes the realized world transition. A rejected
   collision proposal therefore produces a one-hot prior at the agent's
   realized coordinate.
6. A seeded run owns its random stream and does not mutate caller-global RNG
   state.
7. A release verdict is meaningful only after parsing the persisted files
   independently of the producing objects.

## Constraint classification

| Item | Classification | Current decision |
|---|---|---|
| radCAD and cadCAD backends | Soft choice | Keep both behind one state-update contract. |
| YAML and TOML input | Soft choice | Accept both through one strict dataclass schema. |
| CSV and JSON required artefacts | Soft choice | Keep readable interchange formats as the minimum contract. |
| Parquet output | Optional convenience | Write it only when a compatible writer exists. |
| One shared `initial_state` field | Compatibility assumption | Preserve it by broadcasting; add `initial_states` for explicit starts. |
| Fully observed identity `A` | Model scope | Keep as the control condition; partial observation is a new model feature. |
| Deterministic square-grid `B` | Model scope | Do not imply stochastic or obstacle dynamics without new validation. |
| Large policy horizons | Physical resource limit | Fail or redesign before exponential enumeration becomes accidental. |

## Release gates

The release path is intentionally layered:

1. Ruff and deterministic unit tests establish local code invariants.
2. Both simulation engines execute real backend integrations.
3. The pipeline persists trajectory, model, policies, diagnostics, summary,
   log, and renders.
4. The validator reparses configuration, trajectory, model JSON/NPZ, summary,
   policies, diagnostics, dimensions, and render signatures.
5. Notebook execution and manuscript source/build checks complete.

`ValidationReport.ok` and `PipelineResult.ok` remain the only release verdicts.
Passing tests without a passing persisted artefact report is not a release.

## Outside the current scope

Obstacles, stochastic transitions, learned `A`/`B`/`C`, factored state spaces,
unbounded policy search, network-backed research, and empirical claims about
scientific adequacy are deliberately outside this contract. They can be added
only with explicit state semantics, validation rules, tests, and regenerated
documentation and figures.
