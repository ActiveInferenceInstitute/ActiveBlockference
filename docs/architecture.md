# Architecture

The package has one state transition vocabulary. `blockference.gridference._move`
maps a configured action index to a bounded `(y, x)` coordinate. `GridWorld`
uses that mapping for simultaneous collision resolution; `ActiveGridference`
uses it to construct every `B[:, :, action]`; the cadCAD/radCAD policy uses the
same environment resolver for multi-agent updates.

The execution flow is:

```text
ExperimentConfig
      │
      ▼
run_experiment ──► trajectory DataFrame
      │
      ├── persist config, CSV, model, policies, diagnostics, summary
      ├── render PNG and GIF artefacts
      └── validate each boundary and write aggregate report
```

The five generative-model components are:

* `A`: identity observation likelihood.
* `B`: one transition matrix per configured affordance.
* `C`: preferred-observation distribution.
* `D`: initial hidden-state prior.
* `E`: ordered action labels used by policy enumeration and dynamics.

All probability inputs are finite, non-negative, non-empty, shape-checked, and
normalised where a distribution is required. A failed required check makes the
aggregate pipeline result false.

The experiment seed spawns one independent NumPy child generator per run. It
draws random targets and actions without mutating process-global Python or
NumPy random state; adding later runs cannot change earlier trajectories.
Multi-agent policies resolve simultaneous proposals from the common pre-step
occupancy map; the prior persisted after a collision is the one-hot
distribution at the realized coordinate, so diagnostics describe the world
that actually occurred.
