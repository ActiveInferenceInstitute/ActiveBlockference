# Abstract {#sec:abstract}

ActiveBlockference is a Python 3.10+ implementation of discrete Active
Inference agents embedded in radCAD and cadCAD simulations. This manuscript
specifies and evaluates the reconstructed implementation as a methods
artifact. The central design decision is to treat action semantics, state
transitions, numerical probability checks, persistence, rendering, and
validation as one contract rather than as independent conveniences.

The model uses a finite partially observable Markov decision process with an
identity likelihood matrix, affordance-specific transition tensors, explicit
preferences, and enumerated policies. The execution loop performs observation,
state inference, expected-free-energy evaluation, policy marginalisation,
action sampling, and prior propagation. A separate `GridWorld` applies the
same action semantics to simultaneous multi-agent moves, resolving boundary
conditions, occupied targets, shared targets, and swaps deterministically.

The implementation emits a complete run tree: validated configuration,
trajectory, summary, generative model in JSON and NPZ, policy enumeration,
per-step diagnostics, figures, animation, log, and one aggregate validation
report. The report is true only when every required content and rendering check
passes. Seeded radCAD and cadCAD runs are exercised with the same schema, and
the manuscript figures are regenerated from the package itself. The result is
a compact computational object whose equations, transitions, diagnostics, and
artefacts can be inspected and replayed without network access.
