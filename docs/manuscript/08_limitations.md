# Limitations and future extensions {#sec:limitations}

The implementation has clear boundaries.

First, the default grid uses fully observed identity likelihoods. This is a
useful control condition for testing inference, but it does not exercise the
full range of perceptual ambiguity found in partially observed environments.
Second, the current transition tensor is deterministic and the collision rule
is conservative. Stochastic movement, obstacles, and richer interaction
mechanisms would require new validated semantics and new artefact checks.
Third, policy enumeration grows exponentially with planning length and action
count. The package is designed for small discrete models, not unrestricted
large-scale planning.

Fourth, the expected-free-energy implementation records ambiguity and
preference-distance terms. Other Active Inference formulations may include
additional epistemic or temporal terms, so numerical equivalence with another
implementation requires aligning definitions, priors, horizons, and
preferences rather than comparing labels alone. The scholarly literature
contains multiple related formulations [@ParrFriston2019; @Sajid2021].

Finally, a passing validation report establishes software and artefact
integrity, not empirical adequacy of a scientific hypothesis. A researcher
must still choose a task, identify observations and preferences independently,
perform sensitivity analysis, and compare model predictions against data.

Future work can add obstacle maps, factored state spaces, stochastic
transitions, learning of `A`/`B`/`C`, and larger policy search while preserving
the contracts formalised in [@def:generative-model], [@prop:collision-determinism],
and [@def:validation-report].
