# Related work and positioning {#sec:related-work}

The free-energy principle frames adaptive systems in terms of a variational
bound on surprise and links prediction with action [@Friston2010]. Parr and
Friston develop a generalised free-energy treatment that clarifies how policy
selection can be expressed using a generative model of future outcomes
[@ParrFriston2019]. Sajid and colleagues provide an accessible comparison of
the discrete Active Inference formulation with reinforcement-learning
baselines, including the role of prior preferences [@Sajid2021].

The present work sits at the implementation layer. It does not propose a new
inference objective. Instead, it makes the finite-array formulation executable
and inspectable: every operation has a shape contract, every stochastic object
has a normalisation check, and every simulation emits enough intermediate data
to reconstruct the decision path. `pymdp` demonstrates a general Python
library for discrete Active Inference [@Heins2022]; the package's
`BlockferenceAgent` preserves a real adapter boundary for that ecosystem while
the grid pipeline remains independently deterministic at its environment
boundary.

cadCAD and radCAD supply simulation orchestration rather than Active Inference
semantics [@cadCAD2024; @radCAD2024]. ActiveBlockference uses their execution
interfaces but retains ownership of the transition function, configuration
schema, persistence contract, and release validator. This separation lets the
same scientific question be run through two backends without duplicating
behavioral rules.

## Position of the grid abstraction

The grid is a testable substrate for multi-agent interaction, not a claim about
the geometry of a particular organism. Its value is that collision resolution,
boundaries, affordances, observations, and preferences are explicit enough for
property-like tests and exact artefact comparisons. More expressive worlds can
be added by preserving the same principle: model prediction and environment
transition must share a validated semantic source.
