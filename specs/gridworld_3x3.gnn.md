# GNN Specification: ActiveBlockference GridWorld 3x3

## GNNSection

ActInfPOMDP

## GNNVersionAndFlags

GNN v1.1

## ModelName

ActiveBlockference GridWorld 3x3

## ModelAnnotation

Discrete, deterministic, fully observed grid-world POMDP implemented by
`blockference.gridference.ActiveGridference` in ActiveBlockference.

- One hidden state factor over 9 cells of a 3x3 grid in
  row-major `(y, x)` order; observations are the same cells with an identity
  likelihood (`A`), the fully observed control condition of the design
  contract.
- One control factor with 5 boundary-clamped affordances in the declared order
  `UP, DOWN, LEFT, RIGHT, STAY`. `B[next_state, previous_state, action]` is
  deterministic and column-stochastic: every column selects exactly one next
  cell, and moves outside the grid clamp to the border.
- `C` is a one-hot preference over the preferred cell 2, 2; `D` is a one-hot prior over
  the initial cell 0, 0; `E` is the uniform policy prior over the 5 affordances.
- `G` is the expected free energy per policy and `pi` the policy posterior;
  the pipeline persists both components (`update_efe_epistemic`,
  `update_efe_pragmatic`) for every step.

## StateSpaceBlock

A[9,9,type=float]        # Likelihood: identity, fully observed
B[9,9,5,type=float]        # Transition: B[next_state,previous_state,action]
C[9,type=float]        # Preference over observations
D[9,type=float]        # Prior over hidden states
E[5,type=float]        # Policy prior over affordances

s[9,1,type=float]      # Current hidden state distribution
s_prime[9,1,type=float] # Next hidden state distribution
o[9,1,type=int]        # Current observation
pi[5,type=float]       # Policy posterior over affordances
u[1,type=int]            # Sampled affordance index
G[5,type=float]        # Expected free energy per affordance
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

A={
  (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
  (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
  (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
  (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
  (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
  (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
  (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
  (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
  (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
}

B={
  (
    (1.0, 0.0, 1.0, 0.0, 1.0),
    (0.0, 0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0)
  ),
  (
    (0.0, 0.0, 0.0, 1.0, 0.0),
    (1.0, 0.0, 0.0, 0.0, 1.0),
    (0.0, 0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0)
  ),
  (
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0, 0.0),
    (1.0, 0.0, 0.0, 1.0, 1.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0)
  ),
  (
    (0.0, 1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0, 1.0),
    (0.0, 0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0)
  ),
  (
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 1.0),
    (0.0, 0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0)
  ),
  (
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0, 1.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0, 0.0)
  ),
  (
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 1.0, 0.0, 1.0),
    (0.0, 0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0)
  ),
  (
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0, 0.0),
    (0.0, 1.0, 0.0, 0.0, 1.0),
    (0.0, 0.0, 1.0, 0.0, 0.0)
  ),
  (
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0, 0.0),
    (0.0, 1.0, 0.0, 1.0, 1.0)
  )
}

C=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

D=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

E=(0.2, 0.2, 0.2, 0.2, 0.2)

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
ModelTimeHorizon=2

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

grid_dimension: 3
n_states: 9
n_observations: 9
n_actions: 5
n_policies: 25
planning_length: 2
affordances: UP, DOWN, LEFT, RIGHT, STAY
action_labels: up, down, left, right, stay
preferred_state: 2, 2
initial_state: 0, 0
preferred_state_index: 8
initial_state_index: 0
observation_likelihood: identity
transition_dynamics: deterministic_boundary_clamped
b_tensor_order: next_state_previous_state_action

## Footer

ActiveBlockference GridWorld 3x3 — canonical GNN fixture of ActiveBlockference.
Generated by `blockference.gnn.render_spec`; verified with the pinned upstream
parser recorded in `specs/gnn-pin.json`.

## Signature

Generated from the implementation by `scripts/export_gnn_spec.py`. Not cryptographically signed.
