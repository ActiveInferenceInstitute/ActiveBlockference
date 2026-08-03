# Formal model {#sec:formal-model}

## State spaces and generative model

::: definition {#def:state-space}
**Discrete state space.** Let $S = \{0, \ldots, n-1\}$ be the finite hidden
state set, $O = \{0, \ldots, n-1\}$ the observation set, and
$U = \{0, \ldots, m-1\}$ the configured affordance set. A grid of side length
$d$ has $n=d^2$ states, ordered row-major by coordinates $(y,x)$.
:::

::: definition {#def:generative-model}
**Validated generative model.** A model is the tuple $(A,B,C,D,E)$, where
$A \in [0,1]^{|O|\times |S|}$ is column-stochastic,
$B \in [0,1]^{|S|\times |S|\times |U|}$ is column-stochastic for every
action, $C \in [0,1]^{|O|}$ is a normalised preference distribution,
$D \in [0,1]^{|S|}$ is a normalised initial prior, and $E$ is the ordered
affordance vocabulary. Every array is finite, non-negative, non-empty, and
shape-compatible with the other arrays.
:::

For the current grid implementation, $A$ is the identity matrix. Thus an
observation identifies a grid location, while uncertainty can still enter
through the prior used during inference. A configured affordance subset changes
both the action dimension of $B$ and the policy enumeration; it is not merely a
display preference.

## Perception

Let $o_t$ be the observed location and $D_t$ the prior at step $t$. The
posterior is computed by normalising the product of the likelihood column and
the prior:

$$
q_t(s) = \operatorname{softmax}\left(\log A_{o_t,s} + \log D_t(s)\right).
$$ {#eq:state-inference}

The posterior update in [@eq:state-inference] is implemented by `infer_states`.
The implementation clips zero probabilities only inside a finite logarithm;
it never converts negative or non-finite values into a distribution. This
separation matters because a numerical convenience must not turn invalid model
input into a passing run.

## Prediction and expected free energy

For a candidate action $u$, the expected next-state posterior is:

$$
q_{t+1}^{(u)}(s') = \sum_{s \in S} B_{s',s,u}q_t(s).
$$ {#eq:state-prediction}

The prediction in [@eq:state-prediction] is applied once per predicted action.
For a
policy $\pi=(u_t,\ldots,u_{t+H-1})$, repeated application of the
transition slices gives the predicted state sequence. The corresponding
observation distribution is $q(o\mid\pi)=Aq(s\mid\pi)$. The implementation
uses the ambiguity-plus-risk decomposition:

$$
G(\pi) = \sum_{\tau=t}^{t+H-1}
\left[\mathcal{H}\left(Aq_\tau(s\mid\pi)\right)
 + \operatorname{KL}\left(Aq_\tau(s\mid\pi)\,\middle\|\,C\right)\right].
$$ {#eq:expected-free-energy}

The decomposition in [@eq:expected-free-energy] uses expected observation
entropy for the first term and preference distance for the second. This is the
implementation-level decomposition used by `calculate_G_policies_traced`; it is
described as a computational decomposition rather than as a claim that every
formulation of expected free energy has identical terms [@ParrFriston2019].

The policy posterior and first-action marginal are:

$$
Q(\pi) = \operatorname{softmax}\left(-G(\pi)\right), \qquad
P(u_t=u) = \sum_{\pi:\,\pi_0=u} Q(\pi).
$$ {#eq:policy-and-action-posteriors}

The policy and action distributions in [@eq:policy-and-action-posteriors] are
used by the action sampler, which draws from $P(u_t)$ after a final positivity and
normalisation check. Each simulation owns a NumPy generator for stochastic
targets and actions; it does not mutate process-global random state.

## Model fields and implementation mapping

| Field | Mathematical object | ActiveBlockference representation |
|---|---|---|
| `A` | $P(o\mid s)$ | identity likelihood over grid locations |
| `B` | $P(s'\mid s,u)$ | deterministic affordance-indexed tensor |
| `C` | preferred observations | one-hot target preference |
| `D` | $P(s_0)$ | one-hot initial state prior |
| `E` | control labels | ordered validated strings |

: Generative-model fields and their concrete implementation. {#tbl:model-fields}

The `ActiveGridference` object owns the model dimensions and rebuilds `B` when
the affordance vocabulary changes. The separate `BlockferenceAgent` adapter
is intentionally narrow: it delegates to the installed pymdp API when a user
needs that ecosystem, while the grid pipeline remains a NumPy implementation
with its own checked primitives.
