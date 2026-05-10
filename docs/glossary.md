# Glossary

One-line definitions for terms that recur across this repository.

| Term                                | One-line definition                                                                                          |
|-------------------------------------|--------------------------------------------------------------------------------------------------------------|
| **Active Inference (AIF)**          | A first-principles theory of perception, learning, and action: minimise (expected) variational free energy.  |
| **Free energy `F`**                 | Tractable upper bound on surprise; `F = E_q[log q(s) − log p(o, s)]`.                                        |
| **Expected free energy `G(π)`**     | Future-oriented free energy under policy π; decomposes into epistemic (info-gain) + pragmatic (preference).  |
| **Generative model**                | The agent's hypothesised joint distribution `p(o, s, π)` over observations, hidden states, and policies.     |
| **Hidden state `s`**                | The unobserved cause of observations, inferred via Bayes.                                                    |
| **Observation `o`**                 | What the agent's sensors deliver at a timestep.                                                              |
| **Policy `π`**                      | A sequence of actions over `policy_len` future timesteps.                                                    |
| **Action `u`**                      | The single action sampled from the action posterior at this timestep.                                        |
| **Affordance**                      | A label for an action (e.g. `"UP"`, `"DOWN"`, `"STAY"`); the set is called `E`.                              |
| **Likelihood `A`**                  | Matrix `P(o|s)`; observation given hidden state.                                                             |
| **Transition `B`**                  | Tensor `P(s_t | s_{t-1}, u_{t-1})`; controllable dynamics.                                                   |
| **Preference `C`**                  | Vector of (log-)preferences over observations; the agent prefers higher-`C` observations.                    |
| **Initial prior `D`**               | Prior over hidden states at `t = 0`.                                                                         |
| **Posterior `q(s)`**                | Belief over hidden states after seeing the observation.                                                      |
| **Softmax temperature `gamma`**     | Inverse-temperature on `softmax(-G)` controlling action determinism (`gamma=16.0` default in pymdp).         |
| **cadCAD**                          | Discrete-time generative simulation framework for complex adaptive systems (`cadcad.org`).                   |
| **radCAD**                          | Modern reimplementation of cadCAD with the same surface; ActiveBlockference uses it under the hood.          |
| **Partial state update block**      | The cadCAD unit that owns one set of policies + one set of state-update functions.                           |
| **Generative Research Team (GRT)**  | An LLM agent crew assembled to write reports; lives in `GRTs/`. Unrelated to AIF agents.                     |
| **`pymdp`**                         | Upstream Python package providing the canonical AIF primitives (`pymdp.agent.Agent`, `pymdp.maths`, …).      |
