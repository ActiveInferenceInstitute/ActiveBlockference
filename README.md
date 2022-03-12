# 💡 ActiveBlockference

This is a work-in-progress repository for active inference agents in cadCAD.

Active Blockference is an open source project that will be stewarded through deep time by Active Inference Lab.
Reduce your uncertainty about how to participate in ActInfLab or contact us: https://activeinference.org/ 

## 🚧 Getting Started

```
# clone the repo

git clone https://github.com/ActiveInferenceLab/ActiveBlockference.git

cd ActiveBlockference/

# create new python enviornment

python -m venv cad

# activate the environment

source cad/bin/activate

# install requirements

pip install -r requirements.txt
```


## Developing Active Inference Agents in cadCAD

An active inference agent consists of the following matrices:
- $A$ -> $P(o|s)$ the generative model's prior beliefs about how hidden states relate to observations
- $B$ -> $𝑃(𝑠_𝑡∣𝑠_{𝑡−1},𝑢_{𝑡−1})$ the generative model's prior beliefs about controllable transitions between hidden states over time
- $C$ -> $P(o)$ the biased generative model's prior preference for particular observations encoded in terms of probabilities
- $D$ -> $P(s)$ the generative model's prior belief over hidden states at the first timestep 

### pymdp ~ Active Inference
#### Analysis of actinf_from_scratch pymdp tutorial
The pymdp inference loop has the following steps:
- initialize prior to the D matrix
- get observation index from `grid_locations`
- (q_s) perform inference over hidden states with `infer_states`, passing in the observation index, the A matrix, and the prior
- calculate expected free energy, passing in the A, B, C matrices, the inferences (q_s) from the previous step, and available actions
- compute action posterior (it's the softmax of -G)
- sample the action posterior the get the action
- compute the prior for next state with the dot product of the B matrix (indexed with the chosen action) and the current inference (q_s)

In cadCAD terms:
- policy functions:
    - `get_observation`
    - `infer_states`
    - `calc_efe`
    - `calc_action_posterior`
    - `sample_action`
    - `calc_next_prior`
    - `update_env_state`
- states:
    - `prior_A`
    - `prior_B`
    - `prior_C`
    - `env_state`
