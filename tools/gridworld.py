from radcad import Model, Simulation, Experiment

from model import ActiveGridference
import pandas as pd
import pyarrow.feather as feather
import sys

# Additional dependencies

# For analytics
import itertools

# local utils
import utils as u
from control import construct_policies
import random as rand
from pymdp.maths import softmax

def run_grid(dimension, no_agents, no_timesteps):
    print(f'Running Gridference. Grid dimension: {dimension} | Number of agents: {no_agents} | Timesteps: {no_timesteps}')
    grid = list(itertools.product(range(int(dimension)), repeat=2))
    print('Created Grid')

    # create a dict of agents
    agents = {}
    priors = {}
    env_states = {}
    inferences = {}
    actions = {}

    for a in range(int(no_agents)):
        # create new agent
        agent = ActiveGridference(grid)
        # generate target state
        target = (rand.randint(0,int(dimension)-1), rand.randint(0,int(dimension)-1))
        # add target state
        agent.get_C(target)
        # all agents start in the same position
        agent.get_D((0, 0))

        agents[a] = agent
        priors[a] = agent.D
        env_states[a] = agent.env_state
        inferences[a] = agent.current_inference
        actions[a] = agent.current_action
    print('Agents initialized')

    initial_state = {
    'agents': agents,
    'priors': priors,
    'env_states': env_states,
    'actions': actions,
    'inferences': inferences
    }

    params = {
    'preferred_state': grid,
    'initial_state': grid,
    'noise': [0]
    }

    def p_actinf(params, substep, state_history, previous_state):
        # State Variables
        agents = previous_state['agents']

        # list of all updates to the agents in the network
        agent_updates = []

        for source, agent in agents.items():

            policies = construct_policies([agent.n_states], [len(agent.E)], policy_len = agent.policy_len)
            # get obs_idx
            obs_idx = grid.index(agent.env_state)

            # infer_states
            qs_current = u.infer_states(obs_idx, agent.A, agent.prior, params['noise'])

            # calc efe
            _G = u.calculate_G_policies(agent.A, agent.B, agent.C, qs_current, policies=policies)

            # calc action posterior
            Q_pi = u.softmax(-_G, params['noise'])
            # compute the probability of each action
            P_u = u.compute_prob_actions(agent.E, policies, Q_pi)
            
            # sample action
            chosen_action = u.sample(P_u)

            # calc next prior
            prior = agent.B[:,:,chosen_action].dot(qs_current) 

            # update env state
            # action_label = params['actions'][chosen_action]

            (Y, X) = agent.env_state
            Y_new = Y
            X_new = X
            # here

            if chosen_action == 0: # UP
                
                Y_new = Y - 1 if Y > 0 else Y
                X_new = X

            elif chosen_action == 1: # DOWN

                Y_new = Y + 1 if Y < agent.border else Y
                X_new = X

            elif chosen_action == 2: # LEFT
                Y_new = Y
                X_new = X - 1 if X > 0 else X

            elif chosen_action == 3: # RIGHT
                Y_new = Y
                X_new = X +1 if X < agent.border else X

            elif chosen_action == 4: # STAY
                Y_new, X_new = Y, X 
                
            current_state = (Y_new, X_new) # store the new grid location
            agent_update = {'source': source,
                            'update_prior': prior,
                            'update_env': current_state,
                            'update_action': chosen_action,
                            'update_inference': qs_current}
            agent_updates.append(agent_update)

        return {'agent_updates': agent_updates}


    def s_agents(params, substep, state_history, previous_state, policy_input):

        agents_new = previous_state['agents'].copy()

        agent_updates = policy_input['agent_updates']

        if agent_updates != []:
            for update in agent_updates:
                s = update['source']
                agent = agents_new[s]
                update_prior = update['update_prior']
                update_env = update['update_env']
                update_action = update['update_action']
                update_inference = update['update_inference']

                agent.prior = update_prior
                agent.env_state = update_env
                agent.current_action = update_action
                agent.current_inference = update_inference

        return 'agents', agents_new

    def s_priors(params, substep, state_history, previous_state, policy_input):

        priors_new = previous_state['priors'].copy()

        agent_updates = policy_input['agent_updates']

        if agent_updates != []:
            for update in agent_updates:
                s = update['source']
                update_prior = update['update_prior']
                priors_new[s] = update_prior

        return 'priors', priors_new

    def s_env_states(params, substep, state_history, previous_state, policy_input):

        env_states_new = previous_state['env_states'].copy()

        agent_updates = policy_input['agent_updates']

        if agent_updates != []:
            for update in agent_updates:
                s = update['source']
                update_env = update['update_env']
                env_states_new[s] = update_env

        return 'env_states', env_states_new

    def s_actions(params, substep, state_history, previous_state, policy_input):

        actions_new = previous_state['actions'].copy()

        agent_updates = policy_input['agent_updates']

        if agent_updates != []:
            for update in agent_updates:
                s = update['source']
                update_action = update['update_action']
                actions_new[s] = update_action

        return 'actions', actions_new

    def s_inferences(params, substep, state_history, previous_state, policy_input):

        inferences_new = previous_state['inferences'].copy()

        agent_updates = policy_input['agent_updates']

        if agent_updates != []:
            for update in agent_updates:
                s = update['source']
                update_inference = update['update_inference']
                inferences_new[s] = update_inference

        return 'inferences', inferences_new

    state_update_blocks = [
        {
            'policies': {
                'p_actinf': p_actinf
            },
            'variables': {
                'agents': s_agents,
                'priors': s_priors,
                'env_states': s_env_states,
                'actions': s_actions,
                'inferences': s_inferences
            }
        }
    ]

    model = Model(
        # Model initial state
        initial_state=initial_state,
        # Model Partial State Update Blocks
        state_update_blocks=state_update_blocks,
        # System Parameters
        params=params
    )

    simulation = Simulation(
        model=model,
        timesteps=int(no_timesteps),  # Number of timesteps
        runs=1  # Number of Monte Carlo Runs
    )

    result = simulation.run()
    df = pd.DataFrame(result)
    print(df)
    df.to_csv('result.csv')

if __name__ == "__main__":
    run_grid(sys.argv[1], sys.argv[2], sys.argv[3])