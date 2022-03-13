import sys

# adding tools to the system path
sys.path.insert(0, '../tools/')

from tools.model import ActiveGridference
from tools.control import construct_policies
import tools.utils as u
import random as rand
import itertools

def grid(size, dim=2):
    grid = list(itertools.product(range(size), repeat=dim))
    return grid

def init_agents(no_agents, grid):
    # create a dict of agents
    agents = {}

    for a in range(no_agents):
        # create new agent
        agent = ActiveGridference(grid)
        # generate target state
        target = (rand.randint(0,9), rand.randint(0,9))
        # add target state
        agent.get_C(target)
        # all agents start in the same position
        agent.get_D((0, 0))

        agents[a] = agent

    return agents

def actinf_planning_single(agent, env_state, A, B, C, prior):
    policies = construct_policies([agent.n_states], [len(agent.E)], policy_len = agent.policy_len)
    # get obs_idx
    obs_idx = grid.index(env_state)

    # infer_states
    qs_current = u.infer_states(obs_idx, A, prior)

    # calc efe
    G = u.calculate_G_policies(A, B, C, qs_current, policies=policies)

    # calc action posterior
    Q_pi = u.softmax(-G)

    # compute the probability of each action
    P_u = u.compute_prob_actions(agent.E, policies, Q_pi)

    # sample action
    chosen_action = u.sample(P_u)

    # calc next prior
    prior = B[:,:,chosen_action].dot(qs_current) 

    # update env state
    # action_label = params['actions'][chosen_action]

    (Y, X) = env_state
    Y_new = Y
    X_new = X

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

    return {'update_prior': prior,
            'update_env': current_state,
            'update_action': chosen_action,
            'update_inference': qs_current}