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

def actinf_graph(agent_network):

    # list of all updates to the agents in the network
    agent_updates = []

    for agent in agent_network.nodes:

        policies = construct_policies([agent_network.nodes[agent]['agent'].n_states], [len(agent_network.nodes[agent]['agent'].E)], policy_len = agent_network.nodes[agent]['agent'].policy_len)
        # get obs_idx
        obs_idx = grid.index(agent_network.nodes[agent]['env_state'])

        # infer_states
        qs_current = u.infer_states(obs_idx, agent_network.nodes[agent]['prior_A'], agent_network.nodes[agent]['prior'], noise=1)

        # calc efe
        _G = u.calculate_G_policies(agent_network.nodes[agent]['prior_A'], agent_network.nodes[agent]['prior_B'], agent_network.nodes[agent]['prior_C'], qs_current, policies=policies)

        # calc action posterior
        Q_pi = u.softmax(-_G)
        # compute the probability of each action
        P_u = u.compute_prob_actions(agent_network.nodes[agent]['agent'].E, policies, Q_pi)
        
        # sample action
        chosen_action = u.sample(P_u)

        # calc next prior
        prior = agent_network.nodes[agent]['prior_B'][:,:,chosen_action].dot(qs_current) 

        # update env state
        # action_label = params['actions'][chosen_action]

        (Y, X) = agent_network.nodes[agent]['env_state']
        Y_new = Y
        X_new = X
        # here

        if chosen_action == 0: # UP
            
            Y_new = Y - 1 if Y > 0 else Y
            X_new = X

        elif chosen_action == 1: # DOWN

            Y_new = Y + 1 if Y < agent_network.nodes[agent]['agent'].border else Y
            X_new = X

        elif chosen_action == 2: # LEFT
            Y_new = Y
            X_new = X - 1 if X > 0 else X

        elif chosen_action == 3: # RIGHT
            Y_new = Y
            X_new = X +1 if X < agent_network.nodes[agent]['agent'].border else X

        elif chosen_action == 4: # STAY
            Y_new, X_new = Y, X 
            
        current_state = (Y_new, X_new) # store the new grid location
        agent_update = {'source': agent,
                        'update_prior': prior,
                        'update_env': current_state,
                        'update_action': chosen_action,
                        'update_inference': qs_current}
        agent_updates.append(agent_update)

    return {'agent_updates': agent_updates}

def actinf_dict(agents_dict):
    # list of all updates to the agents in the network
    agent_updates = []

    for source, agent in agents_dict.items():

        policies = construct_policies([agent.n_states], [len(agent.E)], policy_len = agent.policy_len)
        # get obs_idx
        obs_idx = grid.index(agent.env_state)

        # infer_states
        qs_current = u.infer_states(obs_idx, agent.A, agent.prior)

        # calc efe
        _G = u.calculate_G_policies(agent.A, agent.B, agent.C, qs_current, policies=policies)

        # calc action posterior
        Q_pi = u.softmax(-_G)
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