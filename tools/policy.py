import tools.utils as u
from tools.control import construct_policies

# Policies for cadCAD actinf simulations


# single-agent with planning
def p_actinf(params, substep, state_history, previous_state, act, grid):

    policies = construct_policies([act.n_states], [len(act.E)], policy_len = act.policy_len)
    # get obs_idx
    obs_idx = grid.index(previous_state['env_state'])

    # infer_states
    qs_current = u.infer_states(obs_idx, previous_state['prior_A'], previous_state['prior'])

    # calc efe
    G = u.calculate_G_policies(previous_state['prior_A'], previous_state['prior_B'], previous_state['prior_C'], qs_current, policies=policies)

    # calc action posterior
    Q_pi = u.softmax(-G)

    # compute the probability of each action
    P_u = u.compute_prob_actions(act.E, policies, Q_pi)

    # sample action
    chosen_action = u.sample(P_u)

    # calc next prior
    prior = previous_state['prior_B'][:,:,chosen_action].dot(qs_current) 

    # update env state
    # action_label = params['actions'][chosen_action]

    (Y, X) = previous_state['env_state']
    Y_new = Y
    X_new = X

    if chosen_action == 0: # UP
          
        Y_new = Y - 1 if Y > 0 else Y
        X_new = X

    elif chosen_action == 1: # DOWN

        Y_new = Y + 1 if Y < act.border else Y
        X_new = X

    elif chosen_action == 2: # LEFT
        Y_new = Y
        X_new = X - 1 if X > 0 else X

    elif chosen_action == 3: # RIGHT
        Y_new = Y
        X_new = X +1 if X < act.border else X

    elif chosen_action == 4: # STAY
        Y_new, X_new = Y, X 
        
    current_state = (Y_new, X_new) # store the new grid location

    return {'update_prior': prior,
            'update_env': current_state,
            'update_action': chosen_action,
            'update_inference': qs_current}


# multi-agent (dict) gridworld
def p_actinf(params, substep, state_history, previous_state, grid):
    # State Variables
    agents = previous_state['agents']

    # list of all updates to the agents in the network
    agent_updates = []

    for source, agent in agents.items():

        policies = construct_policies([agent.n_states], [len(agent.E)], policy_len = agent.policy_len)
        # get obs_idx
        obs_idx = grid.index(agent.env_state)

        # infer_states
        qs_current = u.infer_states(obs_idx, agent.A, agent.prior, 0)

        # calc efe
        _G = u.calculate_G_policies(agent.A, agent.B, agent.C, qs_current, policies=policies)

        # calc action posterior
        Q_pi = u.softmax(-_G, 0)
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
