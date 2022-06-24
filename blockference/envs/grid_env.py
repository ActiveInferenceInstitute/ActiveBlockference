from blockference.gridference import *


class GridAgent():
    def __init__(self, grid_len, num_agents, grid_dim=2) -> None:
        self.grid = self.get_grid(grid_len, grid_dim)
        self.grid_dim = grid_dim
        self.no_actions = 2 * grid_dim + 1
        self.agents = self.init_agents(num_agents)

    def get_grid(self, grid_len, grid_dim):
        g = list(itertools.product(range(grid_len), repeat=grid_dim))
        for i, p in enumerate(g):
            g[i] += (0,)
        return g

    def move_grid(self, agent, chosen_action):
        no_actions = 2 * self.grid_dim
        state = list(agent.env_state)
        new_state = state.copy()

        # here

        if chosen_action == 0:  # STAY
            new_state = state
        else:
            if chosen_action % 2 == 1:
                index = (chosen_action+1) / 2
                new_state[index] = state[index] - 1 if state[index] > 0 else state[index]
            elif chosen_action % 2 == 0:
                index = chosen_action / 2
                new_state[index] = state[index] + 1 if state[index] < agent.border else state[index]
        return new_state

    def init_agents(self, no_agents):
        # create a dict of agents
        agents = {}

        for a in range(no_agents):
            # create new agent
            agent = ActiveGridference(self.grid)
            # generate target state
            target = (rand.randint(0, 9), rand.randint(0, 9))
            # add target state
            agent.get_C(target + (0,))
            # all agents start in the same position
            start = (rand.randint(0, 9), rand.randint(0, 9))
            agent.get_D(start + (1,))

            agents[a] = agent

        return agents

    def actinf_dict(self, agents_dict, g_agent):
        # list of all updates to the agents in the network
        agent_updates = []

        for source, agent in agents_dict.items():

            policies = construct_policies([agent.n_states], [len(agent.E)], policy_len=agent.policy_len)
            # get obs_idx
            obs_idx = g_agent.grid.index(agent.env_state)

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
            prior = agent.B[:, :, chosen_action].dot(qs_current)

            # update env state
            # action_label = params['actions'][chosen_action]

            current_state = self.move_2d(agent, chosen_action)  # store the new grid location
            agent_update = {'source': source,
                            'update_prior': prior,
                            'update_env': current_state,
                            'update_action': chosen_action,
                            'update_inference': qs_current}
            agent_updates.append(agent_update)

        return {'agent_updates': agent_updates}

    def move_2d(self, agent, chosen_action):
        (Y, X) = agent.env_state
        Y_new = Y
        X_new = X
        # here

        if chosen_action == 0:  # UP

            Y_new = Y - 1 if Y > 0 else Y
            X_new = X

        elif chosen_action == 1:  # DOWN

            Y_new = Y + 1 if Y < agent.border else Y
            X_new = X

        elif chosen_action == 2:  # LEFT
            Y_new = Y
            X_new = X - 1 if X > 0 else X

        elif chosen_action == 3:  # RIGHT
            Y_new = Y
            X_new = X + 1 if X < agent.border else X

        elif chosen_action == 4:  # STAY
            Y_new, X_new = Y, X

        return (X_new, Y_new)

    def move_3d(self, agent, chosen_action):
        (Y, X, Z) = agent.env_state
        Y_new = Y
        X_new = X
        Z_new = Z
        # here

        if chosen_action == 0:  # UP

            Y_new = Y - 1 if Y > 0 else Y
            X_new = X
            Z_new = Z

        elif chosen_action == 1:  # DOWN

            Y_new = Y + 1 if Y < agent.border else Y
            X_new = X
            Z_new = Z

        elif chosen_action == 2:  # LEFT
            Y_new = Y
            X_new = X - 1 if X > 0 else X
            Z_new = Z

        elif chosen_action == 3:  # RIGHT
            Y_new = Y
            X_new = X + 1 if X < agent.border else X
            Z_new = Z

        elif chosen_action == 4:  # IN
            X_new = X
            Y_new = Y
            Z_new = Z + 1 if Z < agent.border else Z

        elif chosen_action == 5:  # OUT
            X_new = X
            Y_new = Y
            Z_new = Z - 1 if Z > agent.border else Z

        elif chosen_action == 6:  # STAY
            Y_new, X_new, Z_new = Y, X, Z

        return (X_new, Y_new, Z_new)