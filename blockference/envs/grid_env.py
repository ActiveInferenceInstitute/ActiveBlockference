from blockference.gridference import *


class GridAgent():
    def __init__(self, grid_len, grid_dim=2, agents=[]) -> None:
        self.grid = self.get_grid(grid_len, grid_dim)
        self.grid_dim = grid_dim
        self.no_actions = 2 * grid_dim + 1
        self.n_observations = grid_len ** 2
        self.n_states = grid_len ** 2
        self.border = np.sqrt(self.n_states) - 1
        self.states = [agent.D for agent in agents]
        self.rel_locs = ["NONE", "NEXT_LEFT", "NEXT_RIGHT", "ABOVE", "BELOW"]
        self.agent_locs = [np.nonzero(self.states[0][0])[0][0], np.nonzero(self.states[1][0])[0][0]]
        assert len(self.states) == len(agents)

    def step(self, actions):
        assert len(self.states) == len(actions), "Number of actions received is more than number of agents"
        next_state = copy.deepcopy(self.states)
        
        for idx, action in enumerate(actions):
            new_loc = copy.deepcopy(self.states[idx][0]) # new location of agent on grid
            new_ref = copy.deepcopy(self.states[idx][1]) # new relative position to the other agent on the grid
            
            y, x = self.states[idx][0]

            if action_label == "DOWN":
                next_y = y - 1 if y > 0 else y
                next_x = x
            elif action_label == "UP":
                next_y = y + 1 if y < border else y
                next_x = x
            elif action_label == "LEFT":
                next_x = x - 1 if x > 0 else x
                next_y = y
            elif action_label == "RIGHT":
                next_x = x + 1 if x < border else x
                next_y = y
            elif action_label == "STAY":
                next_x = x
                next_y = y
            new_location = (next_y, next_x)
            try:
                rel_pos = self.get_rel_pos(new_location, self.states[idx+1][0])
            except:
                rel_pos = self.get_rel_pos(new_location, self.states[idx-1][0])
            if rel_pos == "COLLISION":
                new_location = self.states[idx][0]
                next_state = (grid.index(new_location), new_ref)
            else:
                new_ref = self.rel_locs.index(rel_pos)
                next_state[idx] = (grid.index(new_location), new_ref)
            self.agent_locs[idx] = new_location
        return next_state # update both agents at the same time, need to be optimized in future iterations

    def get_rel_pos(self, loc1, loc2):
        rel_pos = ""
        
        if loc1[0] == loc2[0]: # on the same x-position
            if (loc1[1] > loc2[1]) and ((loc1[1] - loc2[1]) == 1): # agent_2 is below agent_1
                rel_pos = "BELOW"
            elif (loc1[1] < loc2[1]) and ((loc1[1] - loc2[1]) == 1): # agent_2 is above agent_1
                rel_pos = "ABOVE"
            else:
                rel_pos = "NONE"
        elif loc1[1] == loc2[1]: # on the same x-position
            if (loc1[0] > loc2[0]) and ((loc1[0] - loc2[0]) == 1): # agent_2 is to the left of agent_1
                rel_pos = "NEXT_LEFT"
            elif (loc1[0] < loc2[0]) and ((loc1[0] - loc2[0]) == 1): # agent_2 is above agent_1
                rel_pos = "NEXT_RIGHT"
            else:
                rel_pos = "NONE"
        elif (loc1[0] == loc2[0]) and (loc1[1] == loc2[1]): # on the same position, need to handle this better
            rel_pos = "COLLISION"
        else:
            rel_pos = "NONE"
        return rel_pos

    def get_grid(self, grid_len, grid_dim):
        g = list(itertools.product(range(grid_len), repeat=grid_dim))
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
                new_state[index] = state[index] + 1 if state[index] < self.border else state[index]
        return new_state


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