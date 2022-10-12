from blockference.gridference import *
from pymdp import utils
import copy


LOCATION_FACTOR_ID = 0
OTHER_AGENT_FACTOR_ID = 1


class TwoGridAgent():
    def __init__(self, grid_len, grid_dim=2, agents=[]) -> None:
        """
        The GridAgent class represent the gridworld environment and keeps track of the locations of the individual agents.
        
        Params:
            grid_len: length of the gridworld
            grid_dim: dimension of the gridworld
            agents: list of agents in the environment
            no_actions: number of actions available to the agents
            n_states: number of states in the environment
            states: list of current agent states in the environment
            pos_dict: dictionary of agent states and their corresponding positions on the grid
        """
        self.grid = self.get_grid(grid_len, grid_dim)
        grid = list(itertools.product(range(3), repeat=2))
        self.border = np.sqrt(len(grid)) - 1
        self.pos_dict = {}
        for i in range(0, len(grid)):
            self.pos_dict[i] = grid[i]
        print(f'pos_dict is {self.pos_dict}')

        self.grid_dim = grid_dim
        self.no_actions = 2 * grid_dim + 1
        self.n_observations = grid_len ** 2
        self.n_states = grid_len ** 2
        # self.border = np.sqrt(self.n_states) - 1
        self.states = agents[0].D # states and locs are now the same thing
        self.E = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
        self._likelihood_dist = self._construct_likelihood_dist()

        assert len(self.states) == len(agents)

    def get_likelihood_dist(self):
        return self._likelihood_dist.copy()

    def _construct_likelihood_dist(self):

        A = utils.obj_array_zeros([ [obs_dim] + self.num_states for _, obs_dim in enumerate(self.num_obs)] )
        
        for loc in range(self.num_states[LOCATION_FACTOR_ID]):
            for reward_condition in range(self.num_states[TRIAL_FACTOR_ID]):

                if loc == 0:  # the case when the agent is in the centre location
                    # When in the centre location, reward observation is always 'no reward', or the outcome with index 0
                    A[REWARD_MODALITY_ID][0, loc, reward_condition] = 1.0

                    # When in the center location, cue observation is always 'no cue', or the outcome with index 0
                    A[CUE_MODALITY_ID][0, loc, reward_condition] = 1.0

                # The case when loc == 3, or the cue location ('bottom arm')
                elif loc == 3:

                    # When in the cue location, reward observation is always 'no reward', or the outcome with index 0
                    A[REWARD_MODALITY_ID][0, loc, reward_condition] = 1.0

                    # When in the cue location, the cue indicates the reward condition umambiguously
                    # signals where the reward is located
                    A[CUE_MODALITY_ID][reward_condition + 1, loc, reward_condition] = 1.0

                # The case when the agent is in one of the (potentially-) rewarding arms
                else:

                    # When location is consistent with reward condition
                    if loc == (reward_condition + 1):
                        # Means highest probability is concentrated over reward outcome
                        high_prob_idx = REWARD_IDX
                        # Lower probability on loss outcome
                        low_prob_idx = LOSS_IDX  #
                    else:
                        # Means highest probability is concentrated over loss outcome
                        high_prob_idx = LOSS_IDX
                        # Lower probability on reward outcome
                        low_prob_idx = REWARD_IDX

                    reward_probs = self.reward_probs[0]
                    A[REWARD_MODALITY_ID][high_prob_idx, loc, reward_condition] = reward_probs
                    reward_probs = self.reward_probs[1]
                    A[REWARD_MODALITY_ID][low_prob_idx, loc, reward_condition] = reward_probs

                    # When in the one of the rewarding arms, cue observation is always 'no cue', or the outcome with index 0
                    A[CUE_MODALITY_ID][0, loc, reward_condition] = 1.0

                # The agent always observes its location, regardless of the reward condition
                A[LOCATION_MODALITY_ID][loc, loc, reward_condition] = 1.0

        return A



    def step(self, actions):
        """
        Step function for the gridworld environment.

        Params:
            actions: list of actions chosen by the agents in the environment
        """
        
        for idx, action in enumerate(actions):
            # get indexes of the current reference agent and the other agent (2-agent case, in the future might be handled with a dict)
            agent_idx = idx
            other_agent_idx = 0 if agent_idx == 1 else 1
            
            # initialize new agent state
            new_agent_state = copy.deepcopy(self.states[agent_idx]) # new location of agent on grid
            other_agent_state = self.states[other_agent_idx]
            
            # get word action label
            action_label = self.E[int(action[0])]

            y, x = self.pos_dict[agent_idx]

            if action_label == "DOWN":
                next_y = y - 1 if y > 0 else y
                next_x = x
            elif action_label == "UP":
                next_y = y + 1 if y < self.border else y
                next_x = x
            elif action_label == "LEFT":
                next_x = x - 1 if x > 0 else x
                next_y = y
            elif action_label == "RIGHT":
                next_x = x + 1 if x < self.border else x
                next_y = y
            elif action_label == "STAY":
                next_x = x
                next_y = y
            else:
                raise ValueError(f'Action {action_label} not recognized')

            new_location = (next_y, next_x)
            new_agent_state = list(self.pos_dict.keys())[list(self.pos_dict.values()).index(new_location)]
            
            # check for collisions
            if new_agent_state == other_agent_state:
                new_agent_state = self.states[agent_idx] # i.e. could not perform the action

            self.states[agent_idx] = new_agent_state # update state

        return self.states # update both agents at the same time, need to be optimized in future iterations

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