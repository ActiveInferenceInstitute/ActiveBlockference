from blockference.gridference import *
from pymdp import utils
import copy


LOCATION_FACTOR_ID = 0
OTHER_AGENT_FACTOR_ID = 1


class TwoMultiGridAgent():
    def __init__(self, grid_len, grid_dim=2, agents=[], init_pos=[], init_obs=[]) -> None:
        """
        The GridAgent class represent the gridworld environment and keeps track of the locations of the individual agents.
        
        Params:
            grid_len: length of the gridworld
            grid_dim: dimension of the gridworld
            agents: list of agents in the environment
            init_pos: list of initial positions of the agents
            init_obs: list of initial observations the agents receive
        """
        self.grid = self.get_grid(grid_len, grid_dim)
        
        self.border = np.sqrt(len(self.grid)) - 1
        
        self.pos_dict = {}
        for i in range(0, len(self.grid)):
            self.pos_dict[i] = self.grid[i]
        print(f'Position dictionary is {self.pos_dict}')

        self.n_states = grid_len ** 2

        self.current_state = init_pos # make them indexes
        print(f'Agents are occupying the states {[self.pos_dict[v] for v in init_pos]}')
        
        self.current_obs = init_obs
        print(f'Initial observation vectors of the agents: {init_obs}')

        self.affordances = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]

        assert len(self.current_state) == len(agents), "Number of occupied states is not equal to the number of agents"

    def step(self, actions):
        """
        Step function for the gridworld environment.

        Params:
            actions: list of actions chosen by the agents in the environment
        """
        new_states = []

        for idx, action in enumerate(actions):
            # get indexes of the current reference agent and the other agent (2-agent case, in the future might be handled with a dict)
            agent_idx = idx
            other_agent_idx = 0 if agent_idx == 1 else 1
            
            # initialize new agent state
            new_agent_state = copy.deepcopy(self.current_state[agent_idx]) # new location of agent on grid
            other_agent_state = self.current_state[other_agent_idx]
            
            # get word action label
            action_label = self.affordances[int(action[0])]

            x, y = self.pos_dict[agent_idx]

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

            new_location = (next_x, next_y)
            new_agent_state = list(self.pos_dict.keys())[list(self.pos_dict.values()).index(new_location)] # returns index!
            
            # check for collisions
            if new_agent_state == other_agent_state:
                new_agent_state = self.current_state[agent_idx] # i.e. could not perform the action
            
            new_states[agent_idx] = new_agent_state

        print(f"New agent states are {new_states}")
        self.current_state = new_states
        
        # Now generate new observations for each agent (after they have both taken a step)
        new_current_obs = []
        
        for i in range(2): # not general, just for the two agents
            agent_idx = i
            other_agent_idx = 0 if agent_idx == 1 else 1
            new_current_obs[i] = [utils.onehot(new_states[agent_idx], self.num_states), utils.onehot(new_states[other_agent_idx], self.num_states)]
        
        print(f"New observations are {new_current_obs}")
        self.current_obs = new_current_obs
            

        return self.current_obs # update both agents at the same time, need to be optimized in future iterations

    def get_grid(self, grid_len, grid_dim):
        g = list(itertools.product(range(grid_len), repeat=grid_dim))
        return g