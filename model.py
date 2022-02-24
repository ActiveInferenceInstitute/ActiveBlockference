import numpy as np
import utils

class ActiveGridference():
    """
    The ActiveInference class is to be used to create a generative model to be used in cadCAD simulations.
    The current focus is on discrete spaces. 
    ------------------------------------------------------
    An actinf generative model consists of the following:

    - (state matrix) A -> the generative model's prior beliefs about how hidden states relate to observations
    - (state-transition matrix) B -> the generative model's prior beliefs about controllable transitions between hidden states over time
    - (preference matrix) C -> the biased generative model's prior preference for particular observations encoded in terms of probabilities
    - (initial state) D -> the generative model's prior belief over hidden states at the first timestep
    - (affordances) E -> the generative model's available actions 
    """
    def __init__(self, grid) -> None:
        self.A = None
        self.B = None
        self.C = None
        self.D = None
        self.E = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]

        self.policy_len = 2

        # environment
        self.grid = grid
        self.n_states = len(self.grid)
        self.n_observations = len(self.grid)
        self.border = np.sqrt(self.n_states) - 1

        if self.grid is not None:
            self.get_A()
            self.get_B()

    def get_A(self):
        """
        State Matrix (identity matrix)
        Params:
            - n_observations: int: number of possible observations
            - n_states: int: number of possible states
        """
        self.A = np.eye(self.n_observations, self.n_states)
    
    def get_B(self):
        """State-Transition Matrix"""
        self.B = np.zeros( (len(self.grid), len(self.grid), len(self.E)) )

        for action_id, action_label in enumerate(self.E):

            for curr_state, grid_location in enumerate(self.grid):

                y, x = grid_location

                if action_label == "UP":
                    next_y = y - 1 if y > 0 else y 
                    next_x = x
                elif action_label == "DOWN":
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
                new_location = (next_y, next_x)
                next_state = self.grid.index(new_location)
                self.B[next_state, curr_state, action_id] = 1.0

    def get_C(self, preferred_state: tuple):
        """Target Location (preferences)"""
        self.C = utils.onehot(self.grid.index(preferred_state), self.n_observations)

    def get_D(self, initial_state):
        """Initial State"""
        self.D = utils.onehot(self.grid.index( initial_state ), self.n_states)