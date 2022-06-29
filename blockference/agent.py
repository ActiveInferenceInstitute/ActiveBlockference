#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Agent Class
__author__: Conor Heins, Alexander Tschantz, Daphne Demekas, Brennan Klein
"""

import warnings
import numpy as np
from pymdp import inference, control, learning
from pymdp import utils, maths
from pymdp.agent import Agent
import copy



class Agent(Agent):
    """
    Agent class
    """

    def __init__(
        self,
        A,
        B,
        C=None,
        D=None,
        E=None,
        pA=None,
        pB=None,
        pD=None,
        num_controls=None,
        policy_len=1,
        inference_horizon=1,
        control_fac_idx=None,
        policies=None,
        gamma=16.0,
        use_utility=True,
        use_states_info_gain=True,
        use_param_info_gain=False,
        action_selection="deterministic",
        inference_algo="VANILLA",
        inference_params=None,
        modalities_to_learn="all",
        lr_pA=1.0,
        factors_to_learn="all",
        lr_pB=1.0,
        lr_pD=1.0,
        use_BMA=True,
        policy_sep_prior=False,
        save_belief_hist=False
    ):
        super().__init__()