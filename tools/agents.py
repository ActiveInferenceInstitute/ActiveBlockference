from tools.model import ActiveGridference
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