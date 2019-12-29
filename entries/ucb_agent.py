import numpy as np
from numpy.random import choice
import pandas as pd

from recogym import Configuration, build_agent_init
from recogym.agents import Agent
from scipy.special import logsumexp
import sys
from scipy.stats.distributions import beta



test_agent_args = {
    'num_products': 10,
}

def beta_ucb(num_clicks, num_displays):
    return beta.ppf(0.975, num_clicks + 1, num_displays - num_clicks+ 1)

# Implement an Agent interface
class TestAgent(Agent):
    def __init__(self, config):
        super(TestAgent, self).__init__(config)

        self.product_rewards = np.zeros(self.config.num_products, dtype=np.float32)
        self.product_counts = np.ones(self.config.num_products, dtype=np.float32)
        
        self.beta_ucb_func = np.vectorize(beta_ucb)
        
    def train(self, observation, action, reward, done):
        if reward is not None and action is not None:
            self.product_rewards[action['a']] += reward
            self.product_counts[action['a']] += 1

    def act(self, observation, reward, done):
        ucb = self.beta_ucb_func(self.product_rewards, self.product_counts)
        action = np.argmax(ucb)
        ps_all = np.zeros(self.config.num_products)
        ps_all[action] = 1.0
        return {
            **super().act(observation, reward, done),
            **{
                'a': action,
                'ps': ps_all[action],
                'ps-a': ps_all,
            },
        }
