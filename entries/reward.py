import numpy as np
import scipy.stats as stats

from recogym import Configuration, build_agent_init, to_categorical
from recogym.agents import Agent

test_agent_args = {
    'num_products': 10,
}

class TestAgent(Agent):
    """Organic counter agent"""

    def __init__(self, config = Configuration(test_agent_args)):
        self.cpt = 0 
        super(TestAgent, self).__init__(config)
        self.trials = np.zeros(self.config.num_products)
        self.win = np.zeros(self.config.num_products)
        self.last_action = 0

    def train(self, observation, action, reward, done = False):
        """Train the model in an online fashion"""
        self.cpt += 1
        #pass
        if action:
            self.trials[action["a"]] += 1
            self.win[action["a"]] += reward

    def act(self, observation, reward, done):
        """Make a recommendation"""
        self.cpt += 1
        if reward:
            self.trials[self.last_action] += 1
            self.win[self.last_action] += reward
        #action = np.argmax(self.win)
        #ratios = self.win /  (self.trials + 1)
        action = np.argmax(self.win)
        #action = np.random.choice(self.config.num_products, p=ratios / ratios.sum())
        ps_all = np.zeros(self.config.num_products)
        self.last_action = action
        ps_all[action] = 1.0
        return {
            **super().act(observation, reward, done),
            **{
                'a': action,
                'ps': "LOL",
                'ps-a': ps_all,
            },
        }


    def update_lpv(self, observation):
        """updates the last product viewed based on the observation"""
        if observation.sessions():
            self.last_product_viewed = observation.sessions()[-1]['v']
