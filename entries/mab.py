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
        super(TestAgent, self).__init__(config)
        self.trials = np.zeros(self.config.num_products)
        self.win = np.zeros(self.config.num_products)
        self.last_action = 0

    def train(self, observation, action, reward, done = False):
        """Train the model in an online fashion"""
        #if observation.sessions():
        #print("Action = {} => {}".format(action, reward))
        #pass
        if action:
            self.trials[action["a"]] += 1
            self.win[action["a"]] += reward

    def act(self, observation, reward, done):
        """Make a recommendation"""
        if reward:
            self.trials[self.last_action] += 1
            self.win[self.last_action] += reward
        #priors = [stats.beta(a=1 + w, b = 1 + t - w) for t, w in zip(self.trials, self.win)]
        #theta_samples = [d.rvs(1) for d in priors]
        #action = self.co_counts[self.last_product_viewed, :].argmax()
        action = np.argmax(self.win)
        ps_all = np.zeros(self.config.num_products)
        self.last_action = action
        ps_all[action] = 1.0
        return {
            **super().act(observation, reward, done),
            **{
                'a': action,
                'ps': 1.0,
                'ps-a': ps_all,
            },
        }


    def update_lpv(self, observation):
        """updates the last product viewed based on the observation"""
        if observation.sessions():
            self.last_product_viewed = observation.sessions()[-1]['v']
