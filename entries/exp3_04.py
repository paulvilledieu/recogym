import numpy as np

from recogym import Configuration, build_agent_init, to_categorical
from recogym.agents import Agent
from scipy.special import logsumexp
from numpy.random import choice

test_agent_args = {
    'num_products': 10,
}

def categorical_draw(probs):
    z = np.random.rand()
    cum_prob = 0.0
    for i in range(len(probs)):
        prob = probs[i]
        cum_prob += prob
        if cum_prob > z:
            return i

class TestAgent(Agent):
    def __init__(self, config = Configuration(test_agent_args)):
        super(TestAgent, self).__init__(config)
        self.gamma = 0.4
        n_arms = self.config.num_products
        self.weights = [1.0 for i in range(n_arms)]

    def select_arm(self):
        n_arms = len(self.weights)
        total_weight = sum(self.weights)
        probs = [0.0 for i in range(n_arms)]
        for arm in range(n_arms):
            probs[arm] = (1 - self.gamma) * (self.weights[arm] / total_weight)
            probs[arm] = probs[arm] + (self.gamma) * (1.0 / float(n_arms))
        return categorical_draw(probs)

    def update(self, chosen_arm, reward):
        n_arms = len(self.weights)
        total_weight = sum(self.weights)
        probs = [0.0 for i in range(n_arms)]
        for arm in range(n_arms):
            probs[arm] = (1 - self.gamma) * (self.weights[arm] / total_weight)
            probs[arm] = probs[arm] + (self.gamma) * (1.0 / float(n_arms))

        x = reward / probs[chosen_arm]

        growth_factor = np.exp((self.gamma / n_arms) * x)
        self.weights[chosen_arm] = self.weights[chosen_arm] * growth_factor

    def act(self, observation, reward, done):
        """Make a recommendation"""
        action = self.select_arm()
        self.train(observation, action, reward, done)
        ps_all = np.zeros(self.config.num_products)
        ps_all[action] = 1.0
        return {
            **super().act(observation, reward, done),
            **{
                'a': action,
                'ps': 1.0,
                'ps-a': ps_all,
            },
        }

    def train(self, observation, action, reward, done = False):
        """Train the model in an online fashion"""
        if observation.sessions():
            sessions = observation.sessions()
            for k in range(len(sessions) - 1):
                current = sessions[k]
                expected = sessions[k + 1]
                action = self.select_arm()
                self.update(action, action == expected)
