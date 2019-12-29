import numpy as np

from recogym import Configuration, build_agent_init, to_categorical
from recogym.agents import Agent

test_agent_args = {
    'num_products': 10,
}


class TestAgent(Agent):
    """Organic counter agent"""

    def __init__(self, config = Configuration(test_agent_args)):
        super(TestAgent, self).__init__(config)
        self.unigrams = dict()
        self.probabilities = None
        self.trained = False

    def train(self, observation, action, reward, done = False):
        """Train the model in an online fashion"""
        if observation.sessions():
            sequence = [sess["v"] for sess in observation.sessions()]
            for k in range(len(sequence) - 1):
                curr_item = sequence[k]
                next_item = sequence[k + 1]
                if not curr_item in self.unigrams:
                    self.unigrams[(curr_item)] = np.zeros((self.config.num_products))
                self.unigrams[(curr_item)][next_item] += 1

    def act(self, observation, reward, done):
        """Make a recommendation"""
        if not self.trained:
            self.probabilities = {key: value / value.sum() for key, value in self.unigrams.items()}
            self.trained =  True
        self.update_lpv(observation)
        history = [self.probabilities[sess['v']].argmax() for sess in observation.sessions()]
        action = self.probabilities[self.last_product_viewed].argmax()
        if len(history) > 1:
            print("Using history")
            values, counts = np.unique(history, return_counts=True)
            ind = np.argmax(counts)
            action = values[ind]
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

    def update_lpv(self, observation):
        """updates the last product viewed based on the observation"""
        if observation.sessions():
            self.last_product_viewed = observation.sessions()[-1]['v']
