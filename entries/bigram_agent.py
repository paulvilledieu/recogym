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
        self.bigrams = dict()
        self.probabilities = None
        self.trained = False
        self.last_product_viewed = 0 
        self.one_to_last_product_viewed = 0 

    def train(self, observation, action, reward, done = False):
        """Train the model in an online fashion"""
        if observation.sessions():
            sequence = [sess["v"] for sess in observation.sessions()]
            for k in range(len(sequence) - 2):
                curr_item = sequence[k]
                next_item = sequence[k + 1]
                next_next_item = sequence[k + 2]
                if not curr_item in self.bigrams:
                    self.bigrams[(curr_item, next_item)] = np.zeros((self.config.num_products))
                self.bigrams[(curr_item, next_item)][next_next_item] += 1

    def act(self, observation, reward, done):
        """Make a recommendation"""
        if not self.trained:
            self.probabilities = {key: value / value.sum() for key, value in self.bigrams.items()}
            self.trained =  True
        self.update_lpv(observation)

        bigram = (self.one_to_last_product_viewed, self.last_product_viewed)
        if not bigram in self.probabilities:
            action = 0
        else:
            action = self.probabilities[bigram].argmax()
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
            if len(observation.sessions()) > 1:
                self.one_to_last_product_viewed = observation.sessions()[-2]['v']
