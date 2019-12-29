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
        self.bigrams = dict()
        self.trigrams = dict()
        self.probabilities = None
        self.trained = False
        self.last_product_viewed = 0 
        self.one_to_last_product_viewed = 0 
        self.second_to_last_product_viewed = 0 

    def train(self, observation, action, reward, done = False):
        """Train the model in an online fashion"""
        if observation.sessions():
            sequence = [sess["v"] for sess in observation.sessions()]
            for k in range(len(sequence) - 1):
                curr_item = sequence[k]
                next_item = sequence[k + 1]
                if not curr_item in self.unigrams:
                    self.unigrams[curr_item] = np.zeros((self.config.num_products))
                self.unigrams[curr_item][next_item] += 1
            for k in range(len(sequence) - 2):
                curr_item = sequence[k]
                next_item = sequence[k + 1]
                next_next_item = sequence[k + 2]
                if not curr_item in self.bigrams:
                    self.bigrams[(curr_item, next_item)] = np.zeros((self.config.num_products))
                self.bigrams[(curr_item, next_item)][next_next_item] += 1
            for k in range(len(sequence) - 3):
                curr_item = sequence[k]
                next_item = sequence[k + 1]
                next_next_item = sequence[k + 2]
                next_next_next_item = sequence[k + 3]
                if not curr_item in self.trigrams:
                    self.trigrams[(curr_item, next_item, next_next_item)] = np.zeros((self.config.num_products))
                self.trigrams[(curr_item, next_item, next_next_item)][next_next_next_item] += 1

    def act(self, observation, reward, done):
        """Make a recommendation"""
        if not self.trained:
            self.unigram_probabilities = {key: value / value.sum() for key, value in self.unigrams.items()}
            self.bigram_probabilities = {key: value / value.sum() for key, value in self.bigrams.items()}
            self.trigram_probabilities = {key: value / value.sum() for key, value in self.trigrams.items()}
            self.trained =  True
        self.update_lpv(observation)

        unigram = (self.last_product_viewed)
        bigram = (self.one_to_last_product_viewed, self.last_product_viewed)
        trigram = (self.second_to_last_product_viewed, self.one_to_last_product_viewed, self.last_product_viewed)
        if not unigram in self.unigram_probabilities:
            unigram_action = self.unigram_probabilities[unigram].argmax()
        else:
            unigram_action = 0
        if not bigram in self.bigram_probabilities:
            bigram_action = unigram_action
        else:
            bigram_action = self.bigram_probabilities[bigram].argmax()
        if not trigram in self.trigram_probabilities:
            trigram_action = 0
        else:
            trigram_action = self.trigram_probabilities[trigram].argmax()
        action = unigram_action
        if trigram_action != unigram_action and bigram_action != unigram_action:
            unigram_prob = self.unigram_probabilities[unigram][unigram_action]
            bigram_prob = self.bigram_probabilities[bigram][bigram_action]
            trigram_prob = self.trigram_probabilities[trigram][trigram_action]
            actions = np.array([unigram_action, bigram_action, trigram_action])
            action = actions[np.argmax([unigram_prob, bigram_prob, trigram_prob])]
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
            if len(observation.sessions()) > 2:
                self.second_to_last_product_viewed = observation.sessions()[-3]['v']
