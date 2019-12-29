import numpy as np
import sklearn as sk
import sklearn.naive_bayes as sknb

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from recogym import Configuration, build_agent_init, to_categorical
from recogym.agents import Agent

test_agent_args = {
    'num_products': 10,
}


class TestAgent(Agent):
    """Organic counter agent"""

    def __init__(self, config = Configuration(test_agent_args), max_length=3):
        super(TestAgent, self).__init__(config)
        self.sequences = []
        self.labels = []
        kernel = 1.0 * RBF(1.0)
        self.model = GaussianProcessClassifier(kernel=kernel)
        self.trained = False

    def act(self, observation, reward, done):
        """Make a recommendation"""
        if not self.trained:
            self.sequences = np.array(self.sequences)
            self.labels = np.array(self.labels)
            print(self.sequences.shape)
            self.model.fit(self.sequences, self.labels)
            self.trained = True
        self.update_lpv(observation)
        action = self.last_product_viewed
        if observation.sessions():
            sequence = [sess["v"] for sess in observation.sessions()]
            if len(sequence) > 3:
                ngram = np.array(sequence[-3:]).reshape(1, -3)
                action = self.model.predict(ngram)
                action = action[0]
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
            sequence = [sess["v"] for sess in sessions]
            if len(sequence) > 2:
                for k in range(len(sequence) - 3):
                    self.sequences.append([sequence[k], sequence[k + 1], sequence[k + 2]])
                    self.labels.append(sequence[k + 3])

    def update_lpv(self, observation):
        """updates the last product viewed based on the observation"""
        if observation.sessions():
            self.last_product_viewed = observation.sessions()[-1]['v']
