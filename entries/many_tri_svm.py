import numpy as np
import sklearn as sk
import sklearn.svm as sksvm
import skealrn.model_selection as skms

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
        self.model = sksvm.SVC()
        self.distributions = {}
        self.trained = False

    def act(self, observation, reward, done):
        """Make a recommendation"""
        if not self.trained:
            self.sequences = np.array(self.sequences)
            self.labels = np.array(self.labels)
            clf.skms.RandomizedSearchCV(self.model, self.distributions, random_state=0)
            print(self.sequences.shape)
                self.models[k].fit(self.sequences, self.labels)
            self.trained = True
        self.update_lpv(observation)
        action = self.last_product_viewed
        if observation.sessions():
            sequence = [sess["v"] for sess in observation.sessions()]
            if len(sequence) > 2:
                ngram = np.array(sequence[-2:]).reshape(1, -1)
                actions = np.array([model.predict(ngram) for model in self.models]).ravel()
                values, counts = np.unique(actions, return_counts=True)
                action = values[np.argmax(counts)]
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
                for k in range(len(sequence) - 2):
                    self.sequences.append([sequence[k], sequence[k + 1]])
                    self.labels.append(sequence[k + 2])

    def update_lpv(self, observation):
        """updates the last product viewed based on the observation"""
        if observation.sessions():
            self.last_product_viewed = observation.sessions()[-1]['v']
