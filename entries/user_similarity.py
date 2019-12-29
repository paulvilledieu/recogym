import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import sklearn.cluster as skclu

from recogym import Configuration, build_agent_init, to_categorical
from recogym.agents import Agent

test_agent_args = {
    'num_products': 10,
}


class TestAgent(Agent):
    """Organic counter agent"""

    def __init__(self, config = Configuration(test_agent_args)):
        super(TestAgent, self).__init__(config)

        self.co_counts = np.zeros((self.config.num_products, self.config.num_products))
        self.corr = None


        self.users = dict()
        self.users_probabilities = dict()
        self.trained = False

    def train(self, observation, action, reward, done = False):
        """Train the model in an online fashion"""
        if observation.sessions():
            user = observation.sessions()[-1]["u"]
            items = [sess["v"] for sess in observation.sessions()]
            if not user in self.users:
                self.users[user] = np.zeros(self.config.num_products)
            for item in items:
                self.users[user][item] += 1

    def act(self, observation, reward, done):
        """Make a recommendation"""
        if not self.trained:
            # Use it to have relative probabilities
            self.users_probabilities = {user: value / value.sum() for user, value in self.users.items()}
            points = np.zeros((len(self.users), self.config.num_products))
            self.clusters = skclu.KMeans(n_clusters=int(len(self.users) * 0.1))
            for idx, (key, value) in enumerate(self.users_probabilities.items()):
                points[idx, :] = np.array(value)
            self.clusters.fit(points)
            self.trained = True
        if observation.sessions():
            user = observation.sessions()[-1]["u"]
            items = [sess["v"] for sess in observation.sessions()]
            user_info = np.zeros(self.config.num_products)
            for item in items:
                user_info[item] += 1
            user_info = user_info / user_info.sum()
        #self.update_lpv(observation)
            action = self.clusters.predict(user_info.reshape(1, -1)).argmax()
        else:
            action = 0 
        #action = self.co_counts[self.last_product_viewed, :].argmax()
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
