import numpy as np
from numpy.random import choice
from recogym.agents import Agent

test_agent_args = {
    'num_products': 10,
}

# Define an Agent class.
class TestAgent(Agent):
    def __init__(self, config):
        # Set number of products as an attribute of the Agent.
        super(TestAgent, self).__init__(config)

        # Track number of times each item viewed in Organic session.
        self.organic_views = np.zeros(self.config.num_products)

    def train(self, observation, action, reward, done):
        """Train method learns from a tuple of data.
            this method can be called for offline or online learning"""

        # Adding organic session to organic view counts.
        if observation:
            for session in observation.sessions():
                self.organic_views[session['v']] += 1

    def act(self, observation, reward, done):
        """Act method returns an action based on current observation and past
            history"""

        # Choosing action randomly in proportion with number of views.
        prob = self.organic_views / sum(self.organic_views)
        action = choice(self.config.num_products, p = prob)

        return {
            **super().act(observation, reward, done),
            **{
                'a': action,
                'ps': prob[action]
            }
        }
