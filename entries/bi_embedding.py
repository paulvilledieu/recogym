import numpy as np

from recogym import Configuration, build_agent_init, to_categorical
from recogym.agents import Agent

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import model

test_agent_args = {
    'num_products': 10,
}

class Embedder(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size=2, feature_dim=128):
        super(Embedder, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, feature_dim)
        self.linear2 = nn.Linear(feature_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)

def get_model(vocab_size, embedding_dim):
    return Embedder(vocab_size, embedding_dim)

class TestAgent(Agent):
    """Organic counter agent"""

    def __init__(self, config = Configuration(test_agent_args), epochs=100):
        super(TestAgent, self).__init__(config)
        self.sequences = []
        self.labels = []
        self.trained = False
        self.epochs = epochs

    def train(self, observation, action, reward, done = False):
        """Train the model in an online fashion"""
        if observation.sessions():
            sequence = [sess["v"] for sess in observation.sessions()]
            if len(sequence) > 3:
                self.sequences.append([([sequence[i], sequence[i + 1]], sequence[i + 2]) for i in range(len(sequence) - 2)])
                self.labels.append(sequence[i + 3])

    def act(self, observation, reward, done):
        """Make a recommendation"""
        if not self.trained:
            self.sequences = np.array(self.sequences)
            self.labels = np.array(self.labels)
            ids = np.random.permutation(len(self.sequences))
            train_size = int(len(ids) * 0.8)
            x_train = self.sequences[ids, :][:train_size]
            x_val  = self.sequences[ids, :][train_size:]

            y_train = self.labels[ids][:train_size]
            y_val = self.labels[ids][train_size:]

            x_train = torch.LongTensor(x_train)
            x_val = torch.LongTensor(x_val)

            y_train = torch.LongTensor(y_train)
            y_val = torch.LongTensor(y_val)

            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.model.parameters())
            model.train_model(self.model, x_train, y_train, x_val, y_val,
                              optimizer, criterion, epochs=self.epochs,
                              verbose=2)
            self.trained = True
        self.update_lpv(observation)

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
