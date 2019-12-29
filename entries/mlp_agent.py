import numpy as np
from numpy.random import choice
import pandas as pd

from recogym import Configuration, build_agent_init, to_categorical
from recogym.agents import Agent

test_agent_args = {
    'num_products': 10,
}


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class FeedForward(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(FeedForward, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, num_classes)
            #self.softmax = torch.nn.Softmax()

        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            output = self.fc2(relu)
            #output = self.softmax(output)
            return output


def get_model(max_length, hidden, num_classes):
    net = FeedForward(max_length, hidden, num_classes)
    return net

def criterion(out, label):
    return (label - out)**2

def train_model(model, x_train, y_train, optimizer, criterion):
    for epoch in range(100):
        for idx, (x, y) in enumerate(zip(x_train, y_train)):
            print(f"ID({id}), X = {x}, y = {y}")
            x = Variable(torch.FloatTensor([x]), requires_grad=True)
            y = Variable(torch.FloatTensor([y]), requires_grad=False)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred.squeeze(), y)
            loss.backward()
            optimizer.step()
            if (i % 10 == 0):
                print("Epoch {} - loss: {}".format(epoch, loss.data[0]))

class TestAgent(Agent):
    def __init__(self, config, max_length=10, epochs=100):
        # Set number of products as an attribute of the Agent.
        super(TestAgent, self).__init__(config)

        # Track number of times each item viewed in Organic session.
        self.user_items = pd.DataFrame(columns=range(self.config.num_products))
        
        self.organic_views = np.zeros(self.config.num_products)
        self.nb_sessions = 0    #count the sessions
        self.training_rate = 10 #train each training_rate session to allow scaling
        self.max_length = max_length
        self.epochs = epochs
        self.model = get_model(self.max_length, 256, self.config.num_products)
        self.trained = False
        self.sequences = []
        self.labels = []

    def train(self, observation, action, reward, done):
        # Adding organic session to organic view counts.
        if observation:
            for session in observation.sessions():
                self.organic_views[session['v']] += 1


        if observation is not None and len(observation.sessions()) > 0:
            user = observation.current_sessions[-1]['u']
            if not user in self.user_items.index:
                self.user_items.loc[user] = np.zeros((self.config.num_products))
            seq = np.zeros(self.max_length)
            idx = 0
            sessions = observation.sessions()
            #print(sessions)
            for elt in sessions[-(self.max_length + 1):-1]:  
                item = elt['v'] # Because we need 0 as a padding value
                seq[idx] = item + 1
                idx += 1 # We want to update the position, but we want to keep last
                self.user_items.loc[[user], [item]] += 1
            idx = min(self.max_length - 1, idx)
            label = sessions[-1]["v"]
            #print(seq, "=", label)

            seq[idx] = 0 # set to mask to remove last
            self.sequences.append(seq)
            self.labels.append(label)

        self.nb_sessions += 1

    def act(self, observation, reward, done):
        """Act method returns an action based on current observation and past
            history"""
        if not self.trained:
            x_train = torch.FloatTensor(self.sequences)
            #print(x_train.size())
            y_train = torch.LongTensor(self.labels)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.01)   
            self.model.train()
            for epoch in range(self.epochs):
                optimizer.zero_grad()    # Forward pass
                y_pred = self.model(x_train)    # Compute Loss
                loss = criterion(y_pred.squeeze(), y_train)
                #print('Epoch {}: train loss: {}'.format(epoch, loss.item()))    # Backward pass
                loss.backward()
                optimizer.step()
            self.model.eval()
            self.trained = True
            
            
        prob = self.organic_views / sum(self.organic_views)

        if observation is not None and len(observation.current_sessions) > 0:# and self.kmeans:
            history = np.zeros((self.max_length))
            #history = []
            sessions = observation.sessions()
            for idx, elt in enumerate(sessions[-(self.max_length):]):  
                item = elt['v'] # Because we need 0 as a padding value
                history[idx] = item + 1
            action = self.model(torch.tensor(history,  dtype=torch.float32, requires_grad=False))
            
            #history = history.reshape((1, history.shape[0]))#reshape because single data prediction
        
            #pred = self.kmeans.predict(history)
            #centroid = self.kmeans.cluster_centers_[pred]
            action = np.argmax(action.detach().numpy())

        else:
            action = choice(self.config.num_products, p = prob)

        return {
            **super().act(observation, reward, done),
            **{
                'a': action,
                'ps': 1#prob[action]
            }
        }

        def __str__(self):
            return "User Similarity Agent"

        def __repr(self):
            return str(self)
