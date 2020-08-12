"""
This file was modified from Udacity's ddpg-pendulum repository
https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/model.py

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

## Neural Network for DQN

class Network(nn.Module):
    def __init__(self, input_size = 37, output_size = 4, hidden_sizes = [400, 300], seed = 0):
        
        super(Network, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        self.hidden_layers = nn.ModuleList([])
        
        # first layer
        self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[0]))
        
        # second to n-1 layer
        for h1, h2 in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            self.hidden_layers.append(nn.Linear(h1, h2))
            
        # last output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        
        # initialize weights and biases
        self.reset_parameters()
    
    def reset_parameters(self):
        for layer in self.hidden_layers:
            f = layer.weight.data.size()[0] # fan_in
            layer.weight.data.uniform_(-1.0/np.sqrt(f), 1.0/np.sqrt(f))
            layer.bias.data.fill_(0.1)
        self.output_layer.weight.data.uniform_(-3e-3, 3e-3)
        self.output_layer.bias.data.fill_(0.1)
        
    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return self.output_layer(x)
