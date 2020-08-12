import random
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from networkModels import Network
from buffer import Buffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


## DQN Agent

class DQN_Agent:
    
    def __init__(self, input_size = 37, output_size = 4, hidden_sizes = [400, 300], seed = 0, lr = 1e-3, gamma = 0.99, tau = 1e-3):
        
        self.seed = random.seed(seed)
        
        self.state_size = input_size
        self.action_size = output_size
        
        self.gamma = gamma
        self.tau = tau

        # DQN networks (1 for training, 1 for target)
        self.local = Network(input_size, output_size, hidden_sizes, seed).to(device)
        self.target = Network(input_size, output_size, hidden_sizes, seed).to(device)

        # optimizer
        self.optimizer = optim.Adam(self.local.parameters(), lr = lr)
        
        # loss
        self.loss = 0
        
        
    def act_eps(self, state_t, eps):
        """
            take action with epsilon greedy
            
            state_t: state tensor of shape (m, 37)
            noise_np - ndarray of shape (m, 4)
        """
        self.local.eval()
        with torch.no_grad():
            action = self.local(state_t).detach().cpu().numpy()
        self.local.train()
        
        # epsilon greedy
        if random.random() > eps:
            return np.argmax(action)
        else:
            return random.randint(0, self.action_size-1)
    
   
    def act(self, state_t):
        """
            state_t: state tensor of shape (m, 37)
            noise_np - ndarray of shape (m, 4)
        """
        self.local.eval()
        with torch.no_grad():
            action = self.local(state_t).detach().cpu().numpy()
        self.local.train()
        
        return np.argmax(action)
    
    
    def learn(self, replays):
        
        states, actions, rewards, next_states, dones = replays
        
        #print("states")
        #print(states)
        #print("actions")
        #print(actions)
        #print("rewards")
        #print(rewards)
        #print("next_states")
        #print(next_states)
        #print("dones")
        #print(dones)
        
        states_t = torch.from_numpy(states).float().to(device)
        actions_t = torch.from_numpy(actions).to(device)
        next_states_t = torch.from_numpy(next_states).float().to(device)
        rewards_t = torch.from_numpy(rewards).float().to(device)
        dones_t = torch.from_numpy(dones.astype(np.uint8)).float().to(device)
        
        local_Q = self.local(states_t).gather(1, actions_t.long())
        #print("local_Q")
        #print(local_Q)
        
        target_Q = self.target(next_states_t).detach().max(1)[0].unsqueeze(1)
        
        #print("target_Q")
        #print(target_Q)
        
        loss = F.mse_loss(local_Q, rewards_t + self.gamma*target_Q*(1-dones_t))
        
        self.loss = loss.clone().detach().cpu().numpy()
        
        #print("loss")
        #print(self.loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # soft-update the target network
        self.soft_update()
       
    
    def soft_update(self):
        for local_param, target_param in zip(self.local.parameters(), self.target.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        
      
    
    
    