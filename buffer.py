"""
This file was modified from the coding exercise of Udacity DRLND Lesson 2
"""

import random
import numpy as np

from collections import deque, namedtuple

## Buffer

class Buffer:
    def __init__(self, buffer_size = int(1e6), batch_size = 64, seed = 0):
        self.cache = deque(maxlen = buffer_size)
        self.batch_size = batch_size
        self.replay = namedtuple("Replay", field_names = ["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        # add a tuple (s, a, r, s', /end)
        replay = self.replay(state, action, reward, next_state, done)
        self.cache.append(replay)
    
    def sample(self):
        """
        outputs:
                states: ndarray of shape (BATCH_SIZE, 33)
                actions: ndarray of shape (BATCH_SIZE, 4)
                rewards: ndarray of shape (BATCH_SIZE, 1)
                next_states: ndarray of shape (BATCH_SIZE, 33)
                dones: ndarray of shape (BATCH_SIZE, 1)
        """
        # sample replays of size = batch_size from the buffer
        # output (batch_size x 1) tensors
        replays = random.sample(self.cache, k = self.batch_size)

        states = np.vstack([x.state for x in replays if x is not None])
        actions = np.vstack([x.action for x in replays if x is not None])
        rewards = np.vstack([x.reward for x in replays if x is not None])
        next_states = np.vstack([x.next_state for x in replays if x is not None])
        dones = np.vstack([x.done for x in replays if x is not None])
        
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.cache)