# DeepRL-navigation-bananas-udacity-drlnd-p1

- Reinforcement learning environment by Unity ML-Agents
- This repository corresponds to __Project #1__ of Udacity's Deep Reinforcement Learning Nanodegree (drlnd)\
  https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893
  
In this project, we used the deep Q-learning (DQN) algorithm to train an agent to navigate in an arena containing yellow and purple bananas.
The goal is to train the agent to collect the yellow bananas as many as possible, while avoiding the purple ones.

The environment is originally from Unity Machine Learning Agents (Unity ML-Agents). For more details and other learning environments, please visit:\
https://github.com/Unity-Technologies/ml-agents

For this project we use a slightly different environment provided by Udacity drlnd.

[![p1-env-demo.png](https://i.postimg.cc/256zS65t/p1-env-demo.png)](https://postimg.cc/gxC9MG8y)\
**Figure:** The navigation environment by Unity ML-Agents

## Project details

**Number of agents**\
There's only one agent in the environment.

**States**\
The size of the state space is 37 consisting of the agent's velocity
and the ray-based perception of object around the agent's moving direction.

**Actions**\
The action space has 4 dimensions (discrete). At each time step the agent can choose among:
  - moving forward (0)
  - moving backward (1)
  - turning left (2)
  - turning right (3)
  
**Rewards**\
During an episode, a reward of +1 is provided when the agent collects a yellow
banana, and a reward of -1 is provided when the agent collects a purple banana.

**Goal**\
The environment is considered solved when the agent learns to obtain an average score of +13 over 100 consecutive episodes.



## Getting started

1. Install conda
2. Clone the Deep-Reinforcement-Learning-Nanodegree GitHub Repo\
    https://github.com/udacity/deep-reinforcement-learning#dependencies
  
> (For Windows 10) If the error "Could not find a version that satisfies the requirement torch==0.4.0 (from unityagents==0.4.0)" occurs, please refer to this thread:\
    https://github.com/udacity/deep-reinforcement-learning/issues/13
  
3. Download Unity's Environment or (Udacity's modified verson)\
    Windows (32-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip \
    Windows (64-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip


## Contents of files

The repo contains:
| File Name | Notes |
| ----------- | ----------- |
| SL_Navigation_DQN.ipynb | main code |
| networkModels.py | architecture of the neural network |
| buffer.py | replay buffer |
| dqnAgent.py | the agent class |
| checkpoint.pth | saved weights |

## How to run the code

#### 1. Open `SL_Navigation_DQN.ipynb` with Jupyter Notebook
#### 2. Run `Box 1` to import packages
    Paste the path to Bananas.exe after the "file_name = "
    for example, file_name = "./Banana_Windows_x86_64/Banana.exe"
#### 3. Run `Box 2` to set hyperparameters
  - `EPISODES` maximum number of episodes to train
  - `UPDATE_EVERY` frequency of updating the networks
  - `hidden_sizes` sets the sizes of the hidden layers
  - `gamma` the discount factor
  - `eps_start` the starting epsilon of the epsilon-greedy policy
  - `eps_end` sets the terminal epsilon
  - `eps_decay` sets the decay factor of epsilon
  - `lr` the learning rate
  - `tau` soft update hyperparameter
  - `buffer_size` sets the size of the replay buffer
  - `batch_size` sets the size of the batch of replays to sample from the buffer
  - `seed` sets the seed
  
#### 4. Run `Box 3` to start training
    After training, the weights of the network will be saved with the file name `checkpoint.pth`
#### 5. (Optional) Run `Box 4` to load the saved weights into the agent and watch it forage for bananas
#### 6. Before closing, simply use the command `env.close()` to close the environment!
