# DeepRL-navigation-bananas-udacity-drlnd-p1

- Reinforcement learning environment by Unity ML-Agents
- This corresponds to __Project #1__ of Udacity's Deep Reinforcement Learning Nanodegree (drlnd)\
  https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893
  
In this project, we used the deep Q-learning (DQN) algorithm to train an agent to navigate in an arena filled with yellow and purple bananas.
The goal is to collect the yellow bananas as many as possible, while avoiding the purple ones.

The environment is originally from Unity Machine Learning Agents (Unity ML-Agents). For more details and other learning environment, please visit:\
https://github.com/Unity-Technologies/ml-agents

For this project we use a slightly different environment provided by Udacity drlnd.


## Project details

**Number of agents**\
There's only one agent in the environment.

**States**\
The state space is of 37 dimensions consisting of the agent's velocity
and the ray-based perception of object around the agent's moving direction.

**Actions**\
The action space has 4 dimensions. The agent can choose among:
    * moving forward (0)
    * moving backward (1)
    * turning left (2)
    * turning right (3)
  
**Rewards**\
During the episode, a reward of +1 is provided when the agent collects a yellow
banana, and a reward of -1 is provided when the agent collects a purple banana.

**Goal**\
The environment is considered solved when the agent learns to obtain an average score of +13 over 100 consecutive eposides.



## Getting started

1. Install conda\
2. Clone the Deep-Reinforcement-Learning-Nanodegree GitHub Repo\
    https://github.com/udacity/deep-reinforcement-learning#dependencies
  
> (For Windows 10) If the error "Could not find a version that satisfies the requirement torch==0.4.0 (from unityagents==0.4.0)" occurs, please refer to this thread:\
    https://github.com/udacity/deep-reinforcement-learning/issues/13
  
3. Download Unity's Environment or (Udacity's modified environment)\
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
| local_weights.pth | saved weights |

## How to run the code

#### 1. Open `SL_Navigation_DQN.ipynb` with Jupyter Notebook
#### 2. Run `Box 1` to import packages
    In **line 30**, paste the path to `Bananas.exe` after the `file_name = `
    for example, 'file_name = "./Banana_Windows_x86_64/Banana.exe"'
#### 3. Run `Box 2` to set hperparameters
  - `EPISODES` Maximum number of episodes to train
  - `UPDATE_EVERY` sets the number of episodes to update the target network
  - `hidden_sizes` sets the sizes of the hidden layers
  - `gamma` the discount factor
  - `eps_start` the starting epsilon for selecting actions (epsilon-greedy)
  - `eps_end` sets the minimum epsilon
  - `eps_decay` sets the decay factor of the epsilon
  - `lr` the learning rate
  - `tau` soft update hyperparameter
  - `buffer_size` sets the size of the replay buffer
  - `batch_size` sets the number of sampled replays
  - `seed` sets the seed
  
#### 4. Run `Box 3` to start training
    After training, the weights of the network will be saved with the file name `checkpoint.pth`
#### 5. (Optional) Run `Box 4` to load saved weights into the agent and watch it forage for bananas
#### 6. Before closing, simply use the command `env.close()` to close the environment
