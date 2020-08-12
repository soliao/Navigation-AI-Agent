# DeepRL-navigation-bananas-udacity-drlnd-p1

- Reinforcement learning environment by Unity ML-Agents
- This corresponds to __Project #1__ of Udacity's Deep Reinforcement Learning Nanodegree (drlnd)\
  https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893
  
In this project, we used the deep Q-learning (DQN) algorithm to train an agent to navigate in an arena filled with yellow and purple bananas.
The goal is to collect the yellow bananas as many as possible, while avoiding the purple ones.\

The environment is originally from Unity Machine Learning Agents (Unity ML-Agents). For more details and other learning environment, please visit:\
https://github.com/Unity-Technologies/ml-agents

For this project we use a slightly different environment provided by Udacity drlnd.

Project Rubrics
Training Code
  - the main code
  - the saved model weights

README
  - project details (s/a spaces, sovling condition, etc.)
  - getting started instructions (installing dependencies, downloading needed files, etc.)
  - instructions on how to run the code to train the agent

Report.md
  - learning algorithm (hyperparameters, model architectures, etc)
  - plot of rewards (goal: average of +13 points over 100 episodes)
  - ideas of future work

## Project details

Number of agents: 1

States: The state space is of 37 dimensions consisting of the agent's velocity
and the ray-based perception of object around the agent's moving direction.

Actions: The action space has 4 dimensions. The agent can choose among:
    * moving forward (0)
    * moving backward (1)
    * turning left (2)
    * turning right (3)
  
Rewards: During the episode, a reward of +1 is provided when the agent collects a yellow
banana, and a reward of -1 is provided when the agent collects a purple banana.

Goal: The goal is to train the agent to reach an average score of +13 over 100 consecutive eposides.



## Getting started

1. Install conda

2. Clone the Deep-Reinforcement-Learning-Nanodegree GitHub Repo:

    https://github.com/udacity/deep-reinforcement-learning#dependencies
  
  For Windows 10 installation encountering "Could not find a version that satisfies the requirement torch==0.4.0 (from unityagents==0.4.0)", please take a look at the following thread:
  
    https://github.com/udacity/deep-reinforcement-learning/issues/13
  
3. Download Unity's Environment or (Udacity's modified environment)
  


## How to run the code
The repo contains:
_xxx_: 
