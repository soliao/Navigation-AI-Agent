# Navigation AI Agent

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

Please follow the steps below to download all the necessary files and dependencies.

1. Install Anaconda (with Python 3.x)\
    https://www.anaconda.com/products/individual
    
2. Create (if you haven't) a new environment with Python 3.6 by typing the following command in the Anaconda Prompt:\
    `conda create --name drlnd python=3.6`
    
3. Install (need only minimal install) `gym` by following the **Installation** section of the OpenAI Gym GitHub:
    https://github.com/openai/gym#id5
    
4. Clone the repository from Udacity's drlnd GitHub
    ``` console
    git clone https://github.com/udacity/deep-reinforcement-learning.git
    cd deep-reinforcement-learning/python
    pip install .
    ```
    (For Windows) If the error "Could not find a version that satisfies the requirement torch==0.4.0 (from unityagents==0.4.0)" occurs, please refer to this thread:\
    https://github.com/udacity/deep-reinforcement-learning/issues/13
  
5. Download the Navigation Environment (Udacity's modified version)\
    Linux: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip \
    Mac OSX: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip \
    Windows (32-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip \
    Windows (64-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip
    
    Extract the .zip file and move the folder `Banana_Windows_x86_64` (or `Banana`, `Banana_Linux`, `Banana_Windows_x86`, depending on the operating system) into the folder `p1_navigation` from Step 4.

6. Download all the files (see the table below) from this repository. Place all files in the folder `p1_navigation` from Step 4.

    | File Name | Notes |
    | ----------- | ----------- |
    | SL_Navigation_DQN.ipynb | main code |
    | networkModels.py | architecture of the neural network |
    | buffer.py | replay buffer |
    | dqnAgent.py | the agent class |
    | checkpoint.pth | saved weights |

7. You're ready to run the code! Please see the next section.

## How to run the code

Please follow the steps below to train the agent or to watch the pre-trained agent perform the task.

#### 1. Run the *Anaconda Prompt* and navigate to the folder `p1_navigation`
``` cmd
cd /path/to/the/p1_navigation/folder
```
#### 2. Activate the drlnd environment
``` cmd
conda activate drlnd
```
#### 3. Run the Jupter Notebook
``` cmd
jupyter notebook
```
#### 4. Open `SL_Navigation_DQN.ipynb` with Jupyter Notebook
#### 5. Run `Box 1` to import packages
Paste the path to `Bananas.exe` after **"file_name = "**

for example, `file_name = "./Banana_Windows_x86_64/Bananas.exe"`
#### 6. Run `Box 2` to set the hyperparameters
For information of the hyperparameters, please refer to `Report.md`
#### 7. Run `Box 3` to start training
After training, the weights of the network will be saved with the file name `checkpoint.pth`
#### 8. (Optional) Run `Box 4` to load the saved weights into the agent and watch it forage for bananas
#### 9. Before closing, simply use the command `env.close()` to close the environment!
