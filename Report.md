# Report.md

## Deep Q-network (DQN)

**The original DQN paper**

https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

**The basics**

We use deep Q-learning to solve this project. DQN is a type of value-based methods in reinforcement learning, where deep artificial neural networks are trained to learn the action values (Q). The updating rule of the action values is similar to the temporal-difference (TD) learning algorithm, where:

`Q(s, a) <-- Q(s, a) + alpha * [r + gamma * max_{a}Q(s', a) - Q(s, a)]`

where max_{a}Q(s', a) means that we chose the maximum Q(s', a) among all possible actions a\

A slight difference in DQN is that here we use 2 seperate Q-networks: a local network and a target network, with the updating rule:

`Q_local(s, a) <-- Q_local(s, a) + alpha * [r + gamma * max_{a}Q_target(s', a) - Q_local(s, a)]`

where we frequently update the weights of Q_local every several (`UPDATE_EVERY`) time steps, and soft update the weights of Q_target.

**The algorithm**

Below is a brief descriptions of the DQN routine:

- initialize two networks (Q_local and Q_target) with identical weights
- t_step = 0
- observe the initial state s (size = 37) from the environment
- while the environment is not solved:
  - t_step += 1
  - use Q_local to calculate the actions values (for all 4 actions)
  - the agent use epsilon-greedy policy to take an action a
  - the agent collects the reward r and enters the next state (s')
  - add the experience tuple (s, a, r, s') into the replay buffer
  - if there enough number (>= `batch_size`) of replays in the buffer:
    - randomly sample a batch of size = `batch_size` replay tuples (s, a, r, s') from the buffer
    - use Q_target to calculate predicted action value `max_{a'}Q_target(s', a')`
    - use Q_local to calculate action value Q_local(s, a)
    - use gradient decent to update the weights of Q_local by minimizing the TD error\
    `r + gamma * max_{a}Q_target(s', a) - Q_local(s, a)`
    - soft-update the weights of Q_target\
    `new_Q_target_weights <-- tau * old_Q_local_weights + (1-tau) * old_Q_target_weights`
  - s <- s'


## Hyperparameters

| Hyperparameter | Value | Description |
| ----------- | ----------- | ----------- |
| hidden_sizes | [400, 300] | units of hidden fully connected layers |
| gamma | 0.99 | discount rate |
| lr | 1e-4 | learning rate |
| tau | 1e-3 | soft update rate <1> |
| UPDATE_EVERY | 10 | frequency of updating Q network |
| eps_start | 1.0 | initial epsilon |
| eps_end | 0.01 | terminal epsilon |
| eps_decay | 0.995 | epsilon decay rate <2> |
| buffer_size | 1e5 | size of the replay buffer |
| batch_size | 64 | number of samples to train the Q-network each time |

**<1> (Soft update)** This algorithm slowly updates the target network every `UPDATE_EVERY` steps in an episodes by the soft updating rule:\
`new_target_weights <-- tau * old_local_weights + (1-tau) * old_target_weights`

**<2> (Epsilon-greedy)** Each step the agent evaluates the action values for all 4 possible actions (forward, backward, left and right).
To select an action, the agent uses the epsilon-greedy policy to select the action. We start the training with `eps_start` = 1 (which encourages exploration) and slowly
decrease it by the decay rate `eps_decay` = 0.995 each time, until it reaches `eps_end` = 0.01 (which encourages exploitation).

## Training result

With the parameters above, the agent solved the task after 483 episodes, i.e., the average score from episode #484 to #583 reaches above +13 points.

[![p1-scores.png](https://i.postimg.cc/vTLRzFB9/p1-scores.png)](https://postimg.cc/8f5npYLP)

## Ideas for future work
Future work will be focused on improving the learning by trying to implement the following methods:\
**1. Prioritized Experience Replay**\
This method imporves learning by sampling the replays that have higher TD error more often than the replays that have smaller TD error.\
**2. Double DQN**\
Q-learning is prone to overestimating the Q-values. Double DQN replaces the TD target `r + gamma * max_{a}Q_target(s', a)` by\
`r + gamma * max_{a' selected from Q_local} Q_target(s', a')`\
that is, we select the action a' that yields that maximum action value which Q_local(s', a') produces, and feed that a' to Q_target to calculate the target action value Q_target(s', a')\

I'm also looking forward to soving the enviornment by using pixels values directly as the input to train the agent. Perhaps I will need to replace the hidden fully-connected layers by some convolutional and pooling layers, or even some dropout layers in my network arcitecture.
