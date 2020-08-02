# Report.md

## Learning Algorithm
- Deep Q-network (DQN)
The DQN algorithm is implemented to solve the project. Two seperate networks were used for local training (train_network) and target (target_network).
 During training, the target network being updated slowly (every UPDATE_EVERY epochs).
- Experience replay
A buffer of size (BUFFER_SIZE = 1e5) is used to stored the tuples (state, action, rewards, next_state, done) from the most recent experieces. Each time of training
a subset (BATCH_SIZE = 64) of the buffer was sampled randomly to train the agent.
- Epsilon greedy strategy
During data collection, we use the epsilon-greedy strategy to choose the action to interact with the environment. During early stages we want the epsilon to be large
(eps_start = 1.0) in favor of exploration. We slowly decreased epsilon by the factor of eps_decay = 0.995 until it reached ips_end = 0.01. By doing so, we gradually 
encourage the agent to exploit the optimal action more frequently.

Hyperparameters

BUFFER_SIZE = 10,000    (size of the buffer)

BATCH_SIZE = 64         (number of tuples sampled from the buffer to train the agent)

GAMMA = 0.99            ()

## Plot of Reward
With the above settings, the agent mastered the task after epochs

## Ideas for future work
