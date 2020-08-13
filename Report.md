# Report.md

## Learning Algorithm
### Deep Q-network (DQN)
In We use deep network to learn the action value

The DQN algorithm is implemented to solve the project. Two seperate networks were used for local training (train_network) and target (target_network).
 During training, the target network being updated slowly (every UPDATE_EVERY epochs).
- Experience replay
A buffer of size (BUFFER_SIZE = 1e5) is used to stored the tuples (state, action, rewards, next_state, done) from the most recent experieces. Each time of training
a subset (BATCH_SIZE = 64) of the buffer was sampled randomly to train the agent.
- Epsilon greedy strategy
During data collection, we use the epsilon-greedy strategy to choose the action to interact with the environment. During early stages we want the epsilon to be large
(eps_start = 1.0) in favor of exploration. We slowly decreased epsilon by the factor of eps_decay = 0.995 until it reached ips_end = 0.01. By doing so, we gradually 
encourage the agent to exploit the optimal action more frequently.

## Hyperparameters

| Hyperparameter | Value | Description |
| ----------- | ----------- | ----------- |
| hidden_sizes | [400, 300] | units of hidden fully connected layers |
| gamma | 0.99 | discount rate |
| lr | 5e-5 | learning rate |
| tau | 1e-3 | soft update rate <1> |
| UPDATE_EVERY | 5 | frequency of updating target network |
| eps_start | 1.0 | initial epsilon |
| eps_end | 0.01 | terminal epsilon |
| eps_decay | 0.995 | epsilon decay rate <2> |
| buffer_size | 1e6 | size of the replay buffer |
| batch_size | 128 | number of samples to train the Q-network each time |

**<1> (Soft update)** This algorithm slowly updates the target network every `UPDATE_EVERY` steps in an episodes by the soft updating rule:\
`new_target_weights <-- tau * old_local_weights + (1-tau) * old_target_weights`

**<2> (Epsilon-greedy)** Each step the agent evaluates the action values for all 4 possible actions (forward, backward, left and right).
To select an action, the agent uses the epsilon-greedy policy to select the action. We start the training with `eps_start` = 1 (which encourages exploration) and slowly
decrease it by the decay rate `eps_decay` = 0.995 each time, until it reaches `eps_end` = 0.01 (which encourages exploitation).

## Training result

With the parameters above, the agent solved the task after 400 epochs, i.e., the scores from episode # ~ # have average above +13 points.




## Ideas for future work
Future work will be focused on improving the learning by implementing:
**1. Prioritized Experience Replay**
**2. Double DQN**
**3. Actor-Critic Method**

I'm also looking forward to soving the enviornment by using pixels values directly as the input to train the agent. Perhaps I will need to replace the hidden fully-connected layers by some convolutional and pooling layers, or even some dropout layers in my network arcitecture.
