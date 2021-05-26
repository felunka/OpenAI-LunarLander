import gym
import random
import json
from keras import Sequential
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.activations import relu, linear

import numpy as np
env = gym.make('LunarLander-v2')
env.seed(0)
np.random.seed(0)

# https://towardsdatascience.com/solving-lunar-lander-openaigym-reinforcement-learning-785675066197
class DQN:
  """ Implementation of deep q learning algorithm """

  def __init__(self, action_space, state_space):
    self.action_space = action_space
    self.state_space = state_space
    self.epsilon = 1.0
    self.gamma = .99
    self.batch_size = 64
    self.epsilon_min = .01
    self.lr = 0.001
    self.epsilon_decay = .996
    self.memory = deque(maxlen=1000000)
    self.model = self.build_model()

  def build_model(self):
    model = Sequential()
    model.add(Dense(150, input_dim=self.state_space, activation=relu))
    model.add(Dense(120, activation=relu))
    model.add(Dense(self.action_space, activation=linear))
    model.compile(loss='mse', optimizer=Adam(lr=self.lr))
    return model

  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))

  def act(self, state):
    if np.random.rand() <= self.epsilon:
        return random.randrange(self.action_space)
    act_values = self.model.predict(state)
    return np.argmax(act_values[0])

  def replay(self):
    if len(self.memory) < self.batch_size:
      return

    minibatch = random.sample(self.memory, self.batch_size)
    states = np.array([i[0] for i in minibatch])
    actions = np.array([i[1] for i in minibatch])
    rewards = np.array([i[2] for i in minibatch])
    next_states = np.array([i[3] for i in minibatch])
    dones = np.array([i[4] for i in minibatch])

    states = np.squeeze(states)
    next_states = np.squeeze(next_states)

    targets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
    targets_full = self.model.predict_on_batch(states)
    ind = np.array([i for i in range(self.batch_size)])
    targets_full[[ind], [actions]] = targets

    self.model.fit(states, targets_full, epochs=1, verbose=0)
    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay


def train_dqn(episode):
  loss = []
  avg = []
  agent = DQN(env.action_space.n, env.observation_space.shape[0])
  for e in range(episode):
    state = env.reset()
    state = np.reshape(state, (1, 8))
    score = 0
    max_steps = 3000
    for i in range(max_steps):
      action = agent.act(state)
      env.render()
      next_state, reward, done, _ = env.step(action)
      score += reward
      next_state = np.reshape(next_state, (1, 8))
      agent.remember(state, action, reward, next_state, done)
      state = next_state
      agent.replay()
      if done:
        print("episode: {}/{}, score: {}".format(e, episode, score))
        break
    loss.append(score)

    # Average score of last 100 episode
    is_solved = np.mean(loss[-100:])
    avg.append(is_solved)
    if is_solved > 200:
      print('\n Task Completed! \n')
      break
    print("Average over last 100 episode: {0:.2f} \n".format(is_solved))
  # Plot avg
  plt.plot([i+1 for i in range(0, len(avg), 2)], avg[::2])
  plt.show()

  # dump
  with open('dqn_loss.json', 'w') as outfile:
    json.dump(loss, outfile)
  with open('dqn_avg.json', 'w') as outfile:
    json.dump(avg, outfile)
  return loss


if __name__ == '__main__':
  print(env.observation_space)
  print(env.action_space)
  episodes = 400
  loss = train_dqn(episodes)
  plt.plot([i+1 for i in range(0, len(loss), 2)], loss[::2])
  plt.show()

# episode: 364/400, score: 252.3769125805014
# Average over last 100 episode: 198.88
# 
# episode: 365/400, score: 275.1089156449221
# Average over last 100 episode: 199.20
# 
# episode: 366/400, score: 234.95311973015984
# Average over last 100 episode: 199.08
# 
# episode: 367/400, score: 284.25774986650066
# Average over last 100 episode: 199.26
# 
# episode: 368/400, score: 289.02234192240275
# Average over last 100 episode: 199.76
# 
#  Task Completed!
