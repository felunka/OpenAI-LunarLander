import gym
import random
import json
from keras import Sequential
from collections import deque
from keras.layers import Dense, Reshape, Convolution2D, Flatten
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.activations import relu, linear

import numpy as np
env = gym.make('LunarLander-v2')
env.seed(0)
np.random.seed(0)

class Gradient:
  # from https://github.com/keon/policy-gradient

  def __init__(self, observation_space, action_space):
    self.observation_space = observation_space
    self.observation_space_size = observation_space.shape[0]
    self.action_space = action_space

    self.discount_factor = 0.99  # gamma
    self.learning_rate = 0.001  # alpha
    self.states = []
    self.gradients = []
    self.rewards = []
    self.probabilities = []
    self.model = self.build_model()

  def build_model(self):
    model = Sequential()
    model.add(Reshape((1, 1, 8), input_shape=(self.observation_space_size,)))
    model.add(Convolution2D(32, (6, 6), strides=(3, 3), padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(self.action_space.n, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer="adam")
    return model

  def add_memory(self, state, action, probability, reward):
    y = np.zeros([self.action_space.n])
    y[action] = 1
    self.gradients.append(np.array(y).astype('float32') - probability)
    self.states.append(state)
    self.rewards.append(reward)

  def action(self, state):
    state = state.reshape([1, state.shape[0]])
    probabilities = self.model.predict(state, batch_size=1).flatten()
    self.probabilities.append(probabilities)
    average_probability = probabilities / np.sum(probabilities)
    action = np.random.choice(self.action_space.n, 1, p=average_probability)[0]
    return action, average_probability

  def discount_rewards(self, rewards):
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, rewards.size)):
      if rewards[t] != 0:
        running_add = 0
      running_add = running_add * self.discount_factor + rewards[t]
      discounted_rewards[t] = running_add
    return discounted_rewards

  def update(self):
    gradients = np.vstack(self.gradients)
    rewards = np.vstack(self.rewards)
    rewards = self.discount_rewards(rewards)
    rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-7)
    gradients *= rewards
    x = np.squeeze(np.vstack([self.states]))
    y = self.probabilities + self.learning_rate * np.squeeze(np.vstack([gradients]))
    self.model.train_on_batch(x, y)
    self.states, self.probabilities, self.gradients, self.rewards = [], [], [], []

# Prams
episodes = 10000
max_number_of_steps = 500
observation_size = env.observation_space.shape[0]
gradient = Gradient(env.observation_space, env.action_space)

# Recording
loss = []
avg = []

for e in range(episodes):
  current_observation = env.reset()
  previous_observation = None

  done = False
  current_step = 0
  score = 0
  while not done and current_step < max_number_of_steps:
    if previous_observation is None:
      x = np.zeros(observation_size)
    else:
      x = current_observation - previous_observation

    action, probability = gradient.action(x)
    # env.render()
    previous_observation = current_observation
    current_observation, reward, done, _ = env.step(action)
    score += reward
    gradient.add_memory(x, action, probability, reward)
    current_step += 1
    if done:
      print("episode: {}/{}, score: {}".format(e, episodes, score))

  loss.append(score)
  gradient.update()

  is_solved = np.mean(loss[-100:])
  avg.append(is_solved)
  if is_solved > 200:
    print('\n Task Completed! \n')
    break
  print("Average over last 100 episode: {0:.2f} \n".format(is_solved))

# dump
with open('rhc_loss.json', 'w') as outfile:
  json.dump(loss, outfile)
with open('rhc_avg.json', 'w') as outfile:
  json.dump(avg, outfile)

# Plot loss
plt.plot([i+1 for i in range(0, len(loss), 2)], loss[::2])
plt.show()
# Plot avg
plt.plot([i+1 for i in range(0, len(avg), 2)], avg[::2])
plt.show()

# recordings at 0, 380, 2500, 6700, 8000
