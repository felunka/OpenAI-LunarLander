import gym
import json
from math import floor
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('LunarLander-v2')
env.seed(0)
np.random.seed(0)

class RandomHillClimbing:
  def __init__(self, dim=8):  
    self.dim = dim
    self.param = self._resample()
    self.max_reward = 0.0
    self.epsilon = 0.1
    self.best_param = self.param
    self.best_list = []
    self.final_param = None

  def _resample(self):
    return 2 * np.random.rand(self.dim) - 1

  def restart(self):
    print(self.max_reward)

    self.best_list.append((self.max_reward, self.best_param))
    self.param = self._resample()
    self.best_param = self.param
    self.max_reward = 0

  def action(self, observation):
    return floor(np.dot(observation, self.param)) % 4

  def final_action(self, observation):
    if self.final_param is None:
      self.best_list.append((self.max_reward, self.best_param))
      self.final_param = max(self.best_list, key=lambda item: item[0])[1]

    return floor(np.dot(observation, self.final_param)) % 4

  def update(self, history):
    total_reward = np.sum([h['reward'] for h in history])
    best_updated = False
    if total_reward > self.max_reward:
      self.max_reward = total_reward
      self.best_param = self.param
      best_updated = True
    else:
      self.param = self.best_param
    self.param += self.epsilon * self._resample()
    return best_updated

# Prams
episodes = 10000
restart_limit = 5000
random_hill_climb = RandomHillClimbing()

# Recording
rounds_without_change = 0
loss = []
avg = []

for e in range(episodes):
  if rounds_without_change > restart_limit:
    random_hill_climb.restart()
    rounds_without_change = 0

  done = False
  observation = env.reset()
  history = []
  score = 0
  while not done:
    action = random_hill_climb.action(observation)
    # env.render()
    observation, reward, done, _ = env.step(action)
    score += reward
    history.append({'reward': reward})

  print("episode: {}/{}, score: {}".format(e, episodes, score))

  if random_hill_climb.update(history):
    rounds_without_change = 0
  else:
    rounds_without_change += 1

  loss.append(score)

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
