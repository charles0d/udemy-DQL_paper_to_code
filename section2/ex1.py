import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')
win_per10games = []
reward_history = []
n_games = 1000
for n_game in range(n_games):
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

    reward_history.append(reward)
    if n_game % 10 == 9:
        win_per10games.append(np.mean(reward_history[-10:]))

plt.plot(win_per10games)
plt.show()
