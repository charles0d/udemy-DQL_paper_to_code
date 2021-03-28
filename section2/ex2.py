import gym
import numpy as np
import matplotlib.pyplot as plt

# Lake: (S=start, F=frozen, H=hole, G=end)
# SFFF
# FHFH
# FFFH
# HFFG

LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3

env = gym.make('FrozenLake-v0')
win_per10games = []
scores = []
n_games = 1000


def policy(o):
    if o in [0, 2, 4, 6, 9, 10]:
        return DOWN
    if o == 3:
        return LEFT
    return RIGHT


for n_game in range(n_games):
    score = 0
    obs = env.reset()
    done = False
    while not done:
        action = policy(obs)
        obs, reward, done, info = env.step(action)
        score += reward

    scores.append(score)
    if n_game % 10 == 9:
        win_per10games.append(np.mean(scores[-10:]))

plt.plot(win_per10games)
plt.show()
