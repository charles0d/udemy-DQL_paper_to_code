import gym
from ex3_agent import Agent
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    N_GAMES = 500000

    env = gym.make('FrozenLake-v0')

    agent = Agent(gamma=0.9, lr=0.001, eps0=1, eps_infty=0.01, eps_decrease=2e-7,
                  n_actions=env.action_space.n, n_states=env.observation_space.n)
    scores = []
    avg_scores = []

    for n_game in range(N_GAMES):
        obs = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.choose_action(obs)
            obs_new, reward, done, info = env.step(action)
            agent.learn(action, obs, obs_new, reward)
            score += reward
            obs = obs_new

        scores.append(score)
        if (n_game+1) % 100 == 0:
            avg_scores.append(np.mean(scores[-100:]))

        if (n_game+1) % 5000 == 0:
            print(f'episode {n_game+1}. Average score: {avg_scores[-1]}. epsilon = {agent.epsilon:.2f}')

    plt.plot(avg_scores)
    plt.show()
