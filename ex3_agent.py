import random


class Agent:
    def __init__(self, gamma, lr, eps0, eps_infty, eps_decrease, n_actions, n_states):
        self.n_states = n_states
        self.n_actions = n_actions
        self.epsilon = eps0
        self.eps_infty = eps_infty
        self.eps_decrease = eps_decrease
        self.lr = lr
        self.gamma = gamma
        self.q = dict()
        state_actions_list = [(s, a) for s in range(n_states) for a in range(n_actions)]
        for (s, a) in state_actions_list:
            self.q[(s, a)] = 0.0

    def choose_action(self, state):
        self.epsilon = max(self.eps_infty, self.epsilon - self.eps_decrease)
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions-1)
        return self.find_max_action(state)

    def learn(self, action, state, new_state, reward):
        a_star = self.find_max_action(new_state)
        self.q[(state, action)] += self.lr * (reward + self.gamma * self.q[(new_state, a_star)]
                                              - self.q[(state, action)])

    def find_max_action(self, state):
        action = 0
        value = 0
        for a in range(self.n_actions):
            if self.q[(state, a)] > value:
                action = a
                value = self.q[(state, a)]
        return action
