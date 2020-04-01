import numpy as np
import matplotlib.pyplot as plt

from bandit import Bandit


class EpsGreedy:
    def __init__(self, bandits, eps):
        self.bandits = bandits
        self.no_bandits = len(self.bandits)
        self.bandit_means = np.zeros(self.no_bandits)
        self.no_updates = np.zeros_like(self.bandit_means)
        self.eps = eps

    def update(self):
        rand_gen = np.random.rand()
        if rand_gen < self.eps:
            bandit = np.random.choice(np.arange(self.no_bandits))
        else:
            bandit = np.argmax(self.bandit_means)

        sample = self.bandits[bandit].pull()
        nj = self.no_updates[bandit]
        self.bandit_means[bandit] = (1 - 1./(nj + 1))*self.bandit_means[bandit] \
                                    + (1./(nj + 1))*sample
        self.no_updates[bandit] += 1
        return sample, self.bandit_means


if __name__ == '__main__':
    true_means = [1.0, 2.0, 3.0]
    no_iterations = int(1e5)
    bandits = [Bandit(i) for i in true_means]

    eps = 0.01
    egreedy_agent = EpsGreedy(bandits=bandits, eps=eps)

    all_returns = []
    for n in range(no_iterations):
        x, _ = egreedy_agent.update()
        all_returns.append(x)

    all_returns = np.cumsum(all_returns)/(np.arange(len(all_returns)) + 1)
    print('Final means for {}: '.format(eps), egreedy_agent.bandit_means)
    plt.plot(range(no_iterations), all_returns, color='r', label=str(eps))

    eps = 0.05
    egreedy_agent = EpsGreedy(bandits=bandits, eps=eps)

    all_returns = []
    for n in range(no_iterations):
        x, _ = egreedy_agent.update()
        all_returns.append(x)

    all_returns = np.cumsum(all_returns)/(np.arange(len(all_returns)) + 1)
    print('Final means for {}: '.format(eps), egreedy_agent.bandit_means)
    plt.plot(range(no_iterations), all_returns, color='b', label=str(eps))

    eps = 0.1
    egreedy_agent = EpsGreedy(bandits=bandits, eps=eps)

    all_returns = []
    for n in range(no_iterations):
        x, _ = egreedy_agent.update()
        all_returns.append(x)

    all_returns = np.cumsum(all_returns)/(np.arange(len(all_returns)) + 1)
    print('Final means for {}: '.format(eps), egreedy_agent.bandit_means)
    plt.plot(range(no_iterations), all_returns, color='g', label=str(eps))
    plt.legend()
    plt.show()
