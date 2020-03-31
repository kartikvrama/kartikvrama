import numpy as np
import matplotlib.pyplot as plt

from bandit import Bandit


class EpsGreedy:
    def __init__(self, bandits, eps):
        self.bandits = bandits
        self.no_bandits = len(self.bandits)
        self.bandit_means = np.zeros(self.no_bandits)
        self.no_updates = np.ones_like(self.bandit_means)
        self.eps = eps

    def update(self):
        rand_gen = np.random.rand()
        if rand_gen < self.eps:
            bandit = np.random.choice(np.arange(self.no_bandits))
        else:
            bandit = np.argmax(self.bandit_means)

        sample = self.bandits[bandit].pull()
        N = self.no_updates[bandit]
        self.bandit_means[bandit] = (1 - 1./N)*self.bandit_means[bandit] \
                                    + (1./N)*sample
        self.no_updates[bandit] += 1
        return self.bandit_means


if __name__ == '__main__':
    true_means = [1.0, 2.0, 3.0]
    no_iterations = int(1e5)
    eps = 0.05

    bandits = [Bandit(i) for i in true_means]
    egreedy_agent = EpsGreedy(bandits=bandits, eps=eps)

    all_means = np.empty([0, len(true_means)])
    for n in range(1, 1 + no_iterations):
        current_means = egreedy_agent.update()
        all_means = np.concatenate([all_means, current_means[np.newaxis, :]],
                                   axis=0)

    print('Final means: ', egreedy_agent.bandit_means)
    plt.plot(range(no_iterations), all_means[:, 0], color='r')
    plt.plot(range(no_iterations), all_means[:, 1], color='g')
    plt.plot(range(no_iterations), all_means[:, 2], color='b')
    plt.show()
