import numpy as np

class Bandit:
    def __init__(self, mean, seed=3421):
        self.m = mean
        np.random.seed(seed)

    def pull(self):
        return np.random.randn() + self.m
