import numpy as np


class AutoregressiveEnvironment:
    def __init__(self, n_rounds, gamma, k, noise_std=0.01, X0=None, random_state=1):
        self.n_rounds = n_rounds
        self.gamma = gamma
        self.noise_std = noise_std
        self.k = k
        self.X0 = X0
        self.random_state = random_state
        self.reset()

    def round(self, a):
        # Compute states
        if self.k > 0:
            obs_X = sum([self.X[-i]*self.gamma[a][i]
                        for i in range(1, self.k+1)]) + \
                        self.gamma[a][0] + self.noise[self.t]
        else:
            obs_X = self.gamma[a][0] + self.noise[self.t]
        self.X = np.append(self.X, obs_X)
        self.t += 1

    def reset(self, i=0):
        self.t = 0
        if self.X0:
            self.X = self.X0
        else:
            self.X = np.zeros(self.k+1)
        np.random.seed(self.random_state+i)
        self.noise = np.random.normal(0, self.noise_std, self.n_rounds) 
        return self
