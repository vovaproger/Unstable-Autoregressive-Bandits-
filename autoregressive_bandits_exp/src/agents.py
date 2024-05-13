from abc import ABC, abstractmethod
import numpy as np
from random import Random


class Agent(ABC):
    def __init__(self, n_arms, random_state=1):
        self.n_arms = n_arms
        self.arms = np.arange(self.n_arms)
        self.t = 0
        self.a_hist = []
        self.last_pull = None
        np.random.seed(random_state)
        self.randgen = Random(random_state)

    @abstractmethod
    def pull_arm(self):
        pass

    @abstractmethod
    def update(self, X, *args, **kwargs):
        pass

    def reset(self):
        self.t = 0
        self.last_pull = None


class Exp3Agent(Agent):
    def __init__(self, n_arms, gamma=1, max_reward=1, random_state=1):
        super().__init__(n_arms, random_state)
        self.gamma = gamma
        self.max_reward = max_reward
        self.reset()

    def reset(self):
        super().reset()
        self.w = np.ones(self.n_arms)
        self.est_rewards = np.zeros(self.n_arms)
        self.probabilities = (1/self.n_arms)*np.ones(self.n_arms)
        self.probabilities[0] = 1 - sum(self.probabilities[1:])
        return self

    def pull_arm(self):
        new_a = np.random.choice(self.arms, p=self.probabilities, size=None)
        self.last_pull = new_a
        self.a_hist.append(new_a)
        return new_a

    def update(self, X):
        X = X/self.max_reward
        self.est_rewards[self.last_pull] = X/self.probabilities[self.last_pull]
        self.w[self.last_pull] *= np.exp(self.gamma *
                                         self.est_rewards[self.last_pull]/self.n_arms)
        self.w[~np.isfinite(self.w)] = 0
        self.probabilities = (1-self.gamma)*self.w / \
            sum(self.w)+self.gamma/self.n_arms
        self.probabilities[0] = 1 - sum(self.probabilities[1:])


class MiniBatchExp3Agent(Exp3Agent):
    def __init__(self, n_arms, gamma=1, max_reward=1, batch_size=1, random_state=1):
        super().__init__(n_arms, gamma, max_reward, random_state)
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        super().reset()
        self.reward_memory = []
        self.last_pull = 0
        return self

    def pull_arm(self):
        if self.t % self.batch_size == 0:
            new_a = super().pull_arm()
            return new_a
        else:
            self.a_hist.append(self.last_pull)
            return self.last_pull

    def update(self, X):
        self.reward_memory.append(X/self.max_reward)
        if self.t % self.batch_size == self.batch_size - 1:
            self.est_rewards[self.last_pull] = np.mean(self.reward_memory[-self.batch_size:])/self.probabilities[self.last_pull]
            self.w[self.last_pull] *= np.exp(self.gamma *
                                            self.est_rewards[self.last_pull]/self.n_arms)
            self.w[~np.isfinite(self.w)] = 0
            self.probabilities = (1-self.gamma)*self.w / \
                sum(self.w)+self.gamma/self.n_arms
            self.probabilities[0] = 1 - sum(self.probabilities[1:])
        self.t+=1


class UCB1Agent(Agent):
    def __init__(self, n_arms, sigma=1):
        super().__init__(n_arms)
        self.sigma = sigma
        self.reset()

    def reset(self):
        super().reset()
        self.avg_reward = np.array([np.inf for _ in range(self.n_arms)])
        self.rewards = [[] for _ in range(self.n_arms)]
        self.n_pulls = np.zeros(self.n_arms)
        return self

    def pull_arm(self):
        ucb1 = [self.avg_reward[a]+self.sigma*
                np.sqrt(2*np.log(self.t)/self.n_pulls[a]) for a in range(self.n_arms)]
        self.last_pull = np.argmax(ucb1)
        new_a = self.arms[self.last_pull]
        self.n_pulls[self.last_pull] += 1
        self.a_hist.append(new_a)
        return new_a

    def update(self, X):
        self.rewards[self.last_pull].append(X)
        if self.n_pulls[self.last_pull] == 1:
            self.avg_reward[self.last_pull] = X
        else:
            self.avg_reward[self.last_pull] = (
                self.avg_reward[self.last_pull]*self.n_pulls[self.last_pull]+X)/(self.n_pulls[self.last_pull]+1)
        self.t += 1


class AR2Agent(Agent):
    def __init__(self, n_arms, alpha, epoch_size, c0, sigma):
        super().__init__(n_arms)
        self.alpha = alpha
        self.epoch_size = epoch_size
        self.c1 = 24*c0
        self.sigma = sigma
        self.reset()

    def reset(self):
        super().reset()
        self.t0 = 1
        self.s = 1
        self.tau_trig = np.ones(self.n_arms)*np.inf
        self.tau = np.ones(self.n_arms)*np.inf
        self.est_rewards = np.ones(self.n_arms)*np.inf
        self.i_sup = None
        self.last_last_pull = None
        self.triggered_arms = []
        return self

    def pull_arm(self):
        if self.t0 <= self.n_arms + (self.s-1)*self.epoch_size:
            new_a = self.n_arms + (self.s-1)*self.epoch_size - self.t0
        else:
            if self.t0 == self.n_arms + (self.s-1)*self.epoch_size:
                self.tau_trig = np.ones(self.n_arms)*np.inf
                self.triggered_arms = []
            if self.est_rewards[self.last_pull] >= self.est_rewards[self.last_last_pull]:
                self.i_sup = self.last_pull
            else:
                self.i_sup = self.last_last_pull
            for i in range(self.n_arms):
                if (i != self.i_sup) and (i not in self.triggered_arms) and (self.est_rewards[self.i_sup]-self.est_rewards[i] <=
                                                                             self.c1*self.sigma*np.sqrt((self.alpha**2-self.alpha**(2*(self.t0-self.tau[i]+1)))/(1-self.alpha**2))):
                    self.triggered_arms.append(i)
                    self.tau_trig[i] = self.t0
            if len(self.triggered_arms) > 0 and self.t0 % 2 == 1:
                new_a = np.random.choice(
                    np.where(self.tau_trig == min(self.tau_trig))[0])
            else:
                new_a = self.i_sup
        self.last_last_pull = self.last_pull
        self.last_pull = new_a
        self.a_hist.append(new_a)
        return new_a

    def update(self, X):
        self.tau[self.last_pull] = self.t0
        if self.t0 < self.n_arms + (self.s-1)*self.epoch_size:
            self.est_rewards[self.last_pull] = self.alpha**(
                self.n_arms-self.t0+self.tau[self.last_pull]-1)*X
        else:
            self.est_rewards[self.last_pull] = self.alpha*X
            for i in range(self.n_arms):
                if i != self.last_pull:
                    self.est_rewards[i] *= self.alpha
            if self.t0 % self.epoch_size == 0:
                self.s += 1
        self.t0 += 1


class AutoregressiveClairvoyant(Agent):
    def __init__(self, n_arms, gamma, X0, k, random_state=1, constant=False):
        super().__init__(n_arms, random_state)
        self.X0 = X0
        self.constant = constant
        self.gamma = gamma
        self.k = k
        self.reset()

    def reset(self):
        super().reset()
        self.X = self.X0
        self.update_z()
        return self

    def pull_arm(self):
        if self.k > 0:
            if not self.constant:
                self.last_pull = np.argmax(
                    np.array([np.dot(self.gamma[a].T, self.z) for a in range(self.n_arms)]))
            else:
                self.last_pull = np.argmax(np.array(
                    [(self.gamma[a][0] / (1 - np.sum(self.gamma[a][1:]))) for a in range(self.n_arms)]))
        else:
            self.last_pull = np.argmax(
                np.array([self.gamma[a][0]*self.z for a in range(self.n_arms)]))
        new_a = self.arms[self.last_pull]
        self.a_hist.append(new_a)
        return new_a

    def update(self, x):
        self.X = np.append(self.X, x)
        self.update_z()
        self.t += 1

    def update_z(self):
        if self.k > 0:
            self.z = np.append(self.X[-self.k:], 1)[::-1].reshape(-1, 1)
        else:
            self.z = 1


class AutoregressiveRidgeAgent(Agent):
    def __init__(self, n_arms, X0, k, m, sigma_, delta_, lambda_=2e-6, random_state=1, constant=False):
        super().__init__(n_arms, random_state)
        self.X0 = X0
        self.k = k
        self.m = m
        self.constant = constant
        self.sigma_ = sigma_
        self.lambda_ = lambda_
        self.delta_ = delta_
        self.reset()

    def reset(self):
        super().reset()
        self.X = self.X0
        self.bound = np.zeros(self.n_arms) 
        self.update_z()
        self.V = [self.lambda_*np.eye(self.k+1) for _ in range(self.n_arms)]
        self.invV = [1/self.lambda_*np.eye(self.k+1)
                     for _ in range(self.n_arms)]
        self.b = [np.zeros(shape=(self.k+1, 1)) for _ in range(self.n_arms)]
        self.est_gamma = [np.nan for _ in range(self.n_arms)]
        return self

    def pull_arm(self):
        if self.t < self.n_arms:
            self.last_pull = self.t % self.n_arms
            new_a = self.arms[self.last_pull]
            self.a_hist.append(new_a)
            return new_a
        else:
            if not self.constant:
                ucb = np.array(
                    [np.dot(self.est_gamma[a].T, self.z)+self.bound[a] for a in range(self.n_arms)])
            else:
                ucb = np.array([(self.est_gamma[a][0] / (1 - np.sum(self.est_gamma[a][1:])
                )) + self.bound[a] for a in range(self.n_arms)])
            mask = np.where(ucb == max(ucb))[0] # (ucb == max(ucb)).any(), len(ucb), (abs(ucb) == abs(max(ucb)).any())
            self.last_pull = self.randgen.sample(list(mask), 1)[0] 
            new_a = self.arms[self.last_pull] 
            self.a_hist.append(new_a)
            return new_a

    def update(self, x):
        self.X = np.append(self.X, x)
        self.update_V()
        self.update_b(x)
        self.update_gamma()
        self.update_z()
        self.update_bound()
        self.t += 1

    def update_z(self):
        if self.k > 0:
            self.z = np.append(self.X[-self.k:], 1)[::-1].reshape(-1, 1)
        else:
            self.z = np.array([[1]])

    def update_V(self):
        self.V[self.last_pull] += np.matmul(self.z, self.z.T)
        self.invV[self.last_pull] = np.linalg.inv(self.V[self.last_pull])

    def update_b(self, x):
        self.b[self.last_pull] += self.z*x

    def update_gamma(self):
        self.est_gamma[self.last_pull] = np.matmul(
            self.invV[self.last_pull], self.b[self.last_pull])

    def update_bound(self):
        beta = np.sqrt(self.lambda_*(self.m**2+1))+self.sigma_*np.sqrt(2*np.log(self.n_arms /
                                                                                self.delta_)+np.log(np.linalg.det(self.V[self.last_pull])/self.lambda_**(self.k+1)))
        self.bound[self.last_pull] = beta *\
            np.sqrt(np.dot(np.matmul(self.z.T,
                                     self.invV[self.last_pull]), self.z))
