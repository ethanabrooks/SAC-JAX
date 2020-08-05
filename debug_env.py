import gym
import numpy as np
import jax
from gym.utils.seeding import np_random


def sigmoid(x):
    return (np.tanh(x) + 1) / 2


class DebugEnv(gym.Env):
    def __init__(self, levels: int, dim: int, std: float):
        self.std = std
        self.random, _ = np_random(0)
        self.embeddings = self.random.random((levels, dim))
        self.acceptable = self.random.random(levels)
        self.iterator = None
        self.observation_space = gym.spaces.Box(low=np.zeros(dim), high=np.ones(dim))
        self.action_space = gym.spaces.Box(low=np.zeros(1), high=np.ones(1))
        self._max_episode_steps = 2

    def seed(self, seed=None):
        self.random, _ = np_random(seed)

    def generator(self):
        t = False
        r = 1 / len(self.embeddings)
        for embedding, acceptable in zip(self.embeddings[:-1], self.acceptable):
            action = yield embedding, r, t, {}
            t = self.random.normal(scale=self.std) < abs(action - acceptable)
        yield self.embeddings[-1], r, True, {}

    def step(self, action):
        return self.iterator.send(action)

    def reset(self):
        self.iterator = self.generator()
        s, _, _, _ = next(self.iterator)
        return s

    def render(self, mode="human"):
        pass
