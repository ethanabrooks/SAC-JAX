import gym
import numpy as np
import jax
from gym.utils.seeding import np_random


def sigmoid(x):
    return (np.tanh(x) + 1) / 2


class Env(gym.Env):
    def __init__(self, levels: int, dim: int, tol: float, seed: int):
        self.tol = tol
        self.embeddings = np.random.random((levels, dim))
        self.acceptable = np.random.random(levels)
        self.iterator = None
        self.random, _ = np_random(seed)

    def generator(self):
        t = False
        r = 1 / len(self.embeddings)
        for embedding, acceptable in zip(self.embeddings[:-1], self.acceptable):
            action = yield embedding, r, t, {}
            t = self.random.random() < sigmoid(abs(action - acceptable) * self.tol)
        yield self.embeddings[-1], r, True, {}

    def step(self, action):
        return self.iterator.send(action)

    def reset(self):
        self.iterator = self.generator()
        s, _, _, _ = next(self.iterator)
        return s

    def render(self, mode="human"):
        pass