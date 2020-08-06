import gym
import numpy as np
import jax
from gym.utils.seeding import np_random


def sigmoid(x):
    return (np.tanh(x) + 1) / 2


class DebugEnv(gym.Env):
    def __init__(self, levels: int, std: float):
        self.std = std
        self.random, _ = np_random(0)
        self.levels = levels
        states = len(list(self.reward_iterator())) + 1
        self.embeddings = np.eye(states)
        self.iterator = None
        self.observation_space = gym.spaces.Box(
            low=np.zeros(states), high=np.ones(states)
        )
        self.action_space = gym.spaces.Box(low=np.zeros(1), high=np.ones(1))

    def reward_iterator(self):
        for i in range(self.levels):
            yield i
            for _ in range(i):
                yield -1

    def seed(self, seed=None):
        self.random, _ = np_random(seed)

    def generator(self):
        t = False
        for r, embedding in zip(self.reward_iterator(), self.embeddings):
            action = yield embedding, r, t, {}
            t = self.random.random() < float(action)
        yield self.embeddings[-1], len(self.embeddings), True, {}

    def step(self, action):
        return self.iterator.send(action)

    def reset(self):
        self.iterator = self.generator()
        s, _, _, _ = next(self.iterator)
        return s

    def render(self, mode="human"):
        pass


def play():
    env = DebugEnv(levels=5, std=100)
    _ = env.reset()
    cumulative = 0
    while True:
        # env.render()
        action = float(input("go"))
        _, r, t, i = env.step(action)
        cumulative += r
        print("reward:", r)
        print("done:", t)
        if t:
            print("cumulative", cumulative)
            cumulative = 0
            env.reset()


if __name__ == "__main__":
    play()
