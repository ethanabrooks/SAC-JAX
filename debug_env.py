import gym
import numpy as np
import jax
from gym.utils.seeding import np_random
from ray import tune


def sigmoid(x):
    return (np.tanh(x) + 1) / 2


class DebugEnv(gym.Env):
    def __init__(self, levels: int):
        self.random, _ = np_random(0)
        self.levels = levels
        states = len(list(self.reward_iterator())) + 1
        self.max_reward = sum(self.reward_iterator()) - 1
        self.embeddings = np.eye(states)
        self.iterator = None
        self.observation_space = gym.spaces.Box(
            low=np.zeros(states), high=np.ones(states)
        )
        self.action_space = gym.spaces.Box(low=np.zeros(1), high=np.ones(1))

    def reward_iterator(self):
        for i in range(self.levels):
            yield 1
            for _ in range(i):
                yield 1
        yield 1

    def seed(self, seed=None):
        self.random, _ = np_random(seed)

    def generator(self):
        *rewards, last_reward = [r / self.max_reward for r in self.reward_iterator()]
        t = False
        b = True
        for r, embedding in zip(rewards, self.embeddings):
            action = yield embedding, r, t, {}
            random = self.random.random()
            t = random < float(action) if b else random > float(action)
            b = not b
        yield self.embeddings[-1], last_reward, True, {}

    def step(self, action):
        return self.iterator.send(action)

    def reset(self):
        self.iterator = self.generator()
        s, _, _, _ = next(self.iterator)
        return s

    def render(self, mode="human"):
        pass


def play():
    env = DebugEnv(levels=2)
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
