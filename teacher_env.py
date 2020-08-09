import collections
import itertools
from pprint import pprint
from typing import Generator

import gym
import numpy as np
from gym.utils import seeding
from ray import tune

from ucb import UCB


class TeacherEnv(gym.Env):
    def __init__(
        self,
        context_length: int,
        std: float,
        choices: int,
        inner_steps: int,
        use_tune: bool,
        report_freq: int,
        min_reward=-100,
        max_reward=100,
        max_action=2,
    ):
        super().__init__()
        self.report_freq = report_freq
        self.use_tune = use_tune
        self.data_size = (inner_steps + 2) * context_length
        self.std_scale = std
        self.random, self._seed = seeding.np_random(0)
        self.context_length = context_length
        self.choices = choices
        self.iterator = None
        self.observation_space = gym.spaces.Box(
            low=np.array([[0, min_reward]] * self.context_length),
            high=np.array([[choices - 1, max_reward]] * self.context_length),
        )
        self.action_space = gym.spaces.Box(
            low=np.zeros(1), high=np.ones(1) * max_action
        )
        self.ucb = UCB(self._seed)

        self.our_selections = np.zeros((self.data_size, self.choices))
        self.our_rewards = np.zeros((self.data_size, self.choices))
        self.their_selections = np.zeros((self.data_size, self.choices))
        self.their_rewards = np.zeros((self.data_size, self.choices))

        self.dataset = np.zeros((self.data_size, self.choices))

    def report(self, **kwargs):
        if self.use_tune:
            tune.report(**kwargs)
        else:
            pprint(kwargs)

    def seed(self, seed=None):
        seed = seed or 0
        self.random, self._seed = seeding.np_random(seed)
        self.ucb = UCB(self._seed)

    def reset(self):
        self.iterator = self._generator()
        s, _, _, _ = next(self.iterator)
        return s

    def step(self, action):
        return self.iterator.send(action)

    def zero_arrays(self):
        self.our_selections[:] = 0
        self.our_rewards[:] = 0
        self.their_selections[:] = 0
        self.their_rewards[:] = 0

    def _generator(self) -> Generator:
        means = self.random.random(self.choices)
        stds = self.random.random(self.choices) * self.std_scale
        optimal = means.max()
        means = np.tile(means, (self.data_size, 1))
        stds = np.tile(stds, (self.data_size, 1))

        self.zero_arrays()
        self.dataset[:] = self.random.normal(means, stds)
        our_loop = self.ucb.train_loop(
            self.dataset, rewards=self.our_rewards, selections=self.our_selections
        )
        base_loop = self.ucb.train_loop(
            self.dataset, rewards=self.their_rewards, selections=self.their_selections
        )
        next(our_loop)
        next(base_loop)
        coefficient = 1

        for _ in itertools.count():

            def interact(loop, c):
                for _ in range(self.context_length):
                    yield loop.send(c)

            actions, rewards = zip(*interact(our_loop, float(coefficient)))
            _, baseline_rewards = zip(*interact(base_loop, 1))
            s = np.array(list(zip(actions, rewards)))
            r = np.mean(rewards)
            i = dict(
                regret=optimal - r, baseline_regret=optimal - np.mean(baseline_rewards)
            )
            self.report(**i)
            coefficient = yield s, r, False, i

    def render(self, mode="human"):
        pass
