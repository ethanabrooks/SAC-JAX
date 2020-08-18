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
        batches: int,
        data_size: int,
        use_tune: bool,
        report_freq: int,
        min_reward=-100,
        max_reward=100,
        max_action=4,
    ):
        super().__init__()
        self.choices = choices
        self.batches = batches
        self.report_freq = report_freq
        self.use_tune = use_tune
        self.std_scale = std
        self.random, self._seed = seeding.np_random(0)
        self.context_length = context_length
        self.iterator = None
        reps = (self.context_length, self.batches, 1)
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.data_size = data_size
        self.observation_space = gym.spaces.Box(
            low=np.tile(np.array([0, min_reward]), reps),
            high=np.tile(np.array([choices - 1, max_reward]), reps),
        )
        self.action_space = gym.spaces.Box(
            low=np.zeros(batches), high=np.ones(batches) * max_action
        )
        self.ucb = UCB(self._seed)
        self.dataset = np.zeros((data_size, self.batches, self.choices))

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

    def _generator(self) -> Generator:
        size = (self.batches, self.choices)

        def sample_dataset(h):
            means = np.random.normal(size=size, scale=1)
            stds = np.random.poisson(size=size)
            return np.tile(means, (h, 1, 1)), np.tile(stds, (h, 1, 1))

        half = int(len(self.dataset) // 2)
        loc1, scale1 = sample_dataset(half)
        loc2, scale2 = sample_dataset(len(self.dataset) - half)
        loc = np.vstack([loc1, loc2])
        scale = np.vstack([scale1, scale2])
        self.dataset = self.random.normal(loc, scale)
        our_loop = self.ucb.train_loop(dataset=self.dataset)
        base_loop = self.ucb.train_loop(dataset=self.dataset)
        optimal = loc.max(axis=-1, initial=-np.inf)

        baseline_return = np.zeros((self.context_length, self.batches))

        next(our_loop)
        next(base_loop)
        coefficient = 2 * np.ones(self.batches)
        ones = np.ones(self.batches * self.context_length, dtype=int)

        for t in itertools.count():

            def interact(loop, c):
                for _ in range(self.context_length):
                    yield loop.send(c)

            actions, rewards = [
                np.stack(x)
                for x in zip(*interact(our_loop, c=np.expand_dims(coefficient, -1)))
            ]
            baseline_actions, baseline_rewards = [
                np.stack(x) for x in zip(*interact(base_loop, c=1))
            ]
            chosen_means = loc[
                ones * t,
                np.tile(np.arange(self.batches), self.context_length),
                actions.astype(int).flatten(),
            ].reshape(self.context_length, self.batches)
            baseline_chosen_means = loc[
                ones * t,
                np.tile(np.arange(self.batches), self.context_length),
                baseline_actions.astype(int).flatten(),
            ].reshape(self.context_length, self.batches)
            baseline_return += baseline_rewards

            s = np.stack([actions, rewards], axis=-1)
            r = np.mean(rewards)
            if t % self.report_freq == 0:
                self.report(
                    baseline_regret=np.mean(optimal[t: t+1] - baseline_chosen_means),
                    baseline_rewards=np.mean(baseline_rewards),
                    regret=np.mean(optimal[t: t+1] - chosen_means),
                    rewards=np.mean(rewards),
                    coefficient=np.mean(coefficient),
                )
            if t == self.data_size:
                self.report(baseline_return=baseline_return)
            coefficient = yield s, r, False, {}


    def render(self, mode="human"):
        pass
