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
        inner_timesteps: int,
        use_tune: bool,
        report_freq: int,
        min_reward=-100,
        max_reward=100,
        max_action=2,
    ):
        super().__init__()
        self.choices = choices
        self.batches = batches
        self.report_freq = report_freq
        self.use_tune = use_tune
        self.data_size = inner_timesteps
        self.std_scale = std
        self.random, self._seed = seeding.np_random(0)
        self.context_length = context_length
        self.iterator = None
        reps = (self.context_length, self.batches, 1)
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.observation_space = gym.spaces.Box(
            low=np.tile(np.array([0, min_reward]), reps),
            high=np.tile(np.array([choices - 1, max_reward]), reps),
        )
        self.action_space = gym.spaces.Box(
            low=np.zeros(batches), high=np.ones(batches) * max_action
        )
        self.ucb = UCB(self._seed)
        self.dataset = np.zeros((self.data_size, self.batches, self.choices))

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
        means = np.random.normal(size=size, scale=1)
        stds = np.random.poisson(size=size)
        loc = np.tile(means, (self.data_size, 1, 1))
        scale = np.tile(stds, (self.data_size, 1, 1))
        self.dataset[:] = self.random.normal(loc, scale)
        our_loop = self.ucb.train_loop(dataset=self.dataset)
        base_loop = self.ucb.train_loop(dataset=self.dataset)
        optimal = means.max(axis=1, initial=-np.inf)

        next(our_loop)
        next(base_loop)
        coefficient = np.ones(self.batches)

        for _ in itertools.count():

            def interact(loop, c):
                for _ in range(self.context_length):
                    yield loop.send(c)

            actions, rewards = [
                np.stack(x)
                for x in zip(*interact(our_loop, np.expand_dims(coefficient, -1)))
            ]
            baseline_actions, _ = [np.stack(x) for x in zip(*interact(base_loop, 1))]
            chosen_means = means[
                np.tile(np.arange(self.batches), self.context_length),
                actions.astype(int).flatten(),
            ].reshape(self.context_length, self.batches)
            baseline_chosen_means = means[
                np.tile(np.arange(self.batches), self.context_length),
                baseline_actions.astype(int).flatten(),
            ].reshape(self.context_length, self.batches)
            s = np.stack([actions, rewards], axis=-1)
            r = np.mean(rewards)
            i = dict(
                regret=np.mean(optimal - chosen_means),
                baseline_regret=np.mean(optimal - baseline_chosen_means),
                coefficient=np.mean(actions),
            )
            self.report(**i)
            coefficient = yield s, r, False, i

    def render(self, mode="human"):
        pass
