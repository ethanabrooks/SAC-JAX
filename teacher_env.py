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
        choices: int,
        batches: int,
        data_size: int,
        use_tune: bool,
        report_freq: int,
        lam=1,
        min_reward=-100,
        max_reward=100,
        max_action=4,
    ):
        super().__init__()
        self.choices = choices
        self.batches = batches
        self.report_freq = report_freq
        self.use_tune = use_tune
        self.lam = lam
        self.random, self._seed = seeding.np_random(0)
        self.context_length = context_length
        self._max_episode_steps = np.inf
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
            # low=np.ones(batches), high=np.ones(batches) * max_action
            low=np.ones(batches),
            high=np.ones(batches) * 1.9,
        )
        self.ucb = UCB(self._seed)
        self.dataset = np.zeros((data_size, self.batches, self.choices))
        size = (self.batches, self.choices)

        def sample_dataset(h):
            means = np.random.normal(size=size, scale=1)
            stds = np.random.poisson(size=size, lam=self.lam)
            return np.tile(means, (h, 1, 1)), np.tile(stds, (h, 1, 1))

        loc, scale = sample_dataset(len(self.dataset))
        self.dataset = np.random.normal(loc, scale)
        self.loc = loc

    def report(self, **kwargs):
        kwargs = {k: np.mean(v) for k, v in kwargs.items()}
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
            stds = np.random.poisson(size=size, lam=self.lam)
            return np.tile(means, (h, 1, 1)), np.tile(stds, (h, 1, 1))

        # half = int(len(self.dataset) // 2)
        # loc1, scale1 = sample_dataset(half)
        # loc2, scale2 = sample_dataset(len(self.dataset) - half)
        # loc = np.vstack([loc1, loc2])
        # scale = np.vstack([scale1, scale2])
        # loc, scale = sample_dataset(len(self.dataset))
        # self.dataset = self.random.normal(loc, scale)
        our_loop = self.ucb.train_loop(dataset=self.dataset)
        base_loop = self.ucb.train_loop(dataset=self.dataset)
        optimal = self.loc.max(axis=-1, initial=-np.inf)

        baseline_return = np.zeros((self.context_length, self.batches))

        next(our_loop)
        next(base_loop)
        coefficient = 0 * np.ones(self.batches)  # TODO
        arange = np.arange(self.batches)

        def interact(loop, c):
            for _ in range(self.context_length):
                yield loop.send(c)

        done = False
        interaction = interact(our_loop, c=np.expand_dims(coefficient, -1))

        for t in itertools.count():
            actions, rewards = [np.stack(x) for x in zip(*interaction)]
            baseline_actions, baseline_rewards = [
                np.stack(x) for x in zip(*interact(base_loop, c=1))
            ]
            chosen_means = self.loc[t][
                np.tile(arange, self.context_length),
                actions.astype(np.int32).flatten(),
            ].reshape(self.context_length, self.batches)
            baseline_chosen_means = self.loc[t][
                np.tile(arange, self.context_length),
                baseline_actions.astype(int).flatten(),
            ].reshape(self.context_length, self.batches)
            baseline_return += np.mean(baseline_rewards)

            s = np.stack([actions, rewards], axis=-1)
            r = np.mean(rewards)
            if t % self.report_freq == 0:
                self.report(
                    baseline_regret=np.mean(optimal[t : t + 1] - baseline_chosen_means),
                    baseline_rewards=np.mean(baseline_rewards),
                    regret=np.mean(optimal[t : t + 1] - chosen_means),
                    rewards=r,
                    coefficient=np.mean(coefficient).item(),
                )
            try:
                interaction = list(
                    interact(our_loop, c=np.expand_dims(coefficient, -1))
                )
                self._max_episode_steps = t
            except RuntimeError:  # StopIteration
                done = True

            if done:
                self.report(baseline_return=baseline_return)

            coefficient = yield np.ones_like(s), r, done, {}

    def render(self, mode="human"):
        pass
