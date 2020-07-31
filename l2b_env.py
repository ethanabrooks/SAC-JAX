import itertools
from typing import Generator

import gym
import jax
import numpy as np

from trainer import Trainer


class L2bEnv(Trainer, gym.Env):
    def __init__(self, context_length, steps_per_update, update_freq):
        super().__init__()
        self.update_freq = update_freq
        self.steps_per_update = steps_per_update
        self.context_length = context_length
        self.iterator = None
        self.rng = jax.random.PRNGKey(self.seed)
        self.observation_space = gym.spaces.Tuple(
            [self.env.observation_space, self.get_context_space()]
        )
        self.action_space = self.env.action_space

    def get_context_space(self):
        obs = self.env.observation_space
        act = self.env.action_space
        assert isinstance(obs, gym.spaces.Box)
        low = np.tile(
            np.concatenate([obs.low, act.low, obs.low] * self.context_length, axis=-1),
            (self.context_length, 1),
        )
        high = np.tile(
            np.concatenate([obs.high, act.high, obs.high], axis=-1),
            (self.context_length, 1),
        )
        return gym.spaces.Box(low=low, high=high)

    def step(self, action):
        return next(self.iterator)

    def reset(self):
        self.rng, rng = jax.random.split(self.rng)
        self.iterator = self._generator(rng)
        s, _, _, _ = next(self.iterator)
        return s

    def _generator(self, rng,) -> Generator:
        replay_buffer, loop, params = self.init(rng)
        s = next(loop.env)
        for i in itertools.count():
            t = i == self.max_timesteps
            r = self.eval_policy(params) if t else 0
            c = self.get_context(params)
            action = yield (s, c), r, t, {}
            s = loop.env.send(action)
            if i > self.start_timesteps and i % self.update_freq == 0:
                for _ in range(self.steps_per_update * self.agent.policy_freq):
                    rng, update_rng = jax.random.split(rng)
                    sample = replay_buffer.sample(self.batch_size, rng)
                    params = loop.train.send(sample)

    def get_context(self, params):
        env = self.make_env()
        env_loop = self.env_loop(params, env)
        s1 = next(env_loop)
        for _ in range(self.context_length):
            self.rng, noise_rng = jax.random.split(self.rng)
            a = self.act(params, s1, noise_rng)
            s2 = env_loop.send(a)
            yield np.concatenate([s1, a, s2], axis=-1)
            s1 = s2

    def render(self, mode="human"):
        pass
