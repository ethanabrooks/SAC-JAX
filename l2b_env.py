import itertools
from typing import Generator

import gym
import jax
import numpy as np

from debug_env import DebugEnv
from trainer import Trainer


class CatObsSpace(gym.ObservationWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.observation_space, gym.spaces.Tuple)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate(
                [space.low.flatten() for space in self.observation_space.spaces]
            ),
            high=np.concatenate(
                [space.high.flatten() for space in self.observation_space.spaces]
            ),
        )

    def observation(self, observation):
        s = np.concatenate([o.flatten() for o in observation])
        assert self.observation_space.contains(s)
        return s


class L2bEnv(Trainer, gym.Env):
    def __init__(self, update_freq, context_length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_freq = update_freq
        self.context_length = context_length
        self.iterator = None
        self.observation_space = gym.spaces.Tuple(
            [self.env.observation_space, self.get_context_space()]
        )
        self.action_space = self.env.action_space
        self.rng = jax.random.PRNGKey(0)

    def seed(self, seed=None):
        seed = seed or 0
        self.rng = jax.random.PRNGKey(seed)

    def get_context_space(self):
        obs = self.env.observation_space
        act = self.env.action_space
        assert isinstance(obs, gym.spaces.Box)
        low = np.tile(
            np.concatenate([obs.low, act.low, obs.low], axis=-1),
            (self.context_length, 1),
        )
        high = np.tile(
            np.concatenate([obs.high, act.high, obs.high], axis=-1),
            (self.context_length, 1),
        )
        return gym.spaces.Box(low=low, high=high)

    def make_env(self):
        return DebugEnv()

    def step(self, action):
        return self.iterator.send(action)

    def reset(self):
        self.rng, rng = jax.random.split(self.rng)
        self.iterator = self._generator(rng)
        s, _, _, _ = next(self.iterator)
        assert self.observation_space.contains(s)
        return s

    def _generator(self, rng,) -> Generator:
        replay_buffer, loop = self.init(rng)
        params = next(loop.train)
        s = next(loop.env)
        c = np.stack(list(self.get_context(params)))
        for i in itertools.count():
            t = i == self.max_timesteps
            r = self.eval_policy(params) if t else 0
            action = yield (s, c), r, t, {}
            step = loop.env.send(action)
            replay_buffer.add(step)
            s = step.obs
            if i > self.start_timesteps and i % self.update_freq == 0:
                for _ in range(self.update_freq):
                    rng, update_rng = jax.random.split(rng)
                    sample = replay_buffer.sample(self.batch_size, rng=rng)
                    params = loop.train.send(sample)

                c = np.stack(list(self.get_context(params)))

    def get_context(self, params):
        env = self.make_env()
        env_loop = self.env_loop(env)
        s1 = next(env_loop)
        for _ in range(self.context_length):
            self.rng, noise_rng = jax.random.split(self.rng)
            a = self.act(params, s1, noise_rng)
            s2 = env_loop.send(a).obs
            yield np.concatenate([s1, a, s2], axis=-1)
            s1 = s2

    def render(self, mode="human"):
        pass
