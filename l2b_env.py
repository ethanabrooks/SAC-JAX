import itertools
from typing import Generator

import gym
import jax
import numpy as np

from debug_env import DebugEnv
from replay_buffer import ReplayBuffer, BufferItem
from trainer import Trainer, Loops
from gym.utils.seeding import np_random


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
        # assert self.observation_space.contains(s)
        return s


class L2bEnv(Trainer, gym.Env):
    def __init__(self, update_freq, context_length, sample_done_prob, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_freq = update_freq
        self.context_length = context_length
        self.iterator = None
        self.observation_space = gym.spaces.Tuple(
            [self.env.observation_space, self.get_context_space()]
        )
        self.action_space = self.env.action_space
        self.rng = jax.random.PRNGKey(0)
        self.replay_buffer = DoubleReplayBuffer(
            obs_shape=self.env.observation_space.shape,
            action_shape=self.env.action_space.shape,
            max_size=self.replay_size,
            sample_done_prob=sample_done_prob,
        )

    def seed(self, seed=None):
        seed = seed or 0
        self.rng = jax.random.PRNGKey(seed)
        self.np_random, _ = np_random(seed)

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
        self.replay_buffer.reset()
        loop = Loops(
            env=self.env_loop(),
            train=self.agent.train_loop(
                rng, sample_obs=self.env.observation_space.sample()
            ),
        )
        params = next(loop.train)
        self.report(new_params=1)
        s = next(loop.env)
        c = np.stack(list(self.get_context(params)))
        r = 0
        for i in itertools.count():
            t = i == self.max_timesteps
            # r = self.eval_policy(params) if t else 0
            action = yield (s, c), r, t, {}
            step = loop.env.send(action)
            r = self.eval_policy(params) if t else step.reward  # TODO
            self.replay_buffer.add(step)
            s = step.obs
            if i > self.start_timesteps and i % self.update_freq == 0:
                for _ in range(self.update_freq):
                    rng, update_rng = jax.random.split(rng)
                    sample = self.replay_buffer.sample(self.batch_size, rng=rng)
                    params = loop.train.send(sample)

                c = np.stack(list(self.get_context(params)))

    def get_context(self, params):
        env_loop = self.env_loop(env=self.make_env(), max_timesteps=self.context_length)
        s1 = next(env_loop)
        for _ in range(self.context_length):
            self.rng, noise_rng = jax.random.split(self.rng)
            a = self.act(params, s1, noise_rng)
            s2 = env_loop.send(a).obs
            yield np.concatenate([s1, a, s2], axis=-1)
            s1 = s2

    def get_inner_env(self):
        return self.env

    def render(self, mode="human"):
        pass


class DoubleReplayBuffer(ReplayBuffer):
    def __init__(self, sample_done_prob, **kwargs):
        super().__init__(**kwargs)
        self.sample_done_prob = sample_done_prob
        self.done_buffer = ReplayBuffer(**kwargs)

    def add(self, item: BufferItem) -> None:
        if item.not_done:
            super().add(item)
        else:
            self.done_buffer.add(item)

    def sample(self, *args, rng, **kwargs) -> BufferItem:
        if jax.random.choice(
            rng, 2, p=[1 - self.sample_done_prob, self.sample_done_prob]
        ):
            return self.done_buffer.sample(*args, rng=rng, **kwargs)
        return super().sample(*args, rng=rng, **kwargs)

    def reset(self):
        self.size = 0
        self.ptr = 0
