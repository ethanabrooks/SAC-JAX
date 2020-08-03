import itertools
from typing import Generator

import gym
import jax
import jax.numpy as jnp
import numpy as np

from debug_env import DebugEnv
from replay_buffer import ReplayBuffer, Sample, Step
from trainer import Trainer, Loops, ReportData


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
    def __init__(self, update_freq, context_length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_freq = update_freq
        self.context_length = context_length
        self.iterator = None
        # self.observation_space = gym.spaces.Tuple(
        #     [self.env.observation_space, self.get_context_space()]
        # )
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.rng = jax.random.PRNGKey(0)
        self.replay_buffer = ReplayBuffer(
            obs_shape=self.env.observation_space.shape,
            action_shape=self.env.action_space.shape,
            max_size=self.replay_size,
        )

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

    # def make_env(self):
    # return DebugEnv()

    def step(self, action):
        return self.iterator.send(action)

    def reset(self):
        self.rng, rng = jax.random.split(self.rng)
        self.iterator = self._generator(rng)
        s, _, _, _ = next(self.iterator)
        # assert self.observation_space.contains(s)
        return s

    def _generator(self, rng,) -> Generator:
        replay_buffer = self.build_replay_buffer()
        loop = Loops(
            env=self.env_loop(report_loop=self.report_loop()),
            train=self.agent.train_loop(
                rng, sample_obs=self.env.observation_space.sample()
            ),
        )
        next(loop.env)
        params = next(loop.train)

        # Evaluate untrained policy.
        # We evaluate for 100 episodes as 10 episodes provide a very noisy estimation in some domains.
        # evaluations = [self.eval_policy(params)]  # TODO
        # best_performance = evaluations[-1]
        # best_actor_params = params
        # if save_model: agent.save(f"./models/{policy}/{file_name}")

        step = loop.env.send(self.env.action_space.sample())
        for t in range(self.max_timesteps) if self.max_timesteps else itertools.count():
            replay_buffer.add(step)
            action = yield step.obs, step.reward, step.done, {}
            if t <= self.start_timesteps:
                action = self.env.action_space.sample()
            else:
                # Select action randomly or according to policy
                rng, noise_rng = jax.random.split(rng)
                action = self.act(params, step.obs, noise_rng)

                # Train agent after collecting sufficient data
                rng, update_rng = jax.random.split(rng)
                sample = replay_buffer.sample(self.batch_size, rng=rng)
                params = loop.train.send(sample)
            step = loop.env.send(action)

        self.report(final_reward=self.eval_policy(params))

    def get_context(self, params):
        env_loop = self.env_loop(env=self.make_env())
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

    def add(self, step: Step) -> None:
        if step.done:
            self.done_buffer.add(step)
        else:
            super().add(step)

    def sample(self, batch_size: int, rng: jnp.ndarray) -> Sample:
        if jax.random.choice(
            rng, 2, p=[1 - self.sample_done_prob, self.sample_done_prob]
        ):
            return self.done_buffer.sample(batch_size, rng)
        return super().sample(batch_size, rng)
