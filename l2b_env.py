import itertools
from typing import Generator

import gym
import jax
import jax.numpy as jnp
import numpy as np

from replay_buffer import ReplayBuffer, Sample, Step
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
        # assert self.observation_space.contains(s)
        return s


class L2bEnv(Trainer, gym.Env):
    def __init__(
        self, update_freq, context_length, alpha, levels, dim, std, *args, **kwargs
    ):
        self.std = std
        self.dim = dim
        self.levels = levels
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.update_freq = update_freq
        self.context_length = context_length
        self.iterator = None
        self.observation_space = gym.spaces.Tuple(
            [self.env.observation_space, self.get_context_space()]
        )
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

    def make_env(self):
        env = DebugEnv(levels=self.levels, std=self.std)
        return TimeLimit(env, max_episode_steps=len(list(env.reward_iterator())))

    def step(self, action):
        return self.iterator.send(action)

    def reset(self):
        self.rng, rng = jax.random.split(self.rng)
        self.iterator = self._generator(rng)
        s, _, _, _ = next(self.iterator)
        assert self.observation_space.contains(s)
        return s

    def _generator(self, rng,) -> Generator:
        self.replay_buffer.size = 0
        self.replay_buffer.ptr = 0
        env_loop = self.env_loop()
        train_loop = self.agent.train_loop(
            rng,
            sample_obs=self.env.observation_space.sample(),
            sample_action=self.env.action_space.sample(),
        )
        next(env_loop)
        params = next(train_loop)
        con = np.stack(list(self.get_context(params)))
        step = env_loop.send(self.env.action_space.sample())
        best_reward = None
        for t in range(self.max_timesteps) if self.max_timesteps else itertools.count():
            self.replay_buffer.add(step)
            obs = step.obs, con
            action = yield obs, self.alpha * step.reward, False, {}
            step = env_loop.send(action)
            if (t + 1) % self.update_freq == 0:
                for _ in range(self.update_freq):
                    rng, update_rng = jax.random.split(rng)
                    sample = self.replay_buffer.sample(self.batch_size, rng=rng)
                    params = train_loop.send(sample)
                con = np.stack(list(self.get_context(params)))

                if (t + 1) % self.update_freq == 0:
                    eval_reward = self.eval_policy(params)
                    self.report(
                        eval_reward=eval_reward,
                        actor_linear_b=params["actor/linear"].b.mean().item(),
                        actor_linear_w=params["actor/linear"].w.mean().item(),
                        actor_linear_1_b=params["actor/linear_1"].b.mean().item(),
                        actor_linear_1_w=params["actor/linear_1"].w.mean().item(),
                        actor_linear_2_b=params["actor/linear_2"].b.mean().item(),
                        actor_linear_2_w=params["actor/linear_2"].w.mean().item(),
                    )
                    if best_reward and eval_reward > best_reward:
                        best_reward = eval_reward
                        self.save(t, params)

        obs = step.obs, con
        yield obs, self.eval_policy(params), True, {}

    def get_context(self, params):
        env_loop = self.env_loop(env=self.make_env())
        s1 = next(env_loop)
        for _ in range(self.context_length):
            self.rng, noise_rng = jax.random.split(self.rng)
            a = self.agent.policy(params, s1, noise_rng)
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
