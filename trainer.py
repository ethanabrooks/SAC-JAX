import itertools
from collections import Counter
from pathlib import Path
from pprint import pprint
from typing import Generator

import gym
import jax
import numpy as np
import ray
from flax.serialization import msgpack_serialize, msgpack_restore
from haiku.data_structures import to_mutable_dict, to_immutable_dict
from jax import numpy as jnp
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch

import configs
from agent import Agent
from replay_buffer import Step, ReplayBuffer

try:
    import pybullet_envs
except ImportError:
    pass


class Trainer:
    def __init__(
        self,
        batch_size,
        env_id,
        eval_freq,
        replay_size,
        seed,
        start_timesteps,
        use_tune,
        max_timesteps=None,
        eval_episodes=100,
        save_dir=None,
        **kwargs,
    ):
        self._save_dir = save_dir
        self.use_tune = use_tune
        self.max_timesteps = int(max_timesteps) if max_timesteps else None
        self.eval_freq = int(eval_freq)
        self.start_timesteps = int(start_timesteps)
        self.batch_size = int(batch_size)
        self.replay_size = int(replay_size)
        self.env_id = env_id
        self.eval_episodes = int(eval_episodes)
        seed = int(seed)
        self.rng = jax.random.PRNGKey(seed)

        self.report(env=env_id, seed=seed)

        self.env = self.make_env()
        self.env.seed(seed)
        self.env.action_space.np_random.seed(seed)
        self.env.observation_space.np_random.seed(seed)
        self.obs_dim = int(np.prod(self.env.action_space.shape))
        self.agent = self.build_agent(**kwargs)

    def make_env(self):
        # env = DebugEnv(levels=self.levels)
        # return TimeLimit(env, max_episode_steps=len(list(env.reward_iterator())))
        return gym.make(self.env_id)

    def build_agent(self, **kwargs):
        return Agent(
            max_action=self.env.action_space.high,
            min_action=self.env.action_space.low,
            action_dim=int(np.prod(self.env.action_space.shape)),
            **kwargs,
        )

    @classmethod
    def run(cls, **kwargs):
        return cls(**kwargs).train()

    @classmethod
    def main(
        cls,
        config,
        use_tune,
        num_samples,
        name,
        gpus_per_trial,
        cpus_per_trial,
        **kwargs,
    ):
        use_tune = use_tune or num_samples
        config = configs.get_config(config)
        config.update(use_tune=use_tune)
        for k, v in kwargs.items():
            if v is not None:
                config[k] = v

        def run(c):
            return cls.run(**c)

        if use_tune:
            local_mode = num_samples is None
            ray.init(dashboard_host="127.0.0.1", local_mode=local_mode)
            metric = "final_reward"

            resources_per_trial = {"gpu": gpus_per_trial, "cpu": cpus_per_trial}
            kwargs = dict()
            if not local_mode:
                kwargs = dict(
                    search_alg=HyperOptSearch(config, metric=metric),
                    num_samples=num_samples,
                )
            tune.run(
                run,
                name=name,
                config=config,
                resources_per_trial=resources_per_trial,
                **kwargs,
            )
        else:
            run(config)

    def report(self, **kwargs):
        if self.use_tune:
            tune.report(**kwargs)
        else:
            pprint(kwargs)

    def env_loop(self, env=None) -> Generator[Step, jnp.ndarray, None]:
        env = env or self.env
        obs, done = env.reset(), False

        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0

        action = yield obs
        counter = Counter()

        for t in itertools.count():
            episode_timesteps += 1

            # Perform action
            next_obs, reward, done, info = env.step(action)
            counter.update(**info)
            episode_reward += reward

            # This 'trick' converts the finite-horizon task into an infinite-horizon one. It does change the problem
            # we are solving, however it has been observed empirically to work pretty well. noinspection
            try:
                # noinspection PyProtectedMember
                max_episode_steps = env._max_episode_steps
            except AttributeError:
                max_episode_steps = np.inf
            done_bool = float(done) if episode_timesteps < max_episode_steps else 0

            action = yield Step(
                obs=obs,
                action=action,
                next_obs=next_obs,
                reward=reward,
                done=done_bool,
            )
            obs = next_obs

            if done:
                obs, done = env.reset(), False

                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                self.report(
                    timestep=t + 1,
                    episode=episode_num + 1,
                    episode_timestep=episode_timesteps,
                    reward=episode_reward,
                    **{"episode_" + k: v for k, v in counter.items()},
                )

                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1
                counter = Counter()

    def train(self):
        rng = self.rng
        replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            max_size=self.replay_size,
        )
        env_loop = self.env_loop()
        train_loop = self.agent.train_loop(
            rng,
            sample_obs=self.env.observation_space.sample(),
            sample_action=self.env.action_space.sample(),
        )

        next(env_loop)
        params = next(train_loop)

        # Evaluate untrained policy.
        # We evaluate for 100 episodes as 10 episodes provide a very noisy estimation in some domains.
        eval_reward = self.eval_policy(params)
        evaluations = [eval_reward]
        best_performance = eval_reward
        best_params = params

        step = env_loop.send(self.env.action_space.sample())
        for t in range(self.max_timesteps) if self.max_timesteps else itertools.count():
            replay_buffer.add(step)
            if t <= self.start_timesteps:
                action = self.env.action_space.sample()
            else:
                # Select action randomly or according to policy
                rng, noise_rng = jax.random.split(rng)
                action = self.agent.policy(params, step.obs, noise_rng)

                # Train agent after collecting sufficient data
                rng, update_rng = jax.random.split(rng)
                sample = replay_buffer.sample(self.batch_size, rng=rng)
                params = train_loop.send(sample)
            step = env_loop.send(action)

            # Evaluate episode
            if (t + 1) % self.eval_freq == 0:
                eval_reward = self.eval_policy(params)
                self.report(eval_reward=eval_reward)
                evaluations.append(eval_reward)
                if best_performance is None or evaluations[-1] > best_performance:
                    best_performance = evaluations[-1]
                    best_params = params
                    save_dir = self.save_dir(t)
                    if save_dir:
                        self.save(save_dir, params)

        # At the end, re-evaluate the policy which is presumed to be best. This ensures an un-biased estimator when
        # reporting the average best results across each run.
        params = best_params
        evaluations.append(self.eval_policy(params))
        self.report(final_reward=self.eval_policy(params))

    def eval_policy(self, params) -> float:
        eval_env = self.make_env()

        avg_reward = 0.0
        for _ in range(self.eval_episodes):
            obs, done = eval_env.reset(), False

            # noinspection PyProtectedMember
            remaining_steps = eval_env._max_episode_steps

            while not done:
                action = self.agent.policy(params, obs)
                obs, reward, done, _ = eval_env.step(action)

                remaining_steps -= 1

                avg_reward += reward

        avg_reward /= self.eval_episodes

        return avg_reward

    def save_dir(self, t=None):
        if self.use_tune:
            with tune.checkpoint_dir(step=t) as save_dir:
                return Path(save_dir)
        return self._save_dir

    def save(self, path: Path, params):
        with Path(path, "params").open("wb") as fp:
            fp.write(msgpack_serialize(to_mutable_dict(params)))

    @staticmethod
    def load(path):
        with Path(path, "params").open("rb") as fp:
            return to_immutable_dict(msgpack_restore(fp.read()))
