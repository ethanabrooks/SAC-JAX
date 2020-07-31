"""
    Credits: https://github.com/sfujim/TD3
"""
from pprint import pprint

import numpy as np

import argparse
import itertools
from dataclasses import dataclass
from typing import Any, Generator

import gym
import jax
import jax.numpy as jnp
from ray.tune.suggest.hyperopt import HyperOptSearch

import configs
import ray
from ray import tune

from agent import Agent
from args import add_arguments
from replay_buffer import ReplayBuffer

OptState = Any


@dataclass
class Loops:
    env: Generator
    train: Generator


class Trainer:
    def __init__(
        self,
        batch_size,
        discount,
        env_id,
        eval_freq,
        expl_noise,
        lr,
        max_timesteps,
        noise_clip,
        policy,
        policy_freq,
        policy_noise,
        replay_size,
        seed,
        start_timesteps,
        use_tune,
        eval_episodes=100,
    ):
        self.use_tune = use_tune
        self.expl_noise = expl_noise
        self.max_timesteps = max_timesteps
        self.eval_freq = eval_freq
        self.start_timesteps = start_timesteps
        self.batch_size = batch_size
        self.replay_size = replay_size
        self.env_id = env_id
        self.eval_episodes = eval_episodes
        self.seed = seed
        self.policy = policy

        self.report(policy=policy, env=env_id, seed=seed)

        # if save_model and not os.path.exists("./models/" + policy):
        #     os.makedirs("./models/" + policy)

        self.env = self.make_env()
        self.env.seed(seed)
        self.max_action = float(self.env.action_space.high[0])
        self.action_dim = int(np.prod(self.env.action_space.shape))
        self.obs_dim = int(np.prod(self.env.action_space.shape))

        self.agent = Agent(
            policy=policy,
            max_action=self.max_action,
            action_dim=self.action_dim,
            lr=lr,
            discount=discount,
            noise_clip=noise_clip,
            policy_noise=policy_noise,
            policy_freq=policy_freq,
        )

    @classmethod
    def run(cls, config):
        pprint(config)
        cls(**config).train()

    @classmethod
    def main(cls, config, use_tune, num_samples, name, **kwargs):
        config = getattr(configs, config)
        config.update(use_tune=use_tune)
        for k, v in kwargs.items():
            if k not in config:
                config[k] = v
        if use_tune:
            local_mode = num_samples is None
            ray.init(webui_host="127.0.0.1", local_mode=local_mode)
            metric = "reward"
            kwargs = dict()
            if not local_mode:
                kwargs = dict(
                    search_alg=HyperOptSearch(config, metric=metric, mode="max"),
                    num_samples=num_samples,
                )

            def run(c):
                return cls.run(c)

            tune.run(
                run,
                name=name,
                config=config,
                resources_per_trial={"gpu": 1, "cpu": 2},
                **kwargs,
            )
        else:
            cls.run(config)

    def report(self, **kwargs):
        if self.use_tune:
            tune.report(**kwargs)
        else:
            pprint(kwargs)

    def env_loop(
        self, env=None, replay_buffer=None
    ) -> Generator[jnp.ndarray, jnp.ndarray, None]:
        env = env or self.env
        obs, done = env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0

        for t in range(int(self.max_timesteps)):

            episode_timesteps += 1

            action = yield obs

            # Perform action
            next_obs, reward, done, _ = env.step(action)
            # This 'trick' converts the finite-horizon task into an infinite-horizon one. It does change the problem
            # we are solving, however it has been observed empirically to work pretty well. noinspection
            # noinspection PyProtectedMember
            steps = env._max_episode_steps
            done_bool = float(done) if episode_timesteps < steps else 0

            # Store data in replay buffer
            if replay_buffer:
                replay_buffer.add(obs, action, next_obs, reward, done_bool)

            obs = next_obs
            episode_reward += reward

            if done:
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                self.report(
                    timestep=t + 1,
                    episode=episode_num + 1,
                    episode_timestep=episode_timesteps,
                    reward=episode_reward,
                )
                # Reset environment
                obs, done = env.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

    def act(self, params, obs, rng):
        return (
            self.agent.policy(params, obs)
            + jax.random.normal(rng, (self.action_dim,))
            * self.max_action
            * self.expl_noise
        ).clip(-self.max_action, self.max_action)

    def train(self):
        rng = jax.random.PRNGKey(self.seed)
        replay_buffer, loop = self.init(rng)
        obs = next(loop.env)
        params = next(loop.train)

        # Evaluate untrained policy.
        # We evaluate for 100 episodes as 10 episodes provide a very noisy estimation in some domains.
        evaluations = [self.eval_policy(params)]  # TODO
        best_performance = evaluations[-1]
        best_actor_params = params
        # if save_model: agent.save(f"./models/{policy}/{file_name}")

        for _ in range(self.start_timesteps):
            obs = loop.env.send(self.env.action_space.sample())
        for t in itertools.count(self.start_timesteps):
            # Select action randomly or according to policy
            rng, noise_rng = jax.random.split(rng)
            action = self.act(params, obs, noise_rng)
            try:
                obs = loop.env.send(action)

                # Train agent after collecting sufficient data
                rng, update_rng = jax.random.split(rng)
                sample = replay_buffer.sample(self.batch_size, rng)
                params = loop.train.send(sample)

            except StopIteration:
                print("Done training")
                return

            # Evaluate episode
            if (t + 1) % self.eval_freq == 0:
                evaluations.append(self.eval_policy(params))
                if evaluations[-1] > best_performance:
                    best_performance = evaluations[-1]
                    best_actor_params = params
                    # if save_model: agent.save(f"./models/{policy}/{file_name}")

        # At the end, re-evaluate the policy which is presumed to be best. This ensures an un-biased estimator when
        # reporting the average best results across each run.
        params = best_actor_params
        evaluations.append(self.eval_policy(params))
        print(f"Selected policy has an average score of: {evaluations[-1]:.3f}")

    def init(self, rng):
        replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            max_size=self.replay_size,
        )
        env_loop = self.env_loop(replay_buffer=replay_buffer)
        train_loop = self.agent.train_loop(
            rng, sample_obs=self.env.observation_space.sample()
        )
        return replay_buffer, Loops(train=train_loop, env=env_loop)

    def make_env(self):
        return gym.make(self.env_id)

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

        self.report(eval_reward=avg_reward)
        return avg_reward


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    add_arguments(PARSER)
    Trainer.main(**vars(PARSER.parse_args()))
