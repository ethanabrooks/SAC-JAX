"""
    Credits: https://github.com/sfujim/TD3
"""
import argparse
import itertools
from dataclasses import dataclass
from pprint import pprint
from typing import Any, Generator

import gym
import jax
import jax.numpy as jnp
import numpy as np
import ray
from gym.envs.classic_control import PendulumEnv
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch

import configs
from agent import Agent
from args import add_arguments
from replay_buffer import ReplayBuffer, BufferItem

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
        self.max_timesteps = int(max_timesteps)
        self.eval_freq = int(eval_freq)
        self.start_timesteps = int(start_timesteps)
        self.batch_size = int(batch_size)
        self.replay_size = int(replay_size)
        self.env_id = env_id
        self.eval_episodes = int(eval_episodes)
        self.policy = policy
        seed = int(seed)
        self.rng = jax.random.PRNGKey(seed)

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
    def main(cls, config, best, use_tune, num_samples, name, **kwargs):
        config = getattr(configs, config)
        config.update(use_tune=use_tune)
        for k, v in kwargs.items():
            if k not in config:
                config[k] = v
        if use_tune:
            local_mode = num_samples is None
            ray.init(webui_host="127.0.0.1", local_mode=local_mode)
            metric = "final_reward"
            kwargs = dict()
            if not local_mode:
                points_to_evaluate = None if best is None else [getattr(configs, best)]
                kwargs = dict(
                    search_alg=HyperOptSearch(
                        config,
                        metric=metric,
                        mode="max",
                        points_to_evaluate=points_to_evaluate,
                    ),
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

    def env_loop(self, env=None) -> Generator[BufferItem, jnp.ndarray, None]:
        env = env or self.env
        obs, done = env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0
        action = yield obs

        for t in range(int(self.max_timesteps)):

            episode_timesteps += 1
            # Perform action
            next_obs, reward, done, _ = env.step(action)
            # This 'trick' converts the finite-horizon task into an infinite-horizon one. It does change the problem
            # we are solving, however it has been observed empirically to work pretty well. noinspection
            # noinspection PyProtectedMember
            steps = env._max_episode_steps
            done_bool = float(done) if episode_timesteps < steps else 0

            action = yield BufferItem(
                obs=obs,
                action=action,
                next_obs=next_obs,
                reward=reward,
                not_done=1 - done_bool,
            )

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
        rng = self.rng
        replay_buffer, loop = self.init(rng)
        next(loop.env)
        params = next(loop.train)

        # Evaluate untrained policy.
        # We evaluate for 100 episodes as 10 episodes provide a very noisy estimation in some domains.
        # evaluations = [self.eval_policy(params)]  # TODO
        # best_performance = evaluations[-1]
        # best_actor_params = params
        # if save_model: agent.save(f"./models/{policy}/{file_name}")

        step = loop.env.send(self.env.action_space.sample())
        for t in itertools.count():
            replay_buffer.add(step)
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
            try:
                step = loop.env.send(action)

            except StopIteration:
                self.report(final_reward=self.eval_policy(params))
                return

            # Evaluate episode
            # if (t + 1) % self.eval_freq == 0:
            # evaluations.append(self.eval_policy(params))
            # if evaluations[-1] > best_performance:
            # best_performance = evaluations[-1]
            # best_actor_params = params
            # if save_model: agent.save(f"./models/{policy}/{file_name}")

        # At the end, re-evaluate the policy which is presumed to be best. This ensures an un-biased estimator when
        # reporting the average best results across each run.
        # params = best_actor_params
        # evaluations.append(self.eval_policy(params))
        # print(f"Selected policy has an average score of: {evaluations[-1]:.3f}")

    def build_replay_buffer(self):
        return ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            max_size=self.replay_size,
        )

    def init(self, rng):
        replay_buffer = self.build_replay_buffer()
        env_loop = self.env_loop()
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
    PARSER.add_argument("config")
    PARSER.add_argument("--best")
    PARSER.add_argument("--name")
    PARSER.add_argument("--num-samples", type=int)
    PARSER.add_argument("--no-tune", dest="use_tune", action="store_false")
    Trainer.main(**vars(PARSER.parse_args()))
