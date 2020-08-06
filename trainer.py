"""
    Credits: https://github.com/sfujim/TD3
"""
import argparse
import itertools
from dataclasses import dataclass
from pprint import pprint
from typing import Any, Generator, Tuple

import gym
import jax
import jax.numpy as jnp
import numpy as np
import ray
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch

import configs
from agent import Agent
from args import add_arguments
from replay_buffer import ReplayBuffer, Sample, Step
from debug_env import DebugEnv

OptState = Any


@dataclass
class Loops:
    env: Generator
    train: Generator


@dataclass
class ReportData:
    reward: float
    done: bool
    t: int


class Trainer:
    def __init__(
        self,
        batch_size,
        env_id,
        eval_freq,
        expl_noise,
        max_timesteps,
        policy,
        replay_size,
        seed,
        start_timesteps,
        use_tune,
        std,
        levels,
        eval_episodes=100,
        **kwargs,
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
        self.std = std
        self.levels = levels
        seed = int(seed)
        self.rng = jax.random.PRNGKey(seed)

        self.report(policy=policy, env=env_id, seed=seed)

        # if save_model and not os.path.exists("./models/" + policy):
        #     os.makedirs("./models/" + policy)

        self.env = self.make_env()
        self.env.seed(seed)
        self.env.action_space.np_random.seed(seed)
        self.env.observation_space.np_random.seed(seed)
        self.max_action = self.env.action_space.high
        self.min_action = self.env.action_space.low
        self.action_dim = int(np.prod(self.env.action_space.shape))
        self.obs_dim = int(np.prod(self.env.action_space.shape))
        self.agent = self.build_agent(**kwargs, policy=policy)

    def build_agent(self, **kwargs):
        return Agent(
            max_action=self.max_action,
            min_action=self.min_action,
            action_dim=self.action_dim,
            **kwargs,
        )

    @classmethod
    def run(cls, config):
        pprint(config)
        cls(**config).train()

    @classmethod
    def main(
        cls,
        config,
        best,
        use_tune,
        num_samples,
        name,
        gpus_per_trial,
        cpus_per_trial,
        **kwargs,
    ):
        config = configs.get_config(config)
        config.update(use_tune=use_tune)
        for k, v in kwargs.items():
            if k not in config:
                config[k] = v
        if use_tune:
            local_mode = num_samples is None
            ray.init(webui_host="127.0.0.1", local_mode=local_mode)
            metric = "final_reward"

            def run(c):
                return cls.run(c)

            resources_per_trial = {"gpu": gpus_per_trial, "cpu": cpus_per_trial}
            kwargs = dict()
            if not local_mode:
                points_to_evaluate = None if best is None else [getattr(configs, best)]
                kwargs = dict(
                    search_alg=HyperOptSearch(
                        config, metric=metric, points_to_evaluate=points_to_evaluate
                    ),
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
            cls.run(config)

    def report(self, **kwargs):
        if self.use_tune:
            tune.report(**kwargs)
        else:
            pprint(kwargs)

    def report_loop(self) -> Generator[None, ReportData, None]:
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0

        while True:
            data = yield
            episode_reward += data.reward

            if data.done:
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                self.report(
                    timestep=data.t + 1,
                    episode=episode_num + 1,
                    episode_timestep=episode_timesteps,
                    reward=episode_reward,
                )
                # Reset environment
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

    def env_loop(
        self, env=None, report_loop=None
    ) -> Generator[Step, jnp.ndarray, None]:
        if report_loop:
            next(report_loop)
        env = env or self.env
        obs, done = env.reset(), False
        episode_timesteps = 0
        action = yield obs

        for t in itertools.count():

            episode_timesteps += 1
            # Perform action
            next_obs, reward, done, _ = env.step(action)
            if report_loop:
                report_loop.send(ReportData(reward=reward, done=done, t=t))
            # This 'trick' converts the finite-horizon task into an infinite-horizon one. It does change the problem
            # we are solving, however it has been observed empirically to work pretty well. noinspection
            # noinspection PyProtectedMember
            done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

            action = yield Step(
                obs=obs,
                action=action,
                next_obs=next_obs,
                reward=reward,
                done=done_bool,
            )
            obs = next_obs

            if done:
                # Reset environment
                obs, done = env.reset(), False

    def act(self, params, obs, rng):
        return self.agent.policy(params, obs, rng)

    def train(self):
        rng = self.rng
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

    def make_env(self):
        # return DebugEnv(levels=self.levels, std=self.std)
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

        return avg_reward


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    add_arguments(PARSER)
    PARSER.add_argument("--std", type=float)
    PARSER.add_argument("--levels", type=int)
    Trainer.main(**vars(PARSER.parse_args()))
