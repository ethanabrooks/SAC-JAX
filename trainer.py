"""
    Credits: https://github.com/sfujim/TD3
"""

import argparse
from typing import Any
import numpy as np

import gym
import jax

from args import add_arguments
from replay_buffer import ReplayBuffer
from agent import Agent

import os

OptState = Any


class Trainer:
    def __init__(
        self,
        env_id,
        seed,
        lr,
        discount,
        noise_clip,
        policy_noise,
        policy_freq,
        replay_size,
        max_timesteps,
        expl_noise,
        policy,
        eval_freq,
        start_timesteps,
        batch_size,
        eval_episodes=100,
    ):
        self.env_id = env_id
        self.eval_episodes = eval_episodes
        idx = 0
        file_name = f"{env_id}_{idx}"
        # For easy extraction of the data, we save all runs using a serially increasing indicator.
        while os.path.exists("./results/" + policy + "/" + file_name + ".npy"):
            idx += 1
            file_name = f"{env_id}_{idx}"

        print("---------------------------------------")
        print(f"Policy: {policy}, Env: {env_id}, Seed: {seed}")
        print("---------------------------------------")

        if not os.path.exists("./results/" + policy):
            os.makedirs("./results/" + policy)

        # if save_model and not os.path.exists("./models/" + policy):
        #     os.makedirs("./models/" + policy)

        env = gym.make(env_id)
        env.seed(seed)

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        self.agent = agent = Agent(
            policy=policy,
            action_dim=action_dim,
            max_action=max_action,
            lr=lr,
            discount=discount,
            noise_clip=noise_clip,
            policy_noise=policy_noise,
            policy_freq=policy_freq,
        )

        obs, done = env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0

        rng = jax.random.PRNGKey(seed)

        iterator = agent.generator(rng, sample_obs=obs)

        params = next(iterator)

        replay_buffer = ReplayBuffer(obs_dim, action_dim, max_size=replay_size)

        # Evaluate untrained policy.
        # We evaluate for 100 episodes as 10 episodes provide a very noisy estimation in some domains.
        evaluations = [self.eval_policy(params)]
        np.save(f"./results/{policy}/{file_name}", evaluations)
        best_performance = evaluations[-1]
        best_actor_params = params
        # if save_model: agent.save(f"./models/{policy}/{file_name}")

        for t in range(int(max_timesteps)):

            episode_timesteps += 1

            # Select action randomly or according to policy
            if t < start_timesteps:
                action = env.action_space.sample()
            else:
                rng, noise_rng = jax.random.split(rng)
                action = (
                    agent.policy(params, obs)
                    + jax.random.normal(noise_rng, (action_dim,))
                    * max_action
                    * expl_noise
                ).clip(-max_action, max_action)

            # Perform action
            next_obs, reward, done, _ = env.step(action)
            # This 'trick' converts the finite-horizon task into an infinite-horizon one. It does change the problem
            # we are solving, however it has been observed empirically to work pretty well. noinspection
            # noinspection PyProtectedMember
            steps = env._max_episode_steps
            done_bool = float(done) if episode_timesteps < steps else 0

            # Store data in replay buffer
            replay_buffer.add(obs, action, next_obs, reward, done_bool)

            obs = next_obs
            episode_reward += reward

            # Train agent after collecting sufficient data
            if t >= start_timesteps:
                rng, update_rng = jax.random.split(rng)
                sample = replay_buffer.sample(batch_size, rng)
                params = iterator.send(sample)

            if done:
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(
                    f"Total T: {t + 1} "
                    f"Episode Num: {episode_num + 1} "
                    f"Episode T: {episode_timesteps} "
                    f"Reward: {episode_reward:.3f} "
                )
                # Reset environment
                obs, done = env.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # Evaluate episode
            if (t + 1) % eval_freq == 0:
                evaluations.append(self.eval_policy(params))
                np.save(f"./results/{policy}/{file_name}", evaluations)
                if evaluations[-1] > best_performance:
                    best_performance = evaluations[-1]
                    best_actor_params = params
                    # if save_model: agent.save(f"./models/{policy}/{file_name}")

        # At the end, re-evaluate the policy which is presumed to be best. This ensures an un-biased estimator when
        # reporting the average best results across each run.
        params = best_actor_params
        evaluations.append(self.eval_policy(params))
        np.save(f"./results/{policy}/{file_name}", evaluations)
        print(f"Selected policy has an average score of: {evaluations[-1]:.3f}")

    def eval_policy(self, params) -> float:
        eval_env = gym.make(self.env_id)

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

        print("---------------------------------------")
        print(f"Evaluation over {self.eval_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")
        return avg_reward


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    add_arguments(PARSER)
    Trainer(**vars(PARSER.parse_args()))
