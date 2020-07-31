"""
    Credits: https://github.com/sfujim/TD3
"""

import argparse
from typing import Any, Generator, Tuple

import gym
import haiku._src.typing as hkt
import jax
import jax.numpy as jnp

from agent import Agent
from args import add_arguments
from replay_buffer import ReplayBuffer

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

        print("---------------------------------------")
        print(f"Policy: {policy}, Env: {env_id}, Seed: {seed}")
        print("---------------------------------------")

        # if save_model and not os.path.exists("./models/" + policy):
        #     os.makedirs("./models/" + policy)

        self.env = gym.make(env_id)
        self.env.seed(seed)

        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])

        self.agent = Agent(
            policy=policy,
            action_dim=self.action_dim,
            max_action=self.max_action,
            lr=lr,
            discount=discount,
            noise_clip=noise_clip,
            policy_noise=policy_noise,
            policy_freq=policy_freq,
        )

    def generator(
        self, rng,
    ) -> Generator[Tuple[jnp.ndarray, hkt.Params], jnp.ndarray, None]:
        obs, done = self.env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0

        iterator = self.agent.generator(rng, sample_obs=obs)

        params = next(iterator)

        replay_buffer = ReplayBuffer(
            self.obs_dim, self.action_dim, max_size=self.replay_size
        )

        # Evaluate untrained policy.
        # We evaluate for 100 episodes as 10 episodes provide a very noisy estimation in some domains.
        evaluations = [self.eval_policy(params)]
        best_performance = evaluations[-1]
        best_actor_params = params
        # if save_model: agent.save(f"./models/{policy}/{file_name}")

        for t in range(int(self.max_timesteps)):

            episode_timesteps += 1

            action = yield obs, params

            # Perform action
            next_obs, reward, done, _ = self.env.step(action)
            # This 'trick' converts the finite-horizon task into an infinite-horizon one. It does change the problem
            # we are solving, however it has been observed empirically to work pretty well. noinspection
            # noinspection PyProtectedMember
            steps = self.env._max_episode_steps
            done_bool = float(done) if episode_timesteps < steps else 0

            # Store data in replay buffer
            replay_buffer.add(obs, action, next_obs, reward, done_bool)

            obs = next_obs
            episode_reward += reward

            # Train agent after collecting sufficient data
            if t >= self.start_timesteps:
                rng, update_rng = jax.random.split(rng)
                sample = replay_buffer.sample(self.batch_size, rng)
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
                obs, done = self.env.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

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

    def train(self):
        rng = jax.random.PRNGKey(self.seed)
        iterator = self.generator(rng)
        obs, params = next(iterator)
        rng = jax.random.PRNGKey(self.seed)

        for _ in range(self.start_timesteps):
            obs, params = iterator.send(self.env.action_space.sample())
        while True:
            # Select action randomly or according to policy
            rng, noise_rng = jax.random.split(rng)
            action = (
                self.agent.policy(params, obs)
                + jax.random.normal(noise_rng, (self.action_dim,))
                * self.max_action
                * self.expl_noise
            ).clip(-self.max_action, self.max_action)
            try:
                obs, params = iterator.send(action)
            except StopIteration:
                print("Done training")
                exit()

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
    Trainer(**vars(PARSER.parse_args())).train()
