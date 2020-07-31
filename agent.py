import itertools
from dataclasses import dataclass
from typing import Any, Tuple
import haiku as hk
import haiku._src.typing as hkt
import jax
import jax.numpy as jnp
from jax.experimental import optix
import rlax
import numpy as np
from networks import Actor, Critic
import functools

OptState = Any


@dataclass
class Params:
    t = hkt.Params
    params: t
    opt_params: t


# Perform Polyak averaging provided two network parameters and the averaging value tau.
@jax.jit
def soft_update(
    target_params: hk.Params, online_params: hk.Params, tau: float = 0.005
) -> hk.Params:
    return jax.tree_multimap(
        lambda x, y: (1 - tau) * x + tau * y, target_params, online_params
    )


class Agent(object):
    """Agent class for the TD3 algorithm. Combines both the agent and the learner functions."""

    def __init__(
        self,
        policy: str,
        action_dim: int,
        max_action: float,
        lr: float,
        discount: float,
        noise_clip: float,
        policy_noise: float,
        policy_freq: int,
    ):
        self.lr = lr
        self.discount = discount
        self.noise_clip = noise_clip
        self.policy_noise = policy_noise
        self.policy_freq = policy_freq
        self.max_action = max_action
        self.td3_update = policy == "TD3"
        self.actor_opt_init, self.actor_opt_update = optix.adam(lr)
        self.critic_opt_init, self.critic_opt_update = optix.adam(lr)
        self.actor = hk.transform(lambda x: Actor(action_dim, max_action)(x))
        self.critic = hk.transform(lambda x: Critic()(x))

    def generator(
        self, rng: jnp.ndarray, sample_state: np.ndarray,
    ):
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)
        actor_params = target_actor_params = self.actor.init(actor_rng, sample_state)
        actor_opt_state = self.actor_opt_init(actor_params)

        action = self.actor.apply(actor_params, sample_state)

        critic_params = target_critic_params = self.critic.init(
            critic_rng, jnp.concatenate((sample_state, action), 0)
        )
        critic_opt_state = self.critic_opt_init(critic_params)

        #    def update(
        #        self, replay_buffer: rb.ReplayBuffer, batch_size: int, rng: jnp.ndarray
        #    ) -> None:
        #        """
        #            Sample batch of transitions and update both the policy and critic networks.
        #            As this function contains a conditional function, periodically updating the actor, we do
        # not jit compile it.
        #        """
        for update in itertools.count():
            sample = yield actor_params
            rng, actor_rng, critic_rng = jax.random.split(rng, 3)

            # Provide each element an independent rng sample.

            # obs, action, next_obs, reward, not_done = replay_buffer.sample(
            #     batch_size, replay_rand
            # )

            critic_params, critic_opt_state = self.update_critic(
                critic_params,
                target_critic_params=target_critic_params,
                target_actor_params=target_actor_params,
                critic_opt_state=critic_opt_state,
                rng=critic_rng,
                **vars(sample),
            )

            if update % self.policy_freq == 0:
                actor_params, actor_opt_state = self.update_actor(
                    actor_params, critic_params, actor_opt_state, sample.obs
                )

                target_actor_params = soft_update(target_actor_params, actor_params)
                target_critic_params = soft_update(target_critic_params, critic_params)

    @functools.partial(jax.jit, static_argnums=0)
    def critic_1(
        self, critic_params: hk.Params, obs_action: np.ndarray
    ) -> jnp.DeviceArray:
        """Retrieves the result from a single critic network. Relevant for the actor update rule."""
        return self.critic.apply(critic_params, obs_action)[0].squeeze(-1)

    @functools.partial(jax.jit, static_argnums=0)
    def actor_loss(
        self, actor_params: hk.Params, critic_params: hk.Params, obs: np.ndarray
    ) -> jnp.DeviceArray:
        """Standard DDPG update rule based on the gradient through a single critic network."""
        action = self.actor.apply(actor_params, obs)
        return -jnp.mean(
            self.critic_1(critic_params, jnp.concatenate((obs, action), 1))
        )

    @functools.partial(jax.jit, static_argnums=0)
    def update_actor(
        self,
        actor_params: hk.Params,
        critic_params: hk.Params,
        actor_opt_state: OptState,
        obs: np.ndarray,
    ) -> Tuple[hk.Params, OptState]:
        """Learning rule (stochastic gradient descent)."""
        _, gradient = jax.value_and_grad(self.actor_loss)(
            actor_params, critic_params, obs
        )
        updates, opt_state = self.actor_opt_update(gradient, actor_opt_state)
        new_params = optix.apply_updates(actor_params, updates)
        return new_params, opt_state

    @functools.partial(jax.jit, static_argnums=0)
    def critic_loss(
        self,
        critic_params: hk.Params,
        target_critic_params: hk.Params,
        target_actor_params: hk.Params,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        reward: np.ndarray,
        not_done: np.ndarray,
        rng: jnp.ndarray,
    ) -> jnp.DeviceArray:
        """
            TD3 adds truncated Gaussian noise to the policy while training the critic.
            Can be seen as a form of 'Exploration Consciousness' https://arxiv.org/abs/1812.05551 or simply as a
            regularization scheme.
            As this helps stabilize the critic, we also use this for the DDPG update rule.
        """
        noise = (jax.random.normal(rng, shape=action.shape) * self.policy_noise).clip(
            -self.noise_clip, self.noise_clip
        )

        # Make sure the noisy action is within the valid bounds.
        next_action = (self.actor.apply(target_actor_params, next_obs) + noise).clip(
            -self.max_action, self.max_action
        )

        next_q_1, next_q_2 = self.critic.apply(
            target_critic_params, jnp.concatenate((next_obs, next_action), 1)
        )
        if self.td3_update:
            next_q = jax.lax.min(next_q_1, next_q_2)
        else:
            # Since the actor uses Q_1 for training, setting this as the target for the critic updates is sufficient to
            # obtain an equivalent update.
            next_q = next_q_1
        # Cut the gradient from flowing through the target critic. This is more efficient, computationally.
        target_q = jax.lax.stop_gradient(reward + self.discount * next_q * not_done)
        q_1, q_2 = self.critic.apply(critic_params, jnp.concatenate((obs, action), 1))

        return jnp.mean(rlax.l2_loss(q_1, target_q) + rlax.l2_loss(q_2, target_q))

    @functools.partial(jax.jit, static_argnums=0)
    def update_critic(
        self,
        critic_params: hk.Params,
        target_critic_params: hk.Params,
        target_actor_params: hk.Params,
        critic_opt_state: OptState,
        rng: jnp.ndarray,
        **kwargs,
    ) -> Tuple[hk.Params, OptState]:
        """Learning rule (stochastic gradient descent)."""
        _, gradient = jax.value_and_grad(self.critic_loss)(
            critic_params,
            target_critic_params=target_critic_params,
            target_actor_params=target_actor_params,
            rng=rng,
            **kwargs,
        )
        updates, opt_state = self.critic_opt_update(gradient, critic_opt_state)
        new_params = optix.apply_updates(critic_params, updates)
        return new_params, opt_state

    @functools.partial(jax.jit, static_argnums=0)
    def policy(self, actor_params: hk.Params, obs: np.ndarray) -> jnp.DeviceArray:
        return self.actor.apply(actor_params, obs)
