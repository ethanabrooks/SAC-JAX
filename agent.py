import functools
import itertools

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax import nn
from jax.experimental import optix
from jax.random import PRNGKey

from networks import Actor, Critic
from util import single_mse, gaussian_likelihood, double_mse, soft_update


class Agent(object):
    """Agent class for the TD3 algorithm. Combines both the agent and the learner functions."""

    def __init__(
        self,
        policy: str,
        min_action: np.array,
        max_action: np.array,
        action_dim: int,
        lr: float,
        discount: float,
        noise_clip: float,
        policy_noise: float,
        policy_freq: int,
    ):
        self.min_action = min_action
        self.action_dim = action_dim
        self.max_action = max_action
        self.lr = lr
        self.discount = discount
        self.noise_clip = noise_clip
        self.policy_noise = policy_noise
        self.policy_freq = policy_freq
        self.td3_update = policy == "TD3"
        self.actor_opt_init, self.actor_opt_update = optix.adam(lr)
        self.critic_opt_init, self.critic_opt_update = optix.adam(lr)
        self.actor = hk.without_apply_rng(hk.transform(self.actor))
        self.critic = hk.without_apply_rng(hk.transform(self.critic))

    def actor(self, x):
        return Actor()(x, action_dim=self.action_dim)

    @staticmethod
    def critic(x, a):
        return Critic()(x, a)

    def train_loop(
        self, rng: jnp.ndarray, sample_obs: np.ndarray, sample_action: np.ndarray,
    ):
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)
        actor_params = target_actor_params = self.actor.init(actor_rng, sample_obs)
        actor_opt_state = self.actor_opt_init(actor_params)

        critic_params = target_critic_params = self.critic.init(
            critic_rng, sample_obs, sample_action
        )
        critic_opt_state = self.critic_opt_init(critic_params)

        for update in itertools.count():
            sample = yield actor_params
            rng, actor_rng, critic_rng = jax.random.split(rng, 3)

            target_Q = jax.lax.stop_gradient(
                self.get_td_target(
                    next_obs=sample.next_obs,
                    reward=sample.reward,
                    not_done=1 - sample.done,
                    actor=actor_params,
                    critic_target=target_critic_params,
                )
            )

            critic_params, critic_opt_state = self.update_critic(
                params=critic_params,
                opt_state=critic_opt_state,
                state=sample.obs,
                action=sample.action,
                target_Q=target_Q,
            )

            if update % self.policy_freq == 0:
                actor_params, actor_opt_state = self.update_actor(
                    rng=actor_rng,
                    params=actor_params,
                    critic=critic_params,
                    state=sample.obs,
                    opt_state=actor_opt_state,
                )

                target_actor_params = soft_update(target_actor_params, actor_params)
                target_critic_params = soft_update(target_critic_params, critic_params)

    @functools.partial(jax.jit, static_argnums=0)
    def update_actor(self, rng, params, critic, state, opt_state):
        def loss(actor_params):
            mu, log_sig = self.actor.apply(actor_params, state)
            pi = mu + jax.random.normal(rng, mu.shape) * jnp.exp(log_sig)
            log_p = gaussian_likelihood(pi, mu, log_sig)
            pi = jnp.tanh(pi)
            log_p -= jnp.sum(jnp.log(nn.relu(1 - pi ** 2) + 1e-6), axis=1)
            actor_action = self.postprocess_action(pi)

            q1, q2 = self.critic.apply(critic, state, actor_action)
            min_q = jnp.minimum(q1, q2)
            actor_loss = single_mse(log_p, min_q)
            return jnp.mean(actor_loss), log_p

        gradient, log_p = jax.grad(loss, has_aux=True)(params)
        updates, opt_state = self.actor_opt_update(gradient, opt_state)
        new_params = optix.apply_updates(params, updates)
        return new_params, opt_state

    @functools.partial(jax.jit, static_argnums=0)
    def update_critic(self, params, opt_state, state, action, target_Q):
        def loss(critic):
            current_Q1, current_Q2 = self.critic.apply(critic, state, action)

            critic_loss = double_mse(current_Q1, current_Q2, target_Q)
            return jnp.mean(critic_loss)

        gradient = jax.grad(loss)(params)
        updates, opt_state = self.critic_opt_update(gradient, opt_state)
        new_params = optix.apply_updates(params, updates)
        return new_params, opt_state

    def postprocess_action(self, pi):
        return jnp.tanh(pi) * (self.max_action - self.min_action) + self.min_action

    @functools.partial(jax.jit, static_argnums=0)
    def policy(
        self, actor_params: hk.Params, obs: np.ndarray, rng: PRNGKey = None
    ) -> jnp.DeviceArray:
        mu, log_sig = self.actor.apply(actor_params, obs)
        pi = mu
        if rng is not None:
            pi += jax.random.normal(rng, mu.shape) * jnp.exp(log_sig)
        return self.postprocess_action(pi)

    @functools.partial(jax.jit, static_argnums=0)
    def get_td_target(
        self, next_obs, reward, not_done, actor, critic_target,
    ):
        mu, _ = self.actor.apply(actor, next_obs)
        next_action = 2 * jnp.tanh(mu)

        target_Q1, target_Q2 = self.critic.apply(critic_target, next_obs, next_action)
        target_Q = jnp.minimum(target_Q1, target_Q2)
        target_Q = reward + not_done * self.discount * target_Q

        return target_Q
