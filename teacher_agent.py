from typing import Tuple

import haiku as hk
import jax
import numpy as np
from jax import numpy as jnp

import networks
from agent import Agent


class ContextEncoder(hk.Module):
    def __call__(self, obs: np.ndarray, context_length, obs_size) -> jnp.DeviceArray:
        encoder = hk.Sequential(
            [
                hk.Linear(
                    256,
                    w_init=hk.initializers.VarianceScaling(
                        scale=2.0, distribution="uniform"
                    ),
                ),
                jax.nn.relu,
                hk.Linear(
                    256,
                    w_init=hk.initializers.VarianceScaling(
                        scale=2.0, distribution="uniform"
                    ),
                ),
            ]
        )
        s, c = jnp.split(obs, [obs_size], axis=-1)
        c = c.reshape(*s.shape[:-1], context_length, -1)
        e = encoder(c)
        e = e.mean(axis=-1)
        se = jnp.concatenate([s, e], axis=-1)
        return se


class Actor(networks.Actor):
    def __call__(
        self, obs: np.ndarray, action_dim, *args, **kwargs
    ) -> Tuple[jnp.DeviceArray, jnp.DeviceArray]:
        obs = ContextEncoder()(obs, *args, **kwargs)
        return super().__call__(obs, action_dim)


class Critic(networks.Critic):
    def __call__(
        self, obs: np.ndarray, action_dim, *args, **kwargs
    ) -> Tuple[jnp.DeviceArray, jnp.DeviceArray]:
        obs = ContextEncoder()(obs, *args, **kwargs)
        return super().__call__(obs, action_dim)


class L2bAgent(Agent):
    def __init__(self, *args, obs_size, context_length, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_length = context_length
        self.obs_size = obs_size

    def actor(self, x):
        return Actor()(
            x, action_dim=self.action_dim, context_length=self.context_length
        )

    def critic(self, x, a):
        return Critic()(x, a, context_length=self.context_length)
