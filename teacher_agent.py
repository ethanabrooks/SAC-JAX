from typing import Tuple

import haiku as hk
import jax
import numpy as np
from jax import numpy as jnp

import networks
from agent import Agent


class ContextEncoder(hk.Module):
    def __call__(self, obs: jnp.DeviceArray) -> jnp.DeviceArray:
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
        # s, c = jnp.split(obs, [obs_size], axis=-1)
        *shape, context_length, _, _ = obs.shape
        c = jnp.reshape(obs, (*shape, context_length, -1))
        e = encoder(c)
        e = e.mean(axis=-2)
        # se = jnp.concatenate([s, e], axis=-1)
        return e.reshape(*shape, -1)


class Actor(networks.Actor):
    def __call__(
        self, obs: jnp.DeviceArray, action_dim, *args, **kwargs
    ) -> Tuple[jnp.DeviceArray, jnp.DeviceArray]:
        obs = ContextEncoder()(obs)
        return super().__call__(obs, action_dim, *args, **kwargs)


class Critic(networks.Critic):
    def __call__(
        self, obs: jnp.DeviceArray, action_dim
    ) -> Tuple[jnp.DeviceArray, jnp.DeviceArray]:
        obs = ContextEncoder()(obs)
        return super().__call__(obs, action_dim)


class L2bAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def actor(self, x):
        return Actor()(x, action_dim=self.action_dim)

    def critic(self, x, a):
        return Critic()(x, a)
