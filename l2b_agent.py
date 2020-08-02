from typing import Tuple

import haiku as hk
import jax
import numpy as np
from jax import numpy as jnp

import networks
from agent import Agent


class ContextEncoder(hk.Module):
    def __init__(self, obs_size: int, context_length: int):
        super(ContextEncoder, self).__init__()
        self.context_length = context_length
        self.obs_size = obs_size

    def __call__(self, obs: np.ndarray) -> jnp.DeviceArray:
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
        d = obs.shape[-1]
        obs = obs.reshape(-1, d)
        n = obs.shape[0]
        s, c = jnp.hsplit(obs, [self.obs_size])
        c = c.reshape(n, self.context_length, -1)
        e = encoder(c)
        e = e.prod(axis=1)
        obs = jnp.concatenate([s, e], axis=-1)
        return obs.squeeze(0)


class Actor(networks.Actor):
    def __init__(self, obs_size: int, context_length: int, **kwargs):
        super(Actor, self).__init__(**kwargs)
        self.encoder = ContextEncoder(obs_size, context_length)

    def __call__(self, obs: np.ndarray) -> jnp.DeviceArray:
        obs = self.encoder(obs)
        return super().__call__(obs)


class Critic(networks.Critic):
    def __init__(self, obs_size: int, context_length: int):
        super(Critic, self).__init__()
        self.encoder = ContextEncoder(obs_size, context_length)

    def __call__(
        self, obs: np.ndarray, action: np.ndarray
    ) -> Tuple[jnp.DeviceArray, jnp.DeviceArray]:
        obs = self.encoder(obs)
        return super().__call__(obs, action)


class L2bAgent(Agent):
    def __init__(self, *args, obs_size, context_length, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_length = context_length
        self.obs_size = obs_size

    def actor(self, x):
        return Actor(
            obs_size=self.obs_size,
            context_length=self.context_length,
            action_dim=self.action_dim,
            min_action=self.min_action,
            max_action=self.max_action,
        )(x)

    def critic(self, x, a):
        return Critic(obs_size=self.obs_size, context_length=self.context_length)(x, a)
