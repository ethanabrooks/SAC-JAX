from typing import Tuple, Union

import haiku as hk
import jax
import numpy as np
from jax import numpy as jnp, nn
from jax.nn import sigmoid
from jax.random import PRNGKey

"""
    Actor and Critic networks defined as in the TD3 paper (Fujimoto et. al.) https://arxiv.org/abs/1802.09477
"""

T = Union[np.ndarray, jnp.DeviceArray]


class Actor(hk.Module):
    def __init__(
        self, action_dim: int, min_action: T, max_action: T, noise_clip: float
    ):
        super(Actor, self).__init__()
        self.noise_clip = noise_clip
        self.action_dim = action_dim
        self.min_action = min_action
        self.max_action = max_action

    def __call__(self, obs: T) -> Tuple[jnp.DeviceArray, jnp.DeviceArray]:
        actor_net = hk.Sequential(
            [
                hk.Flatten(),
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
                jax.nn.relu,
                hk.Linear(
                    2 * self.action_dim,
                    w_init=hk.initializers.VarianceScaling(
                        scale=2.0, distribution="uniform"
                    ),
                ),
            ]
        )
        log_sig_min = -20
        log_sig_max = 2
        out = actor_net(obs)
        mu, log_sig = jnp.split(out, 2, axis=-1)
        log_sig = nn.softplus(log_sig)
        log_sig = jnp.clip(log_sig, log_sig_min, log_sig_max)

        return mu, log_sig


class Critic(hk.Module):
    def __init__(self):
        super(Critic, self).__init__()

    def __call__(self, obs: T, action: T) -> Tuple[jnp.DeviceArray, jnp.DeviceArray]:
        obs_action = jnp.concatenate((obs, action), -1)

        def critic_net():
            return hk.Sequential(
                [
                    hk.Flatten(),
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
                    jax.nn.relu,
                    hk.Linear(
                        1,
                        w_init=hk.initializers.VarianceScaling(
                            scale=2.0, distribution="uniform"
                        ),
                    ),
                ]
            )

        critic_net_1 = critic_net()

        critic_net_2 = critic_net()

        return critic_net_1(obs_action), critic_net_2(obs_action)
