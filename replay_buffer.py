from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp
import jax
from typing import Iterable


@dataclass
class BufferItem:
    T = np.ndarray
    obs: T
    action: T
    next_obs: T
    reward: T
    not_done: T


class ReplayBuffer(object):
    """A simple container for maintaining the history of the agent."""

    def __init__(self, obs_shape: Iterable, action_shape: Iterable, max_size: int):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((max_size, *obs_shape))
        self.action = np.zeros((max_size, *action_shape))
        self.next_obs = np.zeros((max_size, *obs_shape))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

    def add(self, item: BufferItem) -> None:
        """Memory built for per-transition interaction, does not handle batch updates."""
        self.obs[self.ptr] = item.obs
        self.action[self.ptr] = item.action
        self.next_obs[self.ptr] = item.next_obs
        self.reward[self.ptr] = item.reward
        self.not_done[self.ptr] = item.not_done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int, rng: jnp.ndarray) -> BufferItem:
        """Given a JAX PRNG key, sample batch from memory."""
        ind = jax.random.randint(rng, (batch_size,), 0, self.size)

        return BufferItem(
            self.obs[ind],
            self.action[ind],
            self.next_obs[ind],
            self.reward[ind],
            self.not_done[ind],
        )
