from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp
import jax
from typing import Iterable


@dataclass
class Sample:
    T = np.ndarray
    obs: T
    action: T
    next_obs: T
    reward: T
    done: T


@dataclass
class Step(Sample):
    pass


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
        self.done = np.zeros((max_size, 1))

    def add(self, sample: Sample) -> None:
        """Memory built for per-transition interaction, does not handle batch updates."""
        self.obs[self.ptr] = sample.obs
        self.action[self.ptr] = sample.action
        self.next_obs[self.ptr] = sample.next_obs
        self.reward[self.ptr] = sample.reward
        self.done[self.ptr] = sample.done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int, rng: jnp.ndarray) -> Sample:
        """Given a JAX PRNG key, sample batch from memory."""
        if self.size < batch_size:
            raise RuntimeError
        ind = jax.random.randint(rng, (batch_size,), 0, self.size)

        return Sample(
            self.obs[ind],
            self.action[ind],
            self.next_obs[ind],
            self.reward[ind],
            self.done[ind],
        )
