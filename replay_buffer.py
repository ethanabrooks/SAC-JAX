from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp
import jax


@dataclass
class Sample:
    T = np.ndarray
    obs: T
    action: T
    next_obs: T
    reward: T
    not_done: T


class ReplayBuffer(object):
    """A simple container for maintaining the history of the agent."""

    def __init__(self, obs_dim: int, action_dim: int, max_size: int):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((max_size, obs_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_obs = np.zeros((max_size, obs_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        reward: float,
        done: float,
    ) -> None:
        """Memory built for per-transition interaction, does not handle batch updates."""
        self.obs[self.ptr] = obs
        self.action[self.ptr] = action
        self.next_obs[self.ptr] = next_obs
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int, rng: jnp.ndarray) -> Sample:
        """Given a JAX PRNG key, sample batch from memory."""
        ind = jax.random.randint(rng, (batch_size,), 0, self.size)

        return Sample(
            self.obs[ind],
            self.action[ind],
            self.next_obs[ind],
            self.reward[ind],
            self.not_done[ind],
        )
