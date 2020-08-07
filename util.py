import haiku as hk
import jax
import numpy as np
from jax import numpy as jnp


@jax.vmap
def single_mse(log_p, min_q):
    return (log_p - min_q).mean()


@jax.jit
def gaussian_likelihood(sample, mu, log_sig):
    pre_sum = -0.5 * (
        ((sample - mu) / (jnp.exp(log_sig) + 1e-6)) ** 2
        + 2 * log_sig
        + jnp.log(2 * np.pi)
    )
    return jnp.sum(pre_sum, axis=1)


@jax.vmap
def double_mse(q1, q2, qt):
    return jnp.square(qt - q1).mean() + jnp.square(qt - q2).mean()


# Perform Polyak averaging provided two network parameters and the averaging value tau.
@jax.jit
def soft_update(
    target_params: hk.Params, online_params: hk.Params, tau: float = 0.005
) -> hk.Params:
    return jax.tree_multimap(
        lambda x, y: (1 - tau) * x + tau * y, target_params, online_params
    )
