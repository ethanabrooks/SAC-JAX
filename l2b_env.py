import gym
import jax

from trainer import Trainer


class L2bEnv(Trainer, gym.Env):
    def __init__(self):
        super().__init__()
        self.iterator = None
        self.i = 0
        self.rng = jax.random.PRNGKey(self.seed)

    def step(self, action):
        s, params = next(self.iterator)
        self.i += 1
        t = self.i == self.max_timesteps
        r = self.eval_policy(params) if t else 0
        return s, r, t, {}

    def reset(self):
        self.rng, rng = jax.random.split(self.rng)
        self.iterator = self.generator(rng)
        s, _ = next(self.iterator)
        return s

    def render(self, mode="human"):
        pass
