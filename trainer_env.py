import gym

from trainer import Trainer


class TrainerEnv(gym.Env, Trainer):
    def step(self, action):
        pass

    def reset(self):
        pass

    def generator(self):
        pass

    def render(self, mode="human"):
        pass
