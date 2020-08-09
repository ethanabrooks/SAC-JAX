import argparse
import re
import numpy as np

from gym.wrappers import TimeLimit

from teacher_agent import L2bAgent
from teacher_env import TeacherEnv
from main import add_arguments
from trainer import Trainer


class TeacherTrainer(Trainer):
    def __init__(self, make_env, *args, **kwargs):
        self._make_env = make_env
        super().__init__(*args, env_id=None, **kwargs)

    @classmethod
    def run(
        cls, context_length: int, std: float, choices: int, inner_steps: int, **kwargs
    ):
        def make_env():
            return TimeLimit(
                TeacherEnv(
                    context_length=context_length,
                    std=std,
                    choices=choices,
                    inner_steps=inner_steps,
                ),
                max_episode_steps=inner_steps,
            )

        cls(**kwargs, context_length=context_length, make_env=make_env).train()

    def make_env(self):
        return self._make_env()

    def build_agent(self, **kwargs):
        return L2bAgent(
            max_action=self.env.action_space.high,
            min_action=self.env.action_space.low,
            action_dim=int(np.prod(self.env.action_space.shape)),
            **kwargs,
        )


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--context-length", type=int, default=10)
    PARSER.add_argument("--std", type=float, default=1.0)
    PARSER.add_argument("--choices", type=int, default=10)
    PARSER.add_argument("--inner-steps", type=int, default=int(1e4))
    add_arguments(PARSER)
    TeacherTrainer.main(**vars(PARSER.parse_args()))
