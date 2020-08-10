import argparse

import numpy as np
from gym.wrappers import TimeLimit

from main import add_arguments
from teacher_agent import L2bAgent
from teacher_env import TeacherEnv
from trainer import Trainer


class TeacherTrainer(Trainer):
    def __init__(self, make_env, *args, **kwargs):
        self._make_env = make_env
        kwargs.update(env_id=None)
        super().__init__(*args, **kwargs)

    @classmethod
    def run(
        cls,
        batches: int,
        choices: int,
        context_length: int,
        inner_timesteps: int,
        report_freq: int,
        std: float,
        use_tune,
        **kwargs
    ):
        def make_env():
            return TimeLimit(
                TeacherEnv(
                    batches=batches,
                    context_length=context_length,
                    std=std,
                    choices=choices,
                    inner_timesteps=inner_timesteps,
                    use_tune=use_tune,
                    report_freq=report_freq,
                ),
                max_episode_steps=inner_timesteps // context_length - 1,
            )

        cls(**kwargs, use_tune=use_tune, make_env=make_env,).train()

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
    PARSER.add_argument("--choices", "-d", type=int, default=50)
    PARSER.add_argument("--batches", "-b", type=int, default=100)
    PARSER.add_argument("--inner-timesteps", "-T", type=int, default=int(1e3))
    PARSER.add_argument("--report-freq", type=int, default=10)
    add_arguments(PARSER)
    TeacherTrainer.main(**vars(PARSER.parse_args()))
