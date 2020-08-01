import argparse
import re
import sys
from functools import partial

from args import add_arguments
from trainer import Trainer
from l2b_env import L2bEnv


class OuterTrainer(Trainer):
    def __init__(self, trainer_args, env_args):
        self._make_env = lambda: L2bEnv(**env_args)
        super().__init__(**trainer_args)

    @classmethod
    def run(cls, config):
        inner = re.compile(r"^inner_(.*)")
        outer = re.compile(r"^outer_(.*)")

        def get_kwargs(pattern):
            for k, v in config.items():
                if pattern.match(k):
                    yield pattern.sub(r"\1", k), v

        trainer_args = dict(get_kwargs(outer))
        env_args = dict(get_kwargs(inner))
        cls(trainer_args, env_args).train()

    def make_env(self):
        self._make_env()


def add_outer_arguments(parser):
    parser.add_argument(
        "--outer-batch-size", default=256, type=int
    )  # Batch size for both actor and critic
    parser.add_argument(
        "--outer-buffer-size", default=2e6, type=int
    )  # Max size of replay buffer
    parser.add_argument("--outer-discount", default=0.99)  # Discount factor
    parser.add_argument(
        "--outer-eval-freq", default=5e3, type=int
    )  # How often (time steps) we evaluate
    parser.add_argument(
        "--outer-eval-episodes", default=10, type=int
    )  # How often (time steps) we evaluate
    parser.add_argument(
        "--outer-learning-rate", default=3e-4, type=float
    )  # Noise added to target policy during critic update
    parser.add_argument(
        "--outer-actor-freq", default=2, type=int
    )  # Frequency of delayed policy updates
    parser.add_argument(
        "--outer-seed", default=0, type=int
    )  # Sets DM control and JAX seeds
    parser.add_argument(
        "--outer-start-time-steps", default=1e4, type=int
    )  # Time steps initial random policy is used
    parser.add_argument("--outer-tau", default=0.005)  # Target network update rate
    parser.add_argument("--outer-train-steps", default=1, type=int)


class DoubleArgumentParser(argparse.ArgumentParser):
    def add_argument(self, *args, double=True, **kwargs):
        name, *args = args
        if double and name.startswith("--"):
            pattern = re.compile(r"--(.*)")
            name1 = pattern.sub(r"--inner-\1", name)
            super().add_argument(name1, *args, **kwargs)
            name2 = pattern.sub(r"--outer-\1", name)
            super().add_argument(name2, *args, **kwargs)
        else:
            super().add_argument(name, *args, **kwargs)


if __name__ == "__main__":
    PARSER = DoubleArgumentParser(conflict_handler="resolve")
    PARSER.add_argument("config", double=False)
    PARSER.add_argument(
        "--no-tune", dest="use_tune", action="store_false", double=False
    )
    PARSER.add_argument("--num-samples", type=int, double=False)
    PARSER.add_argument("--name", double=False)
    PARSER.add_argument("--best", double=False)
    add_arguments(PARSER)
    OuterTrainer.main(**vars(PARSER.parse_args()))
