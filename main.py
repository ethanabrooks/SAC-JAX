"""
    Credits: https://github.com/sfujim/TD3
"""
import argparse
from pathlib import Path
from typing import Any

from trainer import Trainer

OptState = Any


def add_arguments(parser):
    parser.add_argument(
        "--batch-size", type=int, help="Batch size for both actor and critic",
    )
    parser.add_argument("--discount", type=float)  # Discount factor
    parser.add_argument(
        "--env", dest="env_id", help="OpenAI gym environment name",
    )
    parser.add_argument(
        "--eval-freq", type=int, help="How often (time steps) we evaluate"
    )
    parser.add_argument(
        "--initial-log-alpha", type=float, help="How often (time steps) we evaluate",
    )
    parser.add_argument("--lr", type=float)  # Optimizer learning rates
    parser.add_argument(
        "--max-timesteps", type=int, help="Max time steps to run environment",
    )
    parser.add_argument(
        "--policy-freq", type=int, help="Frequency of delayed policy updates"
    )
    parser.add_argument("--replay-size", type=int, help="Size of the replay buffer")
    parser.add_argument("--seed", type=int, help="Sets Gym, PyTorch and Numpy seeds")
    parser.add_argument(
        "--start-timesteps", type=int, help="Time steps initial random policy is used",
    )
    parser.add_argument("--save-dir", type=Path, help="directory to save model")
    parser.add_argument(
        "config",
        help="name of config file to load from configs/ directory or from configs.configs "
        "dict",
    )
    parser.add_argument("--name", help="name of experiment (for tune)")
    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        help="Number of times to sample from the hyperparameter space. See tune docs for details: "
        "https://docs.ray.io/en/latest/tune/api_docs/execution.html?highlight=run#ray.tune.run",
    )
    parser.add_argument(
        "--tune",
        dest="use_tune",
        action="store_true",
        help="Use tune for logging and hyperparameter search",
    )
    parser.add_argument(
        "--cpus-per-trial",
        "-c",
        type=int,
        default=6,
        help="This parameters is used by tune to " "determine how many runs to launch.",
    )
    parser.add_argument(
        "--gpus-per-trial",
        "-g",
        type=int,
        default=1,
        help="This parameters is used by tune to " "determine how many runs to launch.",
    )


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    add_arguments(PARSER)
    Trainer.main(**vars(PARSER.parse_args()))
