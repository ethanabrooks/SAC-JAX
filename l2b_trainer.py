import argparse
import re
import numpy as np

from gym.wrappers import TimeLimit

from args import add_arguments
from l2b_agent import L2bAgent
from l2b_env import L2bEnv, CatObsSpace, DoubleReplayBuffer
from trainer import Trainer


class L2bTrainer(Trainer):
    def __init__(self, inner_args, sample_done_prob, env_id=None, **outer_args):
        self.sample_done_prob = sample_done_prob
        self.inner_args = inner_args
        super().__init__(**outer_args, env_id=env_id)

    def report(self, **kwargs):
        super().report(
            **{k if k == "final_reward" else "outer_" + k: v for k, v in kwargs.items()}
        )

    @classmethod
    def run(cls, config: dict):
        def run(
            context_length, sample_done_prob, update_freq, use_tune, alpha, **kwargs
        ):
            inner = re.compile(r"^inner_(.*)")
            outer = re.compile(r"^outer_(.*)")

            def get_args(pattern):
                for k, v in kwargs.items():
                    if pattern.match(k):
                        yield pattern.sub(r"\1", k), v

            inner_args = dict(
                get_args(inner),
                context_length=context_length,
                update_freq=update_freq,
                use_tune=use_tune,
                alpha=alpha,
            )
            outer_args = dict(
                **dict(get_args(outer)),
                sample_done_prob=sample_done_prob,
                context_length=context_length,
                use_tune=use_tune,
            )
            cls(**outer_args, inner_args=inner_args).train()

        run(**config)

    def make_env(self):
        def make_env(max_timesteps, **kwargs):
            return TimeLimit(
                CatObsSpace(L2bEnv(**kwargs, max_timesteps=max_timesteps)),
                max_episode_steps=max_timesteps,
            )

        return make_env(**self.inner_args)

    def build_agent(self, **kwargs):
        obs_size = np.prod(self.env.get_inner_env().observation_space.shape)
        return L2bAgent(
            obs_size=obs_size,
            max_action=self.max_action,
            min_action=self.min_action,
            action_dim=self.action_dim,
            **kwargs,
        )

    def build_replay_buffer(self):
        return DoubleReplayBuffer(
            obs_shape=self.env.observation_space.shape,
            action_shape=self.env.action_space.shape,
            max_size=self.replay_size,
            sample_done_prob=self.sample_done_prob,
        )

    def eval_policy(self, params) -> float:
        return 1


class DoubleArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, single_list, **kwargs):
        self.single_list = single_list + ["-h", "--help"]
        super().__init__(*args, **kwargs)

    def add_argument(self, *args, **kwargs):
        name, *args = args
        if name in self.single_list:
            super().add_argument(name, *args, **kwargs)
        else:
            pattern = re.compile(r"(?:--)?(.*)")
            name = pattern.match(name).group(1)
            name1 = "--inner-" + name
            super().add_argument(name1, *args, **kwargs)
            name2 = "--outer-" + name
            super().add_argument(name2, *args, **kwargs)


if __name__ == "__main__":
    PARSER = DoubleArgumentParser(
        conflict_handler="resolve",
        single_list=[
            "config",
            "--alpha",
            "--no-tune",
            "--num-samples",
            "--name",
            "--best",
            "--sample-done-prob",
            "--update-freq",
            "--context-length",
            "--max-episode-steps",
            "--cpus-per-trial",
            "--gpus-per-trial",
        ],
    )
    PARSER.add_argument("--sample-done-prob", type=float, default=0.3)
    PARSER.add_argument("--update-freq", type=int, default=1)
    PARSER.add_argument("--context-length", type=int, default=100)
    PARSER.add_argument("--alpha", type=float, default=0.1)
    add_arguments(PARSER)
    L2bTrainer.main(**vars(PARSER.parse_args()))
