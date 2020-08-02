import argparse
import re
import numpy as np

import jax
from gym.wrappers import TimeLimit

from args import add_arguments
from l2b_agent import L2bAgent
from l2b_env import L2bEnv, CatObsSpace
from replay_buffer import ReplayBuffer, BufferItem
from trainer import Trainer


class DoubleReplayBuffer(ReplayBuffer):
    def __init__(self, sample_done_prob, **kwargs):
        super().__init__(**kwargs)
        self.sample_done_prob = sample_done_prob
        self.done_buffer = ReplayBuffer(**kwargs)

    def add(self, item: BufferItem) -> None:
        if item.not_done:
            super().add(item)
        else:
            self.done_buffer.add(item)

    def sample(self, *args, rng, **kwargs) -> BufferItem:
        if jax.random.choice(
            rng, 2, p=[1 - self.sample_done_prob, self.sample_done_prob]
        ):
            return self.done_buffer.sample(*args, rng=rng, **kwargs)
        return super().sample(*args, rng=rng, **kwargs)


class OuterTrainer(Trainer):
    def __init__(self, make_env, build_agent, sample_done_prob, **trainer_args):
        self.sample_done_prob = sample_done_prob
        self._make_env = make_env
        self._build_agent = build_agent
        super().__init__(**trainer_args, env_id=None)

    def report(self, **kwargs):
        super().report(**{"outer_" + k: v for k, v in kwargs.items()})

    @classmethod
    def run(cls, config: dict):
        def run(
            sample_done_prob,
            use_tune,
            update_freq,
            context_length,
            max_episode_steps,
            env_id,
            **kwargs
        ):
            inner = re.compile(r"^inner_(.*)")
            outer = re.compile(r"^outer_(.*)")

            def get_args(pattern):
                for k, v in kwargs.items():
                    if pattern.match(k):
                        yield pattern.sub(r"\1", k), v

            trainer_args = dict(
                get_args(inner), sample_done_prob=sample_done_prob, use_tune=use_tune,
            )

            def make_env():
                return TimeLimit(
                    CatObsSpace(
                        L2bEnv(
                            **dict(get_args(outer)),
                            update_freq=update_freq,
                            use_tune=use_tune,
                            context_length=context_length,
                            env_id=env_id,
                        )
                    ),
                    max_episode_steps=max_episode_steps,
                )

            def build_agent(obs_size, **_kwargs):
                return L2bAgent(
                    obs_size=obs_size, context_length=context_length, **_kwargs
                )

            cls(make_env=make_env, build_agent=build_agent, **trainer_args).train()

        run(**config)

    def build_replay_buffer(self):
        return DoubleReplayBuffer(
            obs_shape=self.env.observation_space.shape,
            action_shape=self.env.action_space.shape,
            max_size=self.replay_size,
            sample_done_prob=self.sample_done_prob,
        )

    def make_env(self):
        return self._make_env()

    def build_agent(self, **kwargs):
        obs_size = np.prod(self.env.get_inner_env().observation_space.shape)
        return self._build_agent(
            obs_size=obs_size,
            max_action=self.max_action,
            min_action=self.min_action,
            action_dim=self.action_dim,
            **kwargs,
        )


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
    PARSER.add_argument("--sample-done-prob", type=float, default=0.3, double=False)
    PARSER.add_argument("--update-freq", type=int, default=1, double=False)
    PARSER.add_argument("--context-length", type=int, default=100, double=False)
    PARSER.add_argument("--max-episode-steps", type=int, default=10000, double=False)
    add_arguments(PARSER)
    OuterTrainer.main(**vars(PARSER.parse_args()))
