import json
from pathlib import Path

from hyperopt import hp


def get_config(name):
    path = Path("configs", name).with_suffix(".json")
    if path.exists():
        with path.open() as f:
            config = json.load(f)
            del config["use_tune"]
            return config
    return configs[name]


def copy_args(d, prefix):
    for k, v in d.items():
        yield prefix + k, v


def small_values(start, stop):
    return [j for i in range(start, stop) for j in ((10 ** -i), 5 * (10 ** -i))]


def medium_values(start, stop):
    return [2 ** i for i in range(start, stop)]


def big_values(start, stop):
    return [j for i in range(start, stop) for j in ((10 ** i), 5 * (10 ** i))]


pendulum = {
    "batch_size": 256,
    "discount": 0.99,
    "env_id": "Pendulum-v0",
    "eval_freq": 5000.0,
    "expl_noise": 0.01,
    "lr": 0.001,
    "max_timesteps": 30000,
    "noise_clip": 0.01,
    "policy": "TD3",
    "policy_freq": 1,
    "policy_noise": 0.2,
    "replay_size": 200000,
    "seed": 3,
    "start_timesteps": 10,
}
debug4 = get_config("debug4")
search = dict(
    batch_size=hp.choice("batch_size", medium_values(6, 10)),
    expl_noise=hp.choice("expl_noise", small_values(1, 3)),
    noise_clip=hp.choice("noise_clip", small_values(1, 3)),
    lr=hp.choice("lr", small_values(2, 5) + [3e-4]),
    policy_freq=hp.choice("policy_freq", [1, 2, 3]),
    seed=hp.randint("seed", 20),
    start_timesteps=hp.choice("start_timesteps", big_values(0, 2)),
)
outer_search = dict(
    outer_batch_size=hp.choice("outer_batch_size", medium_values(7, 9)),
    outer_expl_noise=hp.choice("outer_expl_noise", small_values(1, 3)),
    outer_noise_clip=hp.choice("outer_noise_clip", small_values(1, 3)),
    outer_lr=hp.choice("outer_lr", small_values(1, 2)),
    outer_policy_freq=hp.choice("outer_policy_freq", [1, 2]),
    outer_seed=hp.randint("outer_seed", 20),
    outer_start_timesteps=1,
)
l2b_search = dict(
    context_length=hp.choice("context_length", [40, 50, 60]),
    sample_done_prob=hp.choice("sample_done_prob", [0.1, 0.3, 0.5]),
    update_freq=hp.choice("update_freq", [40, 50, 75, 100]),
    inner_env_id=None,
    **dict(copy_args(search, "inner_")),
    **outer_search,
)
l2b_search.update(
    inner_max_timesteps=hp.choice("inner_max_timesteps", [200, 500, 1000, 1500]),
    outer_max_timesteps=4000,
)
debug_l2b = dict(
    context_length=2,
    sample_done_prob=0,
    update_freq=10000,  # TODO
    **dict(copy_args(pendulum, "inner_")),
    **dict(copy_args(pendulum, "outer_")),
)
debug_l2b.update(inner_max_timesteps=15000, outer_max_timesteps=15000)
configs = dict(
    search=search,
    pendulum=pendulum,
    debug4=debug4,
    l2b_search=l2b_search,
    debug_l2b=debug_l2b,
)
