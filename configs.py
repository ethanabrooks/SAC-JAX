import json
from pathlib import Path

from hyperopt import hp


def get_config(name):
    path = Path("configs", name).with_suffix(".json")
    if path.exists():
        with path.open() as f:
            config = json.load(f)
            del config["use_tune"]
            del config["max_timesteps"]
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
single = {
    "batch_size": 128,
    "discount": 0.99,
    "eval_freq": 5000.0,
    "expl_noise": 0.5,
    "lr": 0.0005,
    "noise_clip": 0.1,
    "policy": "TD3",
    "policy_freq": 1,
    "policy_noise": 0.2,
    "replay_size": 200000,
    "seed": 8,
    "start_timesteps": 5,
}
outer_search = dict(
    context_length=hp.choice("context_length", [10, 50, 100, 200]),
    sample_done_prob=hp.choice("sample_done_prob", [0.1, 0.3, 0.5]),
    update_freq=hp.choice("update_freq", [10, 20, 30, 40, 50]),
    **dict(copy_args(search, "outer_")),
    **dict(copy_args(debug4, "inner_")),
)
outer_search.update(inner_max_timesteps=hp.choice("inner_max_timesteps", [500, 1000]),)
configs = dict(
    search=search,
    pendulum={
        "batch_size": 256,
        "discount": 0.99,
        "env_id": "Pendulum-v0",
        "eval_freq": 5000.0,
        "expl_noise": 0.01,
        "lr": 0.001,
        "max_timesteps": 15000,
        "noise_clip": 0.01,
        "policy": "TD3",
        "policy_freq": 1,
        "policy_noise": 0.2,
        "replay_size": 200000,
        "seed": 3,
        "start_timesteps": 10,
    },
    debug4=debug4,
    outer_search=outer_search,
    # TODO
    single=single,
    # TODO
    double=dict(
        **dict(copy_args(single, "outer_")),
        **dict(copy_args(single, "inner_")),
        update_freq=20,
        context_length=200,
    ),
)
