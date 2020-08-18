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

search = dict(
    batch_size=hp.choice("batch_size", medium_values(5, 10)),
    lr=hp.choice("lr", small_values(2, 5) + [3e-4]),
    initial_log_alpha=hp.choice("initial_log_alpha", [-50, -20, -10, -5, -3.5]),
    policy_freq=hp.choice("policy_freq", [1, 2]),
    seed=hp.randint("seed", 20),
    start_timesteps=hp.choice("start_timesteps", big_values(0, 1)),
    batches=hp.choice("batches", []),
    eval_freq=5000,
    max_timesteps=None,
    replay_size=200000,
    discount=0.99,
    choices=50,
    context_length=hp.choice("context_length", [10, 20, 30, 40]),
    report_freq=50,
)
configs = dict(search=search, pendulum=pendulum,)
