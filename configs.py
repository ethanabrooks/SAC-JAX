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


def small_values(start, stop):
    return [j for i in range(start, stop) for j in ((10 ** -i), 5 * (10 ** -i))]


def medium_values(start, stop):
    return [2 ** i for i in range(start, stop)]


def big_values(start, stop):
    return [j for i in range(start, stop) for j in ((10 ** i), 5 * (10 ** i))]


search = dict(
    batch_size=hp.choice("batch_size", medium_values(6, 10)),
    lr=hp.choice("lr", small_values(2, 5) + [3e-4]),
    initial_log_alpha=hp.choice("initial_log_alpha", [-10, -5, -3.5, -1]),
    policy_freq=hp.choice("policy_freq", [1, 2, 3]),
    seed=hp.randint("seed", 20),
    start_timesteps=hp.choice("start_timesteps", big_values(0, 2)),
)
configs = dict(search=search)
