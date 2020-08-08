import json
from pathlib import Path

from hyperopt import hp
from ray import tune


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
    batch_size=hp.choice("batch_size", medium_values(5, 10)),
    lr=hp.choice("lr", small_values(2, 5) + [3e-4]),
    initial_log_alpha=hp.choice("initial_log_alpha", [-50, -20, -10, -5, -3.5]),
    policy_freq=hp.choice("policy_freq", [1, 2]),
    seed=hp.randint("seed", 20),
    start_timesteps=hp.choice("start_timesteps", big_values(0, 1)),
    eval_freq=5000,
    max_timesteps=None,
    replay_size=200000,
    discount=0.99,
)
pendulum_seeds = get_config("pendulum")
pendulum_seeds.update(seed=tune.grid_search(list(range(8))))

configs = dict(search=search, pendulum_seeds=pendulum_seeds)
