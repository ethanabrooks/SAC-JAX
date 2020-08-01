from hyperopt import hp


def small_values(start, stop):
    return [j for i in range(start, stop) for j in ((10 ** -i), 5 * (10 ** -i))]


def medium_values(start, stop):
    return [2 ** i for i in range(start, stop)]


def big_values(start, stop):
    return [j for i in range(start, stop) for j in ((10 ** i), 5 * (10 ** i))]


search = dict(
    batch_size=hp.choice("batch_size", medium_values(6, 10)),
    expl_noise=hp.choice("expl_noise", small_values(1, 3)),
    noise_clip=hp.choice("noise_clip", small_values(1, 3)),
    lr=hp.choice("lr", small_values(2, 5) + [3e-4]),
    policy_freq=hp.choice("policy_freq", [1, 2, 3]),
    seed=hp.randint("seed", 20),
    start_timesteps=hp.choice("start_timesteps", big_values(0, 2)),
)

pendulum = {
    "batch_size": 128,
    "discount": 0.99,
    "env_id": "Pendulum-v0",
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

double_search = dict(
    max_timesteps=hp.choice("max_timesteps", big_values(3, 5)),
    **search,
    **{"outer_" + k: v for k, v in search.items()}
)
double_search.update(start_timesteps=0, outer_start_timesteps=1)

# TODO
double = dict(
    **{"outer_" + k: v for k, v in pendulum.items()},
    **{"inner_" + k: v for k, v in pendulum.items()}
)
