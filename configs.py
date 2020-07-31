from hyperopt import hp


def small_values(start, stop):
    return [j for i in range(start, stop) for j in ((10 ** -i), 5 * (10 ** -i))]


def medium_values(start, stop):
    return [2 ** i for i in range(start, stop)]


def big_values(start, stop):
    return [j for i in range(start, stop) for j in ((10 ** i), 5 * (10 ** i))]


search = dict(
    batch_size=hp.choice("batch_size", medium_values(6, 10)),
    expl_noise=hp.choice("lr", small_values(1, 3)),
    noise_clip=hp.choice("lr", small_values(1, 3)),
    lr=hp.choice("lr", small_values(2, 5) + [3e-4]),
    policy_freq=hp.choice("policy_freq", [1, 2, 3]),
    seed=hp.randint("seed", 20),
    start_timesteps=hp.choice("start_timesteps", big_values(0, 2) + [0]),
)

pendulum = dict()

double_search = dict(
    max_timesteps=hp.choice("max_timesteps", big_values(3, 5)),
    **search,
    **{"outer_" + k: v for k, v in search.items()}
)
double_search.update(start_timesteps=0, outer_start_timesteps=1)
