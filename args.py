def add_arguments(parser):
    parser.add_argument(
        "--policy", default="TD3", choices=["TD3", "DDPG"]
    )  # Policy name (TD3, DDPG)
    parser.add_argument(
        "--env", dest="env_id", default="Pendulum-v0"
    )  # OpenAI gym environment name
    parser.add_argument(
        "--seed", type=int, required=True
    )  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument(
        "--start_timesteps", default=25000, type=int
    )  # Time steps initial random policy is used
    parser.add_argument(
        "--eval_freq", default=5e3, type=int
    )  # How often (time steps) we evaluate
    parser.add_argument(
        "--max_timesteps", default=1e6, type=int
    )  # Max time steps to run environment
    parser.add_argument(
        "--replay_size", default=200000, type=int
    )  # Size of the replay buffer
    parser.add_argument(
        "--expl_noise", default=0.1, type=float
    )  # Std of Gaussian exploration noise
    parser.add_argument(
        "--batch_size", default=256, type=int
    )  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument(
        "--policy_noise", default=0.2, type=float
    )  # Noise added to target policy during critic update
    parser.add_argument(
        "--noise_clip", default=0.5, type=float
    )  # Range to clip target policy noise
    parser.add_argument("--lr", default=3e-4, type=float)  # Optimizer learning rates
    parser.add_argument(
        "--policy_freq", default=2, type=int
    )  # Frequency of delayed policy updates