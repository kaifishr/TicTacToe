"""Argument parser.

Holds environment and learning parameters.

"""
import argparse

def argument_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        prog="TicTacToe",
        description="Trains agent to play Tic-tac-toe with reinforcement learning."
    )

    parser.add_argument(
        "-t", "--tag",
        dest="tag",
        default="",
        help="Optional name tag for run.",
        type=str
    )

    parser.add_argument(
        "-rs", "--random-seed",
        dest="random_seed",
        default=42,
        type=int
    )

    #########
    # Agent #
    #########

    parser.add_argument(
        "-a", "--algorithm",
        dest="algorithm",
        help="Reinforcement learning algorithm.",
        default="policy_gradient",
        choices=["policy_gradient", "deep_q_learning"],
        type=str,
    )

    ###########
    # Trainer #
    ###########

    parser.add_argument(
        "-lr", "--learning-rate",
        dest="learning_rate",
        default=1e-4,
        type=float
    )

    parser.add_argument(
        "-ne", "--num-episodes",
        dest="num_episodes",
        default=100000,
        type=int
    )

    parser.add_argument(
        "-g", "--gamma",
        dest="gamma",
        help="Discount or forgetting factor. 0 <= gamma <= 1.",
        default=1.0,
        type=float
    )

    parser.add_argument(
        "-e", "--epsilon",
        dest="epsilon",
        help="Epsilon-greedy value (exploration rate).",
        default=1.0,
        type=float
    )

    parser.add_argument(
        "-em", "--epsilon-min",
        dest="epsilon_min",
        help="Minimum epsilon-greedy value.",
        default=0.01,
        type=float
    )

    parser.add_argument(
        "-dr", "--decay-rate",
        dest="decay_rate",
        help="Decay rate for epsilon-greedy value.",
        default=0.999,
        type=float
    )
    
    parser.add_argument(
        "-ms", "--memory-size",
        dest="memory_size",
        help="Replay memory size. Set to 1 for no memory.",
        default=1000000,
        type=int
    )

    parser.add_argument(
        "-bs", "--batch_size",
        dest="batch_size",
        default=128,
        type=int
    )

    ###############
    # Environment #
    ###############

    parser.add_argument(
        "-fs", "--field-size",
        dest="field_size",
        default=3,
        type=int
    )

    #############################
    # Model / policy parameters #
    #############################

    parser.add_argument(
        "-dp", "--dropout-probability",
        dest="dropout_probability",
        default=0.0,
        type=float
    )

    parser.add_argument(
        "-l", "--layers",
        dest="num_layers",
        default=2,
        type=int
    )

    parser.add_argument(
        "-hu", "--hidden-units",
        dest="num_hidden_units",
        default=32,
        type=int
    )

    parser.add_argument(
        "-mn", "--model-name",
        dest="model_name",
        help="Defines which model to load.",
        default="agent_a",
        type=str
    )

    return parser.parse_args()