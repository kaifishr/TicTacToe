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
        "-rs", "--random-seed",
        dest="random_seed",
        default=42,
        type=int
    )

    # Agent

    parser.add_argument(
        "-m", "--method",
        dest="method",
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
        "-n", "--num-episodes",
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

    # Environment

    parser.add_argument(
        "-fs", "--field-size",
        dest="field_size",
        default=3,
        type=int
    )

    # Model parameters

    parser.add_argument(
        "-p", "--dropout-probability",
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
        "-u", "--hidden-units",
        dest="num_hidden_units",
        default=32,
        type=int
    )

    return parser.parse_args()