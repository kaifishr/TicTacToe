"""Play against an agent."""
import torch
import random

from src.tictactoe import TicTacToe
from src.model import Model


if __name__ == "__main__":

    size = 3
    num_blocks = 4

    model = Model(size=size, num_blocks=num_blocks)
    # TODO: load trained model.

    env = TicTacToe(size=size)
    env.play(model=model)
