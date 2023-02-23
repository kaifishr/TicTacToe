import torch
import random

from src.tictactoe import TicTacToe


if __name__ == "__main__":

    size = 3
    env = TicTacToe(size=size)
    env.play()
