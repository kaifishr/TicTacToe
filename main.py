"""Main file to start training or playing."""
from environment import TicTacToe

# from src.play import PlayGame
# from src.train import Trainer


if __name__ == "__main__":

    size = 3
    env = TicTacToe(size=size)
    env.play()
