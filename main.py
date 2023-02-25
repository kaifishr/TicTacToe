"""Main file to start training or playing."""
from src.tictactoe import TicTacToe

# from src.play import PlayGame
# from src.train import Trainer


if __name__ == "__main__":

    size = 3
    env = TicTacToe(size=size)
    env.play()
