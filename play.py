"""Play against an agent."""
from src.tictactoe import TicTacToe
from src.model import Model
from src.utils import load_checkpoint


if __name__ == "__main__":

    # Playing field size
    size = 3

    model = Model(size=size)
    load_checkpoint(model=model, model_name="agent_a")

    env = TicTacToe(size=size)
    env.play(model=model)
