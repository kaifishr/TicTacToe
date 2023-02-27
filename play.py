"""Play against an agent."""
from src.utils import print_args
from src.utils import load_checkpoint
from src.argparser import argument_parser
from src.model import Model
from src.environment import TicTacToe


if __name__ == "__main__":

    args = argument_parser()

    print_args(args=args)

    model = Model(args=args)
    load_checkpoint(model=model, args=args)

    env = TicTacToe(size=args.field_size)
    env.play(model=model)
