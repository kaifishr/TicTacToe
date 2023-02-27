"""Play against an agent."""
import json

from src.utils import load_checkpoint
from src.argparser import argument_parser
from src.model import Model
from src.environment import TicTacToe


if __name__ == "__main__":

    args = argument_parser()
    print(args)
    for key, value in vars(args).items():
        print(key, value)
    print(json.dumps(vars(args), indent=4))

    model = Model(args=args)
    load_checkpoint(model=model, args=args)

    env = TicTacToe(size=args.field_size)
    env.play(model=model)
