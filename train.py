"""Trains agent to play Tic-Tac-Toe.

Uses the defined optimizer procedure to train
the neural network of an agent to play Tic-Tac-Toe.

"""
from src.utils import print_args
from src.utils import set_random_seed
from src.argparser import argument_parser
from src.environment import TicTacToe
from src.model import Model
from src.policy_gradient import PolicyGradient
from src.deep_q_learning import DeepQLearning
from src.trainer import train


if __name__ == "__main__":

    args = argument_parser()
    print_args(args=args)

    env = TicTacToe(size=args.field_size)

    if args.algorithm == "policy_gradient":
        set_random_seed(seed=args.random_seed)
        model_a = Model(args)
        model_b = Model(args)
        agent_a = PolicyGradient(model=model_a, args=args)
        agent_b = PolicyGradient(model=model_b, args=args)
        train(env=env, agent_a=agent_a, agent_b=agent_b, args=args)

    elif args.algorithm == "deep_q_learning":
        set_random_seed(seed=args.random_seed)
        model_a = Model(args)
        model_b = Model(args)
        agent_a = DeepQLearning(model=model_a, args=args)
        agent_b = DeepQLearning(model=model_b, args=args)
        train(env=env, agent_a=agent_a, agent_b=agent_b, args=args)

    else:
        raise NotImplementedError(f"Reinforcement algorithm {args.algorithm} not implemented")
