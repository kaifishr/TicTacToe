"""Trains agent to play Tic-Tac-Toe.

Uses the defined optimizer procedure to train
the neural network of an agent to play Tic-Tac-Toe.

"""
import random

from torch.utils.tensorboard import SummaryWriter

from src.utils import print_args
from src.utils import save_checkpoint
from src.utils import set_random_seed
from src.argparser import argument_parser
from src.environment import Environment
from src.environment import TicTacToe
from src.model import Model
from src.agent import Agent
from src.policy_gradient import PolicyGradient
from src.deep_q_learning import DeepQLearning


def run_agents(env: Environment, agent_a: Agent, agent_b: Agent, args) -> None:
    """Train agents with Policy Gradients."""

    writer = SummaryWriter()

    for episode in range(args.num_episodes):

        # Run episode and let the agents compete.
        if random.random() < 0.5:
            events_a, events_b = env.run_episode(agent_a, agent_b)
        else:
            events_b, events_a = env.run_episode(agent_b, agent_a)

        # Update network.
        agent_a.step(events_a)
        agent_b.step(events_b)

        if episode % 500 == 0:

            for key, value in agent_a.stats.items():
                if value:
                    writer.add_scalar(f"agent_a/{key}", value, episode)

            for key, value in agent_b.stats.items():
                if value:
                    writer.add_scalar(f"agent_b/{key}", value, episode)

    writer.close()

    save_checkpoint(model=agent_a.model, model_name="agent_a", args=args)
    save_checkpoint(model=agent_b.model, model_name="agent_b", args=args)


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
        run_agents(env=env, agent_a=agent_a, agent_b=agent_b, args=args)

    elif args.algorithm == "deep_q_learning":
        set_random_seed(seed=args.random_seed)
        model_a = Model(args)
        model_b = Model(args)
        agent_a = DeepQLearning(model=model_a, args=args)
        agent_b = DeepQLearning(model=model_b, args=args)
        run_agents(env=env, agent_a=agent_a, agent_b=agent_b, args=args)

    else:
        raise NotImplementedError(
            f"Reinforcement algorithm {args.algorithm} not implemented"
        )
