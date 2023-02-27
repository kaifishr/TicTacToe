"""Trains agent to play Tic-Tac-Toe.

Uses the defined optimizer procedure to train
the neural network of an agent to play Tic-Tac-Toe.

"""
import random

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.argparser import argument_parser
from src.tictactoe import Environment
from src.tictactoe import TicTacToe
from src.model import Model  # PGNetwork
from src.policy_gradient import PolicyGradient
from src.deep_q_learning import DeepQLearning
from src.agent import Agent
from src.utils import save_checkpoint
from src.utils import set_random_seed


def run_agents(env: Environment, agent_a: Agent, model_b: Agent, args) -> None:
    """Train agents with Policy Gradients."""

    writer = SummaryWriter()

    for episode in range(args.num_episodes):

        # Run episode and let the agents compete.
        if random.random() < 0.5:
            events_a, events_b = env.run_episode(agent_a, agent_b)
        else:
            events_b, events_a = env.run_episode(agent_b, agent_a)

        # Update network.
        loss_a, reward_a = agent_a.step(events_a)
        loss_b, reward_b = agent_b.step(events_b)

        if episode % 500 == 0:
            writer.add_scalar("loss/a", loss_a, episode)
            writer.add_scalar("loss/b", loss_b, episode)
            writer.add_scalar("reward/a", reward_a, episode)
            writer.add_scalar("reward/b", reward_b, episode)

    writer.close()

    save_checkpoint(model=agent_a.model, model_name="agent_a")
    save_checkpoint(model=agent_b.model, model_name="agent_b")


def run_deep_q_agents(env: Environment, model_a: nn.Module, model_b: nn.Module, args) -> None:
    """Train agents with Deep Q-Learning."""

    # Trainer
    num_episodes = args.num_episodes

    agent_a = DeepQLearning(model=model_a, args=args)
    agent_b = DeepQLearning(model=model_b, args=args)

    writer = SummaryWriter()

    for episode in range(num_episodes):

        # Run episode and let the agents compete.
        if random.random() < 0.5:
            events_a, events_b = env.run_episode(agent_a, agent_b)
        else:
            events_b, events_a = env.run_episode(agent_b, agent_a)

        # Update network.
        loss_a, reward_a = agent_a.step(events_a)
        loss_b, reward_b = agent_b.step(events_b)

        if episode % 500 == 0:
            writer.add_scalar("loss/a", loss_a, episode)
            writer.add_scalar("loss/b", loss_b, episode)
            writer.add_scalar("reward/a", reward_a, episode)
            writer.add_scalar("reward/b", reward_b, episode)

    writer.close()

    save_checkpoint(model=agent_a.model, model_name="agent_a")
    save_checkpoint(model=agent_b.model, model_name="agent_b")


if __name__ == "__main__":

    args = argument_parser()
    
    set_random_seed(seed=args.random_seed)

    # Playing field size
    size = args.field_size
    env = TicTacToe(size=args.field_size)

    args.method = "policy_gradient"
    model_a = Model(args)
    model_b = Model(args)
    agent_a = PolicyGradient(model=model_a, args=args)
    agent_b = PolicyGradient(model=model_b, args=args)
    run_agents(env=env, agent_a=agent_a, agent_b=agent_b, args=args)

    args.method = "deep_q_learning"
    model_a = Model(args)
    model_b = Model(args)
    agent_a = DeepQLearning(model=model_a, args=args)
    agent_b = DeepQLearning(model=model_b, args=args)
    run_agents(env=env, agent_a=agent_a, agent_b=agent_b, args=args)
