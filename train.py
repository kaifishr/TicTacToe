"""Trains agent to play Tic-Tac-Toe.

Uses the defined optimizer procedure to train
the neural network of an agent to play Tic-Tac-Toe.

"""
import random

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.tictactoe import Environment
from src.tictactoe import TicTacToe
from src.model import Model  # PGNetwork
from src.model import QNetwork
from src.policy_gradients import PolicyGradients
from src.deep_q_learning import DeepQLearner
from src.utils import save_checkpoint
from src.utils import set_random_seed


def train_policy_gradients(env: Environment, model_a: nn.Module, model_b: nn.Module) -> None:
    """Train agents with Policy Gradients."""

    # Trainer
    num_episodes = 200000
    learning_rate = 0.0005
    gamma = 1.0

    agent_a = PolicyGradients(model=model_a, learning_rate=learning_rate, gamma=gamma)
    agent_b = PolicyGradients(model=model_b, learning_rate=learning_rate, gamma=gamma)

    writer = SummaryWriter()

    for episode in range(num_episodes):

        # Let the agents compete. Rollout one episode.
        if random.random() > 0.5:
            events_a, events_b = env.episode(model_a, model_b)
        else:
            events_b, events_a = env.episode(model_b, model_a)

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


def train_deep_q(env: Environment, model_a: nn.Module, model_b: nn.Module) -> None:
    """Train agents with Deep Q-Learning."""

    # Trainer
    num_episodes = 100000
    learning_rate = 0.0005
    gamma = 0.99

    agent_a = DeepQLearner(model=model_a, learning_rate=learning_rate, gamma=gamma)
    agent_b = DeepQLearner(model=model_b, learning_rate=learning_rate, gamma=gamma)

    writer = SummaryWriter()

    for episode in range(num_episodes):

        # Let the agents compete. Rollout one episode.
        if random.random() > 0.5:
            events_a, events_b = env.episode(model_a, model_b)  # TODO: do env.episode(agent_a, agent_b)
        else:
            events_b, events_a = env.episode(model_b, model_a)

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

    set_random_seed(seed=42)

    # Playing field size
    size = 3

    env = TicTacToe(size=size)

    # model_a = Model(size=size)
    # model_b = Model(size=size)
    # train_policy_gradients(env=env, model_a=model_a, model_b=model_b)

    model_a = QNetwork(size=size)
    model_b = QNetwork(size=size)
    train_deep_q(env=env, model_a=model_a, model_b=model_b)
