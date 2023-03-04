"""Holds the training method."""
import random

from torch.utils.tensorboard import SummaryWriter

from src.utils import save_checkpoint
from src.environment import Environment
from src.agent import Agent


def train(env: Environment, agent_a: Agent, agent_b: Agent, args) -> None:
    """Trains agents with selected reinforcement algorithm."""

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
