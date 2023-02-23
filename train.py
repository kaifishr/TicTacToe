"""Trains agent to play Tic-Tac-Toe.

Uses the defined optimizer procedure to train
the neural network of an agent to play Tic-Tac-Toe.

"""
import copy
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from src.tictactoe import Environment
from src.tictactoe import TicTacToe
from src.model import Model


class PolicyGradients:
    """Policy gradient agent."""

    def __init__(self, model: nn.Module, learning_rate: float, gamma: float) -> None:
        """Initializes class."""
        self.size = 3
        self.model = model  # policy
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def step(self, model: nn.Module, events: dict) -> float:
        """Runs single optimization step. Updates the network."""

        states = events["states"]
        actions = events["actions"]
        rewards = events["rewards"]

        reward_sum = 0.0
        rewards_to_go = []

        for reward in rewards[::-1]:
            reward_sum = reward + self.gamma * reward_sum
            rewards_to_go.append(reward_sum)

        rewards_to_go.reverse()
        # rewards_to_go = rewards_to_go[::-1]

        rewards_to_go = torch.tensor(rewards_to_go)

        # if len(rewards_to_go) > 1:
        #     std = torch.std(rewards_to_go)
        #     if std != 0:
        #         mean = torch.mean(rewards_to_go)
        #         rewards_to_go = (rewards_to_go - mean) / (std + 1e-5)  # the sample weight

        states = torch.vstack(states)
        target_actions = F.one_hot(torch.tensor(actions), num_classes=self.size**2).float()

        # https://discuss.pytorch.org/t/per-class-and-per-sample-weighting/25530/3
        self.optimizer.zero_grad()
        output_actions = self.model(states)
        loss = self.criterion(output_actions, target_actions)
        loss = loss * rewards_to_go
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()

        return loss.item(), sum(rewards)


# class SimulatedAnnealing:
#     """Optimizer class for simulated annealing with single agent."""
#
#     def __init__(self, num_episodes: int) -> None:
#         """Initializes optimizer"""
#         self.model = None
#         self.model_old = None
#
#         self.perturbation_probability_initial = 0.1     # initial->start
#         self.perturbation_probability_final = 0.001     # final->end
#         self.perturbation_rate_initial = 0.1
#         self.perturbation_rate_final = 0.01
#         self.temp_initial = 10.0
#         self.temp_final = 1e-6
#         self.temp = self.temp_initial
#         num_iterations = num_episodes
#
#         self.gamma = (1.0 / num_iterations) * math.log(self.temp_initial / self.temp_final)
#         self.reward_old = float("-inf")
#
#         # Parameters to be visualized with Tensorboard.
#         self.stats = {
#             "reward": None,
#             "loss": None,
#             "temperature": None,
#         }
#
#         # Scalars
#         self.iteration = 0
#         self.idx_best = 0
#
#     def _scheduler(self) -> None:
#         """Decreases temperature according to exponential decay."""
#         self.temp = self.temp_initial * math.exp(-self.gamma * self.iteration)
#         if self.temp < self.temp_final:
#             self.temp = self.temp_final
#
#     @torch.no_grad()
#     def _perturb(self, module: nn.Module) -> None:
#         """Mutates weights of model.
#
#         Args:
#             module: Pytorch module object.
#
#         """
#         eta = self.temp / self.temp_initial
#
#         pert_prob_init = self.perturbation_probability_initial
#         pert_prob_final = self.perturbation_probability_final
#         perturbation_prob = (pert_prob_init - pert_prob_final) * eta + pert_prob_final
#
#         pert_rate_init = self.perturbation_rate_initial
#         pert_rate_final = self.perturbation_rate_final
#         perturbation_rate = (pert_rate_init - pert_rate_final) * eta + pert_rate_final
#
#         if isinstance(module, nn.Linear):
#             mask = torch.rand_like(module.weight) < perturbation_prob
#             mutation = perturbation_rate * torch.randn_like(module.weight)
#             module.weight.add_(mask * mutation)
#
#             if module.bias is not None:
#                 mask = torch.rand_like(module.bias) < perturbation_prob
#                 mutation = perturbation_rate * torch.randn_like(module.bias)
#                 module.bias.add_(mask * mutation)
#
#     def step(self, model: nn.Module, events: dict) -> None:
#         """Runs single optimization step."""
#
#         if self.model_old is None:
#             self.model_old = copy.deepcopy(model)
#
#         # Extra rewards for every game move.
#         reward = sum(events["rewards"])
#
#         self.stats["temperature"] = self.temp
#
#         delta_reward = reward - self.reward_old
#
#         # Accept configuration if reward is higher or with probability p = exp(delta_reward / temp)
#         if (delta_reward > 0) or (math.exp(delta_reward / self.temp) > random.random()):
#             # Save network if current reward is higher.
#             self.model_old = copy.deepcopy(self.model)
#             self.reward_old = reward
#         else:
#             # Do not accept current state. Return to previous state.
#             model = copy.deepcopy(self.model_old)
#
#         # Reduce temperature according to scheduler
#         self._scheduler()
#
#         # Perturb weights for next iteration.
#         model.apply(self._perturb)
#
#         self.iteration += 1
#
#         return reward


# def train_simulated_annealing(env: Environment, model_a: nn.Module, model_b: nn.Module) -> None:
#     """Train agents with simulated annealing."""
#
#     # Trainer
#     num_iterations = 200000
#     num_episodes = 20
#
#     optimizer = SimulatedAnnealing(num_episodes=num_iterations)
#     writer = SummaryWriter()
#
#     for iteration in range(num_iterations):
#
#         events_a = dict(states=[], actions=[], rewards=[], new_states=[], dones=[])
#         events_b = dict(states=[], actions=[], rewards=[], new_states=[], dones=[])
#
#         for episode in range(num_episodes):
#
#             # Let the agents compete.
#             if random.random() > 0.5:
#             # if episode % 2 == 0:
#                 tmp_events_a, tmp_events_b = env.episode(model_a, model_b)
#             else:
#                 tmp_events_b, tmp_events_a = env.episode(model_b, model_a)
#
#             for key, value in tmp_events_a.items():
#                 events_a[key].extend(value)
#
#             for key, value in tmp_events_b.items():
#                 events_b[key].extend(value)
#
#         reward_a = optimizer.step(model_a, events_a)
#         reward_b = optimizer.step(model_b, events_b)
#
#         if iteration % 500 == 0:
#             writer.add_scalar("reward/a", reward_a, iteration)
#             writer.add_scalar("reward/b", reward_b, iteration)
#
#     env.play(model=model_a)
#     env.play(model=model_b)
#
#     writer.close()


def train_policy_gradients(env: Environment, model_a: nn.Module, model_b: nn.Module) -> None:
    """Train agents with Policy Gradients."""

    # Trainer
    # Good parameters!
    num_episodes = 500000
    learning_rate = 0.0005
    gamma = 1.0

    agent_a = PolicyGradients(model=model_a, learning_rate=learning_rate, gamma=gamma)
    agent_b = PolicyGradients(model=model_b, learning_rate=learning_rate, gamma=gamma)

    writer = SummaryWriter()

    for episode in range(num_episodes):

        events_a = dict(states=[], actions=[], rewards=[], new_states=[], dones=[])
        events_b = dict(states=[], actions=[], rewards=[], new_states=[], dones=[])

        # Let the agents compete. Rollout one episode.
        if random.random() > 0.5:
            # if episode % 2 == 0:
            events_a, events_b = env.episode(model_a, model_b)
        else:
            events_b, events_a = env.episode(model_b, model_a)

        # Update network.
        loss_a, reward_a = agent_a.step(model_a, events_a)
        loss_b, reward_b = agent_b.step(model_b, events_b)

        if episode % 500 == 0:
            writer.add_scalar("loss/a", loss_a, episode)
            writer.add_scalar("loss/b", loss_b, episode)
            writer.add_scalar("reward/a", reward_a, episode)
            writer.add_scalar("reward/b", reward_b, episode)

    writer.close()

    print("Model a")
    env.play(model=model_a)
    print("Model b")
    env.play(model=model_b)


# def train_policy_gradients_2(env: Environment, model_a: nn.Module, model_b: nn.Module) -> None:
#     """Train agents with simulated annealing."""
#
#     # Trainer
#     num_iterations = 20000
#     num_episodes = 20
#     learning_rate = 0.0005
#     gamma = 1.0
#
#     agent_a = PolicyGradients(model=model_a, learning_rate=learning_rate, gamma=gamma)
#     agent_b = PolicyGradients(model=model_b, learning_rate=learning_rate, gamma=gamma)
#
#     writer = SummaryWriter()
#
#     for iteration in range(num_iterations):
#
#         events_a = dict(states=[], actions=[], rewards=[], new_states=[], dones=[])
#         events_b = dict(states=[], actions=[], rewards=[], new_states=[], dones=[])
#
#         for episode in range(num_episodes):
#
#             # Let the agents compete.
#             if random.random() > 0.5:
#             # if episode % 2 == 0:
#                 tmp_events_a, tmp_events_b = env.episode(model_a, model_b)
#             else:
#                 tmp_events_b, tmp_events_a = env.episode(model_b, model_a)
#
#             for key, value in tmp_events_a.items():
#                 events_a[key].extend(value)
#
#             for key, value in tmp_events_b.items():
#                 events_b[key].extend(value)
#
#         # Update network
#         loss_a, reward_a = agent_a.step(model_a, events_a)
#         loss_b, reward_b = agent_b.step(model_b, events_b)
#
#         if iteration % 50 == 0:
#             writer.add_scalar("loss/a", loss_a, iteration)
#             writer.add_scalar("loss/b", loss_b, iteration)
#             writer.add_scalar("reward/a", reward_a, iteration)
#             writer.add_scalar("reward/b", reward_b, iteration)
#
#     print("Model a")
#     env.play(model=model_a)
#     print("Model b")
#     env.play(model=model_b)
#
#     writer.close()

if __name__ == "__main__":

    # Playfield
    size = 3

    env = TicTacToe(size=size)

    model_a = Model(size=size)
    model_b = Model(size=size)

    train_policy_gradients(env=env, model_a=model_a, model_b=model_b)
    # train_policy_gradients_2(env=env, model_a=model_a, model_b=model_b)
    # train_simulated_annealing(env=env, model_a=model_a, model_b=model_b)
