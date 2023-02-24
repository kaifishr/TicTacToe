"""Policy gradients class.

Uses policy gradients to learn task.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def step(self, events: dict) -> float:
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