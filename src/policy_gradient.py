"""Policy gradients class.

Uses policy gradients to learn task.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agent import Agent


class PolicyGradient(Agent):
    """Policy gradient agent.
    
    Attributes:
        size:
        learning_rate:
        gamma:
        optimizer:
        criterion:
    
    """

    def __init__(self, model: nn.Module, args) -> None:
        """Initializes class."""
        super().__init__(model=model)

        self.size = args.field_size
        self.learning_rate = args.learning_rate
        self.gamma = args.gamma

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="none")

    @torch.no_grad()
    def get_action(self, state: torch.Tensor) -> int:
        """Samples action given a state.

        Args:
            state: Flattended playfield of size `size**2`.

        Returns:
            Sampled action represented by an integer.
        """
        self.model.eval() # TODO: Use decorator for eval / train
        # Build the probability density function (PDF) for the given state.
        action_prob = self.model(state)
        # Sample action according to (PDF)
        # TODO: Add temperature scaling
        action = torch.multinomial(action_prob, num_samples=1).item()
        self.model.train()
        return action

    def step(self, events: dict) -> None:
        """Runs single optimization step. Updates the network."""

        states = events["states"]
        actions = events["actions"]
        rewards = events["rewards"]

        reward_sum = 0.0
        discounted_rewards = []

        # TODO: Get rid of for-loop
        for reward in rewards[::-1]:
            reward_sum = reward + self.gamma * reward_sum
            discounted_rewards.append(reward_sum)

        discounted_rewards.reverse()
        # discounted_rewards = discounted_rewards[::-1]

        discounted_rewards = torch.tensor(discounted_rewards)

        # TODO: Move this to base class
        # if len(discounted_rewards) > 1:
        #     std = torch.std(discounted_rewards)
        #     if std != 0:
        #         mean = torch.mean(discounted_rewards)
        #         discounted_rewards = (discounted_rewards - mean) / (std + 1e-5)  # the sample weight

        states = torch.vstack(states)
        target_actions = F.one_hot(torch.tensor(actions), num_classes=self.size**2).float()

        # https://discuss.pytorch.org/t/per-class-and-per-sample-weighting/25530/3
        self.optimizer.zero_grad()
        output_actions = self.model(states)
        loss = self.criterion(output_actions, target_actions)
        loss = loss * discounted_rewards
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()

        self.stats["loss"] = loss
        self.stats["reward"] = sum(rewards)
