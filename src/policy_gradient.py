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
        """Samples an action from a discrete action space given a state.

        We use the current policy-model to map the environment observation,
        the state, to a probability distribution of the actions, and sample
        from this distribution.

        Args:
            state: Tensor representing playing field state.

        Returns:
            Sampled action represented by an integer.
        """
        self.model.eval()
        # Build the probability density function (PDF) for the given state.
        action_prob = self.model(state)
        # Sample action from the distribution (PDF).
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

        for reward in rewards[::-1]:
            reward_sum = reward + self.gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards = discounted_rewards[::-1]

        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = self._normalize_rewards(rewards=discounted_rewards)

        states = torch.vstack(states)
        target_actions = F.one_hot(torch.tensor(actions), num_classes=self.size**2).float()

        # https://discuss.pytorch.org/t/per-class-and-per-sample-weighting/25530/3
        self.optimizer.zero_grad()
        output_actions = self.model(states)
        loss = self.criterion(output_actions, target_actions)
        loss = discounted_rewards * loss
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()

        self.stats["loss"] = loss
        self.stats["reward"] = sum(rewards)
