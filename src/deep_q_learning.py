"""Deep Q-learning class."""
import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepQLearner:
    """Deep Q-Learner."""

    def __init__(self, model: nn.Module, learning_rate: float, gamma: float) -> None:
        """Initializes class."""
        self.size = 3
        self.model = model  # policy
        self.learning_rate = learning_rate

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.gamma = 0.99  # decay rate
        self.memory_size = 10000

        self.memory = deque(maxlen=self.memory_size)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def select_action(self, state: torch.Tensor) -> int:
        """Selects an action from a discrete action space.

        Action is random with probability `epsilon` (epsilon-greedy value)
        to encourage exploration. 

        Args:
            state: State observed by agent.

        Returns:
            Action according to current policy or random action.
        """
        if random.random() < self.epsilon:
            # Exploration by choosing random action.
            action = random.randint(0, self.size**2 - 1)
        else:
            # Exploitation by selecting action according to policy
            # with highest predicted utility at current state.
            action = self.model.get_action(state)

        return action

    def step(self, events: dict) -> float:
        """Runs single optimization step. Updates the network."""

        states = events["states"]
        actions = events["actions"]
        rewards = events["rewards"]
        dones = events["dones"]

        reward_sum = 0.0
        discounted_rewards = []

        # TODO: Get rid of for-loop
        for reward in rewards[::-1]:
            reward_sum = reward + self.gamma * reward_sum
            discounted_rewards.append(reward_sum)

        discounted_rewards.reverse()
        # discounted_rewards = discounted_rewards[::-1]

        discounted_rewards = torch.tensor(discounted_rewards)

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

        return loss.item(), sum(rewards)
