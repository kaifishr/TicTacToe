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
        self.batch_size = 128

        # self.epsilon = 0.9
        # self.epsilon_min = 0.01
        # self.decay_rate = 0.999
        self.gamma = 1.0
        self.memory_size = 10000  # no replay with memory_size = 1

        self.memory = deque(maxlen=self.memory_size)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss(reduction="mean")

    # def select_action(self, state: torch.Tensor) -> int:
    #     """Selects an action from a discrete action space.

    #     Action is random with probability `epsilon` (epsilon-greedy value)
    #     to encourage exploration.

    #     Args:
    #         state: State observed by agent.

    #     Returns:
    #         Action according to current policy or random action.
    #     """
    #     if random.random() < self.epsilon:
    #         # Exploration by choosing random action.
    #         action = random.randint(0, self.size**2 - 1)  # m * n - 1
    #     else:
    #         # Exploitation by selecting action according to policy
    #         # with highest predicted utility at current state.
    #         action = self.model.get_action(state)

    #     return action

    def _memorize(self, events: dict) -> None:
        """Writes current events to memory (replay buffer).

        Args:
            events: Dictionary holding (states, actions, rewards, new_states, dones) tuple.

        """
        for state, action, rewards, new_state, done in zip(*events.values()):
            self.memory.append([state, action, rewards, new_state, done])

    # def _epsilon_scheduler(self) -> None:
    #     """Decays epsilon-greedy value."""
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.decay_rate

    def _create_training_set(self) -> None:
        """Create training set from memory."""

        # Use subset of replay memory for training as transitions are strongly correlated.
        replay_batch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        # TODO: Normalize the rewards.

        # Get states from replay buffer.
        states = torch.vstack([memory[0] for memory in replay_batch])
        new_states = torch.vstack([memory[3] for memory in replay_batch])

        with torch.no_grad():
            self.model.eval()
            q_targets = self.model(states)
            q_targets_new = self.model(new_states)
            self.model.train()

        for i, (_, action, reward, _, done) in enumerate(replay_batch):
            if not done:
                q_targets[i, action] = reward + self.gamma * torch.amax(q_targets_new[i]).item()
            else:
                q_targets[i, action] = reward

        return states, q_targets

    @staticmethod
    def print_events(events: dict) -> None:
        """Prints events in a better format."""
        for key, value in events.items():
            print(f"{key} = \n")
            for item in value:
                print(f"{item}\n")

    def step(self, events: dict) -> float:
        """Runs single optimization step. Updates the network."""

        rewards = events["rewards"]

        self._memorize(events=events)

        states, q_targets = self._create_training_set()

        # Update parameters.
        self.optimizer.zero_grad()
        output_actions = self.model(states)
        loss = self.criterion(output_actions, q_targets)
        loss.backward()
        self.optimizer.step()

        # self._epsilon_scheduler()

        return loss.item(), sum(rewards)
