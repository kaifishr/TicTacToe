"""Deep Q-learning class."""
import random
from collections import deque

import torch
import torch.nn as nn

from src.agent import Agent


class DeepQLearning(Agent):
    """Deep Q-Learner.

    More class information.

    Attributes:
        size:
        learning_rate:
        batch_size:
        epsilon:
        epsilon_min:
        decay_rate:
        gamma:
        memory_size:
        memory:
        optimizer:
        criterion:
    """

    def __init__(self, model: nn.Module, args) -> None:
        """Initializes class."""
        super().__init__(model=model)

        self.size = args.field_size
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.epsilon = args.epsilon
        self.epsilon_min = args.epsilon_min
        self.decay_rate = args.decay_rate
        self.gamma = args.gamma
        self.memory_size = args.memory_size

        self.memory = deque(maxlen=self.memory_size)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    @torch.no_grad()
    def get_action(self, state: torch.Tensor) -> int:
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
            action = random.randint(0, self.size**2 - 1)  # m * n - 1
        else:
            # Exploitation by selecting action according to policy
            # with highest predicted utility at current state.
            action = self.model.predict(state)

        return action

    def _memorize(self, events: dict) -> None:
        """Writes current events to memory (replay buffer).

        Args:
            events: Dictionary holding (states, actions, rewards, new_states, dones) tuple.

        """
        for state, action, rewards, new_state, done in zip(*events.values()):
            self.memory.append([state, action, rewards, new_state, done])

    def _epsilon_scheduler(self) -> None:
        """Decays epsilon-greedy value."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.decay_rate

    @torch.no_grad()
    def _create_training_set(self) -> None:
        """Create training set from memory."""

        # Use subset of replay memory for training as transitions are strongly correlated.
        replay_batch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        # Normalize the rewards of sampled batch.
        rewards = torch.tensor([memory[2] for memory in replay_batch])
        rewards = self._normalize_rewards(rewards=rewards)
        for memory, reward in zip(replay_batch, rewards):
            memory[2] = reward

        # Get states from replay buffer.
        states = torch.vstack([memory[0] for memory in replay_batch])
        new_states = torch.vstack([memory[3] for memory in replay_batch])

        self.model.eval()
        q_targets = self.model(states)
        q_targets_new = self.model(new_states)
        self.model.train()

        for i, (_, action, reward, _, done) in enumerate(replay_batch):
            if not done:
                q_targets[i, action] = reward + self.gamma * torch.amax(q_targets_new[i])
            else:
                q_targets[i, action] = reward

        return states, q_targets

    def train_on_batch(self, states: torch.Tensor, q_targets: torch.Tensor) -> float:
        """Performs single optimization step for batch of training data.

        Args:
            states: Tensor holding states.
            q_targets: Tensor holding q-targets.
        """
        self.optimizer.zero_grad()
        output_actions = self.model(states)
        loss = self.criterion(output_actions, q_targets)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def step(self, events: dict) -> None:
        """Runs single optimization step. Updates the network.

        Args:
            events: Tuple holding events.
        """

        self._memorize(events=events)
        states, q_targets = self._create_training_set()
        loss = self.train_on_batch(states=states, q_targets=q_targets)
        self._epsilon_scheduler()

        self.stats["loss"] = loss
        self.stats["reward"] = sum(events["rewards"])
        self.stats["epsilon"] = self.epsilon
