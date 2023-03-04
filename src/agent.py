"""Holds agent base class.

The agent or learner.

"""
import torch


class Agent:
    """Abstract agent class.

    Attributes:
        model: Policy network.

    """

    def __init__(self, model: torch.nn.Module) -> None:
        """Initializes abstract learner class."""
        self.model = model

        self.stats = {"epsilon": None, "loss": None, "reward": None}

    def _normalize_rewards(self, rewards: torch.Tensor, eps: float = 1e-05) -> torch.Tensor:
        """Normalizes rewards.

        Normalizes rewards if there is more than one reward
        and if standard-deviation is non-zeros.

        Args:
            rewards: The agent's rewards.
            eps: Value added to the denominator for numerical stability.

        Returns:
            Normalized rewards.
        """
        if len(rewards) > 1:
            std = torch.std(rewards)
            if std != 0:
                mean = torch.mean(rewards)
                rewards = (rewards - mean) / (std + eps)
        return rewards

    @staticmethod
    def print_events(events: dict) -> None:
        """Prints events in a better format.

        Useful for debugging.

        Args:
                events: Tuple holding states, actions, rewards, new states, and termination token.
        """
        for key, value in events.items():
            print(f"{key} = \n")
            for item in value:
                print(f"{item}\n")
