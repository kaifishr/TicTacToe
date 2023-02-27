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

        self.stats = {
            "epsilon": None,
            "loss": None,
            "reward": None
        }

    @torch.no_grad()
    def predict(self, state: torch.Tensor) -> int:  # predict -> get_action
        """Predicts action given a state.

        Args:
            state: Flattended playfield of size `size**2`.

        Returns:
            The action represented by an integer.
        """
        self.model.eval()  # TODO: Write decorator for eval() train() block
        prediction = self.model(state)
        action = torch.argmax(prediction, dim=-1).item()
        self.model.train()
        return action
    
    def normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalizes rewards."""

        # TODO: Move this to base class
        # if len(discounted_rewards) > 1:
        #     std = torch.std(discounted_rewards)
        #     if std != 0:
        #         mean = torch.mean(discounted_rewards)
        #         discounted_rewards = (discounted_rewards - mean) / (std + 1e-5)  # the sample weight

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