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