"""Models used to learn playing Tic-Tac-Toe.

Fully-connected neural network with residual connections.
The models represent the agent's brain and map states
to actions.

"""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Simple MLP-Block."""

    def __init__(self, in_features: int, out_features: int) -> None:
        """Initializes the model."""
        super().__init__()

        prob_dropout = 0.05

        self.mlp_block = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.GELU(),
            nn.Linear(in_features=out_features, out_features=out_features),
            nn.Dropout(p=prob_dropout),
            nn.LayerNorm(out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp_block(x)


class Model(nn.Module):
    def __init__(self, size: int) -> None:
        """Initializes the model."""
        super().__init__()

        in_features = size**2  # state dimensions.
        out_features = size**2  # number of actions.
        hidden_features = 128
        prob_dropout = 0.0

        self.model = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(p=prob_dropout),
            nn.Linear(in_features=hidden_features, out_features=hidden_features),
            nn.GELU(),
            nn.Dropout(p=prob_dropout),
            nn.Linear(in_features=hidden_features, out_features=out_features),
            nn.Softmax(dim=-1),
        )

    @torch.no_grad()  # TODO: Move this to PolicyGradients?
    def predict(self, state: torch.Tensor) -> int:
        """Predicts action given a state.

        Args:
            state: Flattended playfield of size `size**2`.

        Returns:
            The action represented by an integer.
        """
        self.eval()
        prediction = self(state)
        action = torch.argmax(prediction, dim=-1).item()
        self.train()
        return action

    @torch.no_grad()  # TODO: Move this to PolicyGradients?
    def sample_action(self, state: torch.Tensor) -> int:
        """Samples action given a state.

        Args:
            state: Flattended playfield of size `size**2`.

        Returns:
            Sampled action represented by an integer.
        """
        self.eval()
        # Build the probability density function (PDF) for the given state.
        action_prob = self(state)
        # Sample action according to (PDF)
        action = torch.multinomial(action_prob, num_samples=1).item()
        self.train()
        return action

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x
