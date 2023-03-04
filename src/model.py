"""Policy model.

Fully-connected neural network with residual connections.
The models represent the agent's policy and map states
to actions.

"""
import torch
import torch.nn as nn

from src.utils import eval


class ResidualBlock(nn.Module):
    """Simple MLP-block."""

    def __init__(self, in_features: int, out_features: int, args) -> None:
        """Initializes residual MLP-block."""
        super().__init__()

        self.mlp_block = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.GELU(),
            nn.Linear(in_features=out_features, out_features=out_features),
            nn.Dropout(p=args.dropout_probability),
            nn.LayerNorm(out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp_block(x)


class Model(nn.Module):
    def __init__(self, args) -> None:
        """Initializes the model."""
        super().__init__()

        field_size = args.field_size
        dims_state = field_size**2
        num_actions = field_size**2
        hidden_features = args.num_hidden_units
        num_layers = args.num_layers
        prob_dropout = args.dropout_probability

        input_layer = [
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=dims_state, out_features=hidden_features),
            nn.GELU(),
            nn.Dropout(p=prob_dropout),
        ]

        hidden_layers = [
            ResidualBlock(in_features=hidden_features, out_features=hidden_features, args=args)
            for _ in range(num_layers)
        ]

        output_layer = [
            nn.Linear(in_features=hidden_features, out_features=num_actions),
            nn.Softmax(dim=-1) if args.algorithm == "policy_gradient" else nn.Identity(),
        ]

        self.model = nn.Sequential(*input_layer, *hidden_layers, *output_layer)

    @eval
    @torch.no_grad()
    def predict(self, state: torch.Tensor) -> int:
        """Predicts action for given state.

        Args:
            state: Tensor representing state of playing field.

        Returns:
            The action represented by an integer.
        """
        prediction = self(state)
        action = torch.argmax(prediction, dim=-1).item()
        return action

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x
