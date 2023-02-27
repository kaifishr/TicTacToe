"""Models used to learn playing Tic-Tac-Toe.

Fully-connected neural network with residual connections.
The models represent the agent's brain and map states
to actions.

"""
import torch
import torch.nn as nn


# class ResidualBlock(nn.Module):
#     """Simple MLP-Block."""
# 
#     def __init__(self, in_features: int, out_features: int) -> None:
#         """Initializes the model."""
#         super().__init__()
# 
#         prob_dropout = 0.05
# 
#         self.mlp_block = nn.Sequential(
#             nn.Linear(in_features=in_features, out_features=out_features),
#             nn.GELU(),
#             nn.Linear(in_features=out_features, out_features=out_features),
#             nn.Dropout(p=prob_dropout),
#             nn.LayerNorm(out_features),
#         )
# 
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return x + self.mlp_block(x)


class Model(nn.Module):

    def __init__(self, args) -> None:
        """Initializes the model."""
        super().__init__()

        field_size = args.field_size
        dims_state = field_size**2     
        num_actions = field_size**2    
        hidden_features = args.num_hidden_units
        prob_dropout = args.dropout_probability

        self.model = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=dims_state, out_features=hidden_features),
            nn.GELU(),
            nn.Dropout(p=prob_dropout),
            nn.Linear(in_features=hidden_features, out_features=hidden_features),
            nn.GELU(),
            nn.Dropout(p=prob_dropout),
            nn.Linear(in_features=hidden_features, out_features=num_actions),
            nn.Softmax(dim=-1) if args.algorithm == "policy_gradient" else nn.Identity()
        )

    @torch.no_grad() 
    def predict(self, state: torch.Tensor) -> int:  # predict -> get_action
        """Predicts action given a state.

        Args:
            state: Flattended playfield of size `size**2`.

        Returns:
            The action represented by an integer.
        """
        self.eval()  # TODO: Write decorator for eval() train() block
        prediction = self(state)
        action = torch.argmax(prediction, dim=-1).item()
        self.train()
        return action

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x