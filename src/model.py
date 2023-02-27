"""Models used to learn playing Tic-Tac-Toe.

Fully-connected neural network with residual connections.
The models represent the agent's brain and map states
to actions.

"""
import random
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
            nn.Softmax(dim=-1) if args.method == "policy_gradient" else nn.Identity()
        )

    @torch.no_grad()  # TODO: Move this to PolicyGradients?
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


class QNetwork(nn.Module):
    """Class information.

    More detailed class information.

    Attribute:
        in_features:
        out_features:

    TODO: Finish docstring.

    """

    def __init__(self, size: int) -> None:
        """Initializes the Q network."""
        super().__init__()

        self.size = size
        in_features = size**2  # state dimensions.
        out_features = size**2  # number of actions.
        hidden_features = 128
        prob_dropout = 0.0

        # self.epsilon = 0.9
        # self.epsilon_min = 0.01
        # self.decay_rate = 0.999

        self.model = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(p=prob_dropout),
            nn.Linear(in_features=hidden_features, out_features=hidden_features),
            nn.GELU(),
            nn.Dropout(p=prob_dropout),
            nn.Linear(in_features=hidden_features, out_features=out_features),
        )

    # @torch.no_grad()  # TODO: Move this to DeepQLearner?
    # def get_action(self, state: torch.Tensor) -> int:  # predict -> get_action
    #     """Returns action based on given state for current policy.

    #     Args:
    #         state: Flattened playing field of size `size**2`.

    #     Returns:
    #         The action represented by an integer.
    #     """
    #     self.eval()  # TODO: Write decorator for eval() train() block
    #     actions = self(state)
    #     action = torch.argmax(actions, dim=-1).item()
    #     self.train()
    #     return action

    # def _epsilon_scheduler(self) -> None:
    #     """Decays epsilon-greedy value."""
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.decay_rate

    @torch.no_grad()  # TODO: Move this to DeepQLearner?
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

    # @torch.no_grad()  # TODO: Move this to DeepQLearner?
    # def get_action(self, state: torch.Tensor) -> int:
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
    #         action = self.predict(state)

    #     self._epsilon_scheduler()

    #     return action

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x
