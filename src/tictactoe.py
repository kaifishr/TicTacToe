"""Tic-Tac-Toe game engine.

Tic-Tac-Toe game with option for arbitrarily large
playing field.

Typical usage:

    # Play a game against the computer.
    env = TicTacToe(size=3)
    env.play()

"""
import copy
import random
import torch
import torch.nn as nn


class Environment:
    """Environment base class."""

    def __init__(self) -> None:
        """Initializes Environment."""


class TicTacToe(Environment):
    """Tic-tac-toe environment.

    A simple environment for the game Tic-Tac-Toe.

    o|o|x
    o|x|o
    x|o|x

    # Action Space

    The action is an integer which can take values [0, size**2 - 1] indicating
    where the agent marks the field with either 1 (x) or -1 (o).

    # Observation Space

    The observation is a PyTorch tensor with shape (size, size) representing the
    playfield.

    # Rewards

        +1 for winning
        -1 for loosing
        0 if game is draw
        -1 for a wrong move

    # Initial State

    An empty playfield represented by a PyTorch tensor with shape (size, size)
    initialized with zeros.

    # Episode End

    The episode ends if any of the following events occur:

    1. The game was lost / won.
    2. Game ends in a draw.
    3. A wrong move was made (forces the agent to learn the game's rules).
    """

    def __init__(self, size: int = 3) -> None:
        """Initializes a square TicTacToe field."""
        super().__init__()
        self.size = size
        self.field = torch.zeros(size=(size, size), dtype=torch.long)

    @torch.no_grad()
    def _has_won(self, player: int) -> bool:
        """Checks if player has won the game.

        Args:
            player: The player's id.

        Returns:
            Boolean indicating if game is finished
            (True if game is finished, False otherwise).
        """
        criterion = player * self.size

        # Rows
        for i in range(self.size):
            if self.field[i, :].sum() == criterion:
                return True

        # Columns
        for i in range(self.size):
            if self.field[:, i].sum() == criterion:
                return True

        # First diagonal
        if torch.diagonal(self.field).sum() == criterion:
            return True

        # Second diagonal
        if torch.diagonal(torch.rot90(self.field)).sum() == criterion:
            return True

        return False

    @torch.no_grad()
    def _is_draw(self) -> bool:
        """Checks if the game is tied.

        Return:
            Boolean indicating if game is a draw."""
        return torch.all(self.field != 0)

    @torch.no_grad()
    def is_finished(self) -> tuple[bool, int]:
        """Checks if game is finished.

        Returns:
            Tuple with boolean indicating if game
            is finished (True if game is finished, False otherwise)
            and winner of game (player X (1) or player O (-1))
        """
        if self._has_won(player=1):
            return True, 1
        elif self._has_won(player=-1):
            return True, -1
        elif self._is_draw():
            return True, 0
        else:
            return False, 0

    def is_free(self, x: int, y: int) -> bool:
        """Checks whether field is free or already marked.

        Args:
            x: x-coordinate.
            y: y-coordinate.

        Returns:
            True is field is free. False if field is already marked.
        """
        return self.field[x, y].item() == 0.0

    @torch.no_grad()
    def mark_field(self, x: int, y: int, player: int) -> None:
        """Marks field with either X (1) or O (-1)

        Args:
            x: x-coordinate.
            y: y-coordinate.
            player: Integer representing player A (1) or B (-1).
        """
        if player == 1:
            self.field[x, y] = 1.0
        else:  # player == -1
            self.field[x, y] = -1.0

    def index_to_coordinate(self, index: int) -> tuple[int, int]:
        """Converts a flat index into a coordinate tuple.

        Args:
            index: The index to be converted to a coordinate.

        Returns:
            Tuple with coordinates.
        """
        x = index // self.size
        y = index % self.size
        return x, y

    def step(self, action: int, player: int) -> tuple:
        """Performs a single game move for player.

        Args:
            action: The action predicted by the neural network.
            The action is represented by an integer in the range
            [0, size**2 - 1].

        Returns:
            A tuple holding information about state, reward, and
            if game is finished.

        """
        x, y = self.index_to_coordinate(action)

        if self.is_free(x=x, y=y):
            self.mark_field(x=x, y=y, player=player)
            is_finished, winner = self.is_finished()
            if is_finished and winner == player:
                # Maximum reward for winning the game.
                reward = 1.0
            elif is_finished and winner == 0:
                # No reward if game is a draw.
                reward = 0.0
            else:
                # Zero / positive reward for correctly marking a field.
                reward = 0.0
        else:  # Negative reward and end of game if occupied field is marked.
            reward = -1.0
            is_finished = True

        state = self.field.float()[None, ...]
        done = is_finished

        return state, reward, done

    def episode(self, model_a: nn.Module, model_b: nn.Module) -> tuple:  # episode -> play_episode / rollout_episode
        """Agents play one episode of the game.

        The episode stops if the game is won, lost or a draw.

        Args:
            model_a: The model representing the agent a's policy.
            model_b: The model representing the agent b's policy.

        Returns:
            Tuple for each agent holding states, actions, rewards,
            new_states, and dones of the episode played.
        """
        events_a = dict(states=[], actions=[], rewards=[], new_states=[], dones=[])
        events_b = dict(states=[], actions=[], rewards=[], new_states=[], dones=[])

        state = self.reset()
        done = False

        while not done:

            # Agent a
            # action = model_a.predict(state)
            action = model_a.sample_action(state)
            new_state, reward, done = self.step(action=action, player=-1)

            events_a["states"].append(copy.deepcopy(state))
            events_a["actions"].append(action)
            events_a["rewards"].append(reward)
            events_a["new_states"].append(copy.deepcopy(new_state))
            events_a["dones"].append(done)

            # TODO: Check reward scheme: Make sure player gets negative reward if other player wins.
            if done and reward == 1:
                events_b["rewards"][-1] = -1

            state = new_state

            # Agent b
            if not done:
                # action = model_b.predict(state)
                action = model_b.sample_action(state)
                new_state, reward, done = self.step(action=action, player=1)

                events_b["states"].append(copy.deepcopy(state))
                events_b["actions"].append(action)
                events_b["rewards"].append(reward)
                events_b["new_states"].append(copy.deepcopy(new_state))
                events_b["dones"].append(done)

                # TODO: Add reward -1 to player that lost the game.
                if done and reward == 1:
                    events_a["rewards"][-1] = -1

                state = new_state

        return events_a, events_b

    def play(self, model: nn.Module, num_rounds: int = 10) -> None:
        """Plays game against an agent."""

        for _ in range(num_rounds):
            state = self.reset()
            print(self)

            done = False

            while not done:

                # Machine
                print("Agent")
                player = -1
                action = model.predict(state)
                state, reward, done = self.step(action=action, player=player)
                print(self)

                if done:
                    if reward == 1:
                        print("You lose.")
                    if reward == -1:
                        print("Computer lost.")

                # Human
                if not done:
                    player = 1
                    action = int(input("Enter an index: "))
                    state, reward, done = self.step(action=action, player=player)
                    print(self)

                    if done:
                        if reward == 1:
                            print("You win.")
                        if reward == -1:
                            print("Computer won.")

    def reset(self) -> torch.Tensor:
        """Resets the playflied."""
        self.field = torch.zeros(size=(self.size, self.size), dtype=torch.long)
        state = self.field.float()[None, ...]
        return state

    def __repr__(self) -> str:
        """Prints playing field."""
        frmt = "{:^3}" * self.size
        return "\n" + "\n".join(frmt.format(*row) for row in self.field.tolist()) + "\n"
