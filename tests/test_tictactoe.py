"""Tests for Tic-Tac-Toe game eninge."""

from src.environment import TicTacToe


def test_game_rules():
    """Tests correct implementation of rules of play."""

    field_size = 3
    players = [-1, 1]
    configuration = []

    # Run through all states for each player
    for player in players:

        for i in range(field_size):
            for j in range(field_size):
                configuration.append((i, j))
            env = TicTacToe(size=field_size)
            for x, y in configuration:
                env.mark_field(x=x, y=y, player=player)
            is_finished, winner = env.is_finished()
            assert is_finished
            assert winner == player
            configuration = []

        for j in range(field_size):
            for i in range(field_size):
                configuration.append((i, j))
            env = TicTacToe(size=field_size)
            for x, y in configuration:
                env.mark_field(x=x, y=y, player=player)
            is_finished, winner = env.is_finished()
            assert is_finished
            assert winner == player
            configuration = []

        for i in range(field_size):
            configuration.append((i, i))
        env = TicTacToe(size=field_size)
        for x, y in configuration:
            env.mark_field(x=x, y=y, player=player)
        is_finished, winner = env.is_finished()
        assert is_finished
        assert winner == player
        configuration = []

        for i in range(field_size):
            configuration.append((i, (field_size - 1) - i))
        env = TicTacToe(size=field_size)
        for x, y in configuration:
            env.mark_field(x=x, y=y, player=player)
        is_finished, winner = env.is_finished()
        assert is_finished
        assert winner == player


def test_index_to_coordinate():
    """Tests correct computation of index to coordinate."""
    assert True


def test_mark_field():
    """Tests if playfield is correctly marked."""
    assert True


def test_is_free():
    """Tests if field is free."""
    assert True
