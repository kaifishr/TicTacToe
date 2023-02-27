"""Script with helper functions."""
import numpy
import pathlib
import random

import torch


def set_random_seed(seed: int = 0) -> None:
    """Sets random seed."""
    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoint(model: torch.nn.Module, model_name: str, args) -> None:
    """Saves model checkpoint.

    Uses torch.save() to save PyTorch models.

    Args:
        model: A PyTorch model.
        model_name:
        args:
    """
    checkpoint_name = f"{f'{model_name}_{args.algorithm}' if model_name else 'model'}"
    checkpoint_path = "weights"
    model_path = pathlib.Path(checkpoint_path) / f"{checkpoint_name}.pth"

    torch.save(obj=model.state_dict(), f=model_path)


def load_checkpoint(model: torch.nn.Module, model_name: str, args) -> None:
    """Loads model from checkpoint.

    Args:
        model: Neural network.
        model_name:
        args:
    """
    checkpoint_name = f"{f'{model_name}_{args.algorithm}' if model_name else 'model'}"
    checkpoint_path = "weights"
    model_path = pathlib.Path(checkpoint_path) / f"{checkpoint_name}.pth"

    if model_path.is_file():
        state_dict = torch.load(f=model_path)
        model.load_state_dict(state_dict=state_dict)
        print("Model loaded.")
    else:
        print(f"Model checkpoint '{checkpoint_name}' not found. " "Continuing with random weights.")
