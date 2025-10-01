"""
Controller network for the Mario World Model.

The controller maps the latent state and memory hidden state to an action
probability distribution.  A simple multi-layer perceptron (MLP) is used.
"""

import torch
import torch.nn as nn


class Controller(nn.Module):
    def __init__(self,
                 input_dim: int,
                 action_dim: int,
                 hidden_sizes: tuple = ()):  # type: ignore[assignment]
        """
        Args:
            input_dim: Size of the input vector (latent + hidden state).
            action_dim: Number of discrete actions.
            hidden_sizes: Tuple of hidden layer sizes (empty for linear).
        """
        super().__init__()
        layers = []
        prev = input_dim
        for hs in hidden_sizes:
            layers.append(nn.Linear(prev, hs))
            layers.append(nn.Tanh())
            prev = hs
        layers.append(nn.Linear(prev, action_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """Forward pass producing action logits."""
        return self.model(x)

    def act(self, x: torch.Tensor, deterministic: bool = False):
        """
        Select an action given the input features.

        Args:
            x: Input tensor of shape (batch, input_dim)
            deterministic: If True return argmax action instead of sampling.

        Returns:
            actions: Tensor of shape (batch,) with chosen actions.
        """
        logits = self.forward(x)
        if deterministic:
            return torch.argmax(logits, dim=-1)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample()

