"""
Configuration module for the Mario World Model project.

This module defines dataclasses encapsulating the hyper‑parameters
for each component of the World Models architecture as well as
environment settings.  Modify these values directly or override them
via command‑line arguments in the training scripts.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class VAEConfig:
    """Configuration for the variational autoencoder (VAE)."""
    latent_size: int = 32
    beta: float = 4.0
    learning_rate: float = 1e-4
    batch_size: int = 64
    epochs: int = 50


@dataclass
class MDNRNNConfig:
    """Configuration for the mixture density network recurrent model."""
    hidden_size: int = 256
    num_mixtures: int = 5
    sequence_length: int = 100
    learning_rate: float = 1e-4
    batch_size: int = 64
    epochs: int = 30


@dataclass
class ControllerConfig:
    """Configuration for the controller."""
    hidden_sizes: Tuple[int, ...] = field(default_factory=tuple)
    population_size: int = 64
    sigma: float = 0.5
    iterations: int = 200


@dataclass
class PPOConfig:
    """Configuration for the PPO baseline."""
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    batch_size: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95


@dataclass
class MarioConfig:
    """Top‑level configuration for the Mario World Model project."""
    env_name: str = "SuperMarioBros-1-1-v0"
    vae: VAEConfig = VAEConfig()
    mdnrnn: MDNRNNConfig = MDNRNNConfig()
    controller: ControllerConfig = ControllerConfig()
    ppo: PPOConfig = PPOConfig()