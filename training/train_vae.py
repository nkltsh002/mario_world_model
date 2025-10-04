"""
Training script for the VAE component of the Mario World Model.

This script reads a dataset of random rollouts, extracts the observations,
and trains a convolutional VAE to compress them into a latent space.
The trained model is saved to disk.

Usage:
    python -m mario_world_model.training.train_vae --dataset data/mario_rollouts.pkl \
        --epochs 50 --batch-size 64 --latent-size 32 --learning-rate 1e-4 \
        --output checkpoints/vae_mario.pt
"""

import argparse
import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from mario_world_model.models.vae import ConvVAE
from mario_world_model.config import MarioConfig


def load_dataset(path: str):
    """Load rollouts from a pickle file and return a numpy array of observations."""
    with open(path, "rb") as f:
        trajectories = pickle.load(f)
    # Concatenate all observations across episodes
    obs_list = [traj["observations"] for traj in trajectories]
    # obs_list elements are arrays of shape (T, frame_stack, H, W)
    observations = np.concatenate(obs_list, axis=0)
    return observations  # shape (N, C, H, W)


def train_vae(dataset_path: str,
              epochs: int,
              batch_size: int,
              latent_size: int,
              learning_rate: float,
              output_path: str,
              beta: float = 4.0):
    """Train the VAE on the given dataset and save the model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_dataset(dataset_path)
    # Convert to tensor
    x = torch.tensor(data, dtype=torch.float32)
    # DataLoader
    ds = TensorDataset(x)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # Determine input channels from data
    input_channels = x.shape[1]
    vae = ConvVAE(input_channels=input_channels, latent_dim=latent_size).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        vae.train()
        total_loss = 0.0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            recon, mu, logvar = vae(batch_x)
            loss, rec_loss, kl = vae.loss_function(recon, batch_x, mu, logvar, beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{epochs}, loss={avg_loss:.4f}")

    # Save the trained model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(vae.state_dict(), output_path)
    print(f"Saved VAE to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train VAE for Mario World Model.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to rollouts pickle file")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--latent-size", type=int, default=None, help="Latent dimensionality")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--output", type=str, required=True, help="Path to save the VAE weights (.pt)")
    args = parser.parse_args()

    config = MarioConfig()
    epochs = args.epochs
    batch_size = args.batch_size or config.vae.batch_size
    latent_size = args.latent_size or config.vae.latent_size
    lr = args.learning_rate or config.vae.learning_rate

    train_vae(args.dataset, epochs, batch_size, latent_size, lr, args.output, beta=config.vae.beta)


if __name__ == "__main__":
    main()