"""
Training script for the MDN‑RNN component of the Mario World Model.

This script reads a dataset of random rollouts, encodes observations using
a trained VAE, forms sequences of latent vectors and actions, and trains
an MDN‑RNN to predict the next latent vector.  The trained model is saved
to disk.

Usage:
    python -m mario_world_model.training.train_mdn_rnn --dataset data/mario_rollouts.pkl \
        --vae checkpoints/vae_mario.pt \
        --sequence-length 100 --epochs 30 \
        --output checkpoints/mdn_rnn_mario.pt
"""

import argparse
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from models.vae import ConvVAE
from models.mdn_rnn import MDNRNN
from config import MarioConfig


class LatentSequenceDataset(Dataset):
    """
    Dataset of latent sequences and action sequences for MDN‑RNN training.

    Each item is a tuple (inputs, targets) where:
      inputs: Tensor of shape (seq_len, latent_dim + action_dim)
      targets: Tensor of shape (seq_len, latent_dim)
    """

    def __init__(self,
                 trajectories,
                 vae: ConvVAE,
                 sequence_length: int,
                 action_dim: int,
                 device: torch.device):
        self.inputs = []
        self.targets = []
        self.device = device
        vae.eval()
        # Preprocess each trajectory
        for traj in trajectories:
            obs = traj["observations"]  # shape (T, C, H, W)
            actions = traj["actions"]  # shape (T,)
            # Encode observations to latent means
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
                mu, logvar = vae.encode(obs_tensor)
                latents = mu.cpu().numpy()  # shape (T, latent_dim)
            # One-hot encode actions
            one_hot_actions = np.eye(action_dim)[actions]
            # Form sequences
            T = len(actions)
            if T <= sequence_length:
                continue
            for i in range(T - sequence_length):
                latent_seq = latents[i : i + sequence_length]
                action_seq = one_hot_actions[i : i + sequence_length]
                # Input is concatenation of latent and action
                inp = np.concatenate([latent_seq, action_seq], axis=1)
                target = latents[i + 1 : i + sequence_length + 1]
                self.inputs.append(torch.tensor(inp, dtype=torch.float32))
                self.targets.append(torch.tensor(target, dtype=torch.float32))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def load_dataset(path: str):
    with open(path, "rb") as f:
        trajectories = pickle.load(f)
    return trajectories


def train_mdn_rnn(dataset_path: str,
                  vae_path: str,
                  sequence_length: int,
                  hidden_size: int,
                  num_gaussians: int,
                  epochs: int,
                  batch_size: int,
                  learning_rate: float,
                  output_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load dataset and determine action dimension
    trajectories = load_dataset(dataset_path)
    if len(trajectories) == 0:
        raise ValueError('Dataset is empty - collect rollouts first.')
    # Determine action dimension from max action
    max_action = 0
    for traj in trajectories:
        if traj["actions"].max() > max_action:
            max_action = int(traj["actions"].max())
    action_dim = max_action + 1

    # Load VAE
    sample_obs = None
    for traj in trajectories:
        if len(traj['observations']) > 0:
            sample_obs = traj['observations'][0]
            break
    if sample_obs is None:
        raise ValueError('No observations available to infer frame dimensions.')
    frame_stack = sample_obs.shape[0]
    # Recreate VAE with appropriate dimensions
    latent_size = MarioConfig().vae.latent_size
    vae = ConvVAE(input_channels=frame_stack, latent_dim=latent_size).to(device)
    vae.load_state_dict(torch.load(vae_path, map_location=device))

    # Build dataset
    ds = LatentSequenceDataset(trajectories, vae, sequence_length, action_dim, device)
    if len(ds) == 0:
        raise ValueError("No latent sequences were generated. Collect longer episodes or decrease --sequence-length.")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    # Instantiate MDN-RNN
    mdnrnn = MDNRNN(latent_dim=latent_size,
                    action_dim=action_dim,
                    hidden_size=hidden_size,
                    num_gaussians=num_gaussians).to(device)
    optimizer = torch.optim.Adam(mdnrnn.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        mdnrnn.train()
        total_loss = 0.0
        for inputs, targets in loader:
            # inputs: (batch, seq_len, latent_dim + action_dim)
            # Transpose to (seq_len, batch, input_dim)
            inputs = inputs.to(device).transpose(0, 1)
            targets = targets.to(device).transpose(0, 1)  # (seq_len, batch, latent_dim)
            optimizer.zero_grad()
            log_pi, mu, sigma, _ = mdnrnn(inputs)
            loss = mdnrnn.mdn_loss(log_pi, mu, sigma, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{epochs}, loss={avg_loss:.4f}")

    # Save model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(mdnrnn.state_dict(), output_path)
    print(f"Saved MDN-RNN to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train MDN-RNN for Mario World Model.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to rollouts pickle file")
    parser.add_argument("--vae", type=str, required=True, help="Path to trained VAE weights (.pt)")
    parser.add_argument("--sequence-length", type=int, default=None, help="Sequence length")
    parser.add_argument("--hidden-size", type=int, default=None, help="LSTM hidden size")
    parser.add_argument("--num-gaussians", type=int, default=None, help="Number of mixture components")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--output", type=str, required=True, help="Path to save MDN-RNN weights (.pt)")
    args = parser.parse_args()

    config = MarioConfig()
    seq_len = args.sequence_length or config.mdnrnn.sequence_length
    hidden_size = args.hidden_size or config.mdnrnn.hidden_size
    num_gaussians = args.num_gaussians or config.mdnrnn.num_mixtures
    epochs = args.epochs or config.mdnrnn.epochs
    batch_size = args.batch_size or config.mdnrnn.batch_size
    lr = args.learning_rate or config.mdnrnn.learning_rate

    train_mdn_rnn(args.dataset, args.vae, seq_len, hidden_size, num_gaussians,
                  epochs, batch_size, lr, args.output)


if __name__ == "__main__":
    main()