"""
Convolutional Variational Autoencoder (VAE) used in the Mario World Model.

This VAE compresses stacked grayscale frames into a low‑dimensional latent
vector.  It uses a convolutional encoder and a deconvolutional decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    def __init__(self, input_channels: int = 4, latent_dim: int = 32):
        """
        Args:
            input_channels: Number of channels in the input (frame stack).
            latent_dim: Dimensionality of the latent space.
        """
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: conv layers reduce 64x64 input to flat latent representation
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 8x8 -> 4x4
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

        # Decoder: transpose conv layers to reconstruct image
        self.decoder_input = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 4x4 -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),  # 32x32 -> 64x64
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor):
        """Encode input images into latent mean and log variance."""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        """Reparameterisation trick to sample from N(mu, sigma^2)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        """Decode latent samples back into image space."""
        h = self.decoder_input(z)
        h = h.view(h.size(0), 256, 4, 4)
        recon = self.decoder(h)
        return recon

    def forward(self, x: torch.Tensor):
        """Forward pass returning reconstruction and latent parameters."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @staticmethod
    def loss_function(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 4.0):
        """
        Compute the beta‑VAE loss: reconstruction + beta * KL divergence.

        Args:
            recon_x: Reconstructed images.
            x: Original images.
            mu: Latent mean.
            logvar: Latent log variance.
            beta: Weight of the KL divergence term.

        Returns:
            total_loss, reconstruction_loss, kl_divergence
        """
        # Binary cross entropy (reconstruction loss)
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.size(0)
        # KL divergence between approximate posterior and unit Gaussian
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + beta * kl, recon_loss, kl