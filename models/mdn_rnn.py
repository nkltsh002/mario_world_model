"""
Mixture Density Network RNN (MDN‑RNN) used in the Mario World Model.

The MDN‑RNN models the temporal dynamics in the latent space of the VAE.
Given the current latent vector and the action taken, it predicts a
probability distribution over the next latent vector.  We model the
distribution as a mixture of Gaussians with diagonal covariance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class MDNRNN(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 action_dim: int,
                 hidden_size: int = 256,
                 num_gaussians: int = 5,
                 num_layers: int = 1):
        """
        Args:
            latent_dim: Dimensionality of the latent space from the VAE.
            action_dim: Number of discrete actions in the environment.
            hidden_size: Hidden size of the LSTM.
            num_gaussians: Number of mixture components.
            num_layers: Number of LSTM layers.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_gaussians = num_gaussians
        self.num_layers = num_layers

        self.lstm = nn.LSTM(latent_dim + action_dim, hidden_size, num_layers)
        # Output layers produce mixture parameters per latent dimension
        # We predict a mixture of Gaussians independently for each dimension.
        self.fc_pi = nn.Linear(hidden_size, latent_dim * num_gaussians)
        self.fc_mu = nn.Linear(hidden_size, latent_dim * num_gaussians)
        self.fc_sigma = nn.Linear(hidden_size, latent_dim * num_gaussians)

    def forward(self, inputs: torch.Tensor, hidden=None):
        """
        Forward pass through the LSTM and MDN heads.

        Args:
            inputs: Tensor of shape (seq_len, batch, latent_dim + action_dim)
            hidden: Optional tuple (h0, c0) of initial hidden state

        Returns:
            log_pi, mu, sigma, hidden: mixture parameters and final hidden state
        """
        outputs, hidden = self.lstm(inputs, hidden)
        # outputs: (seq_len, batch, hidden_size)
        # Reshape to (seq_len * batch, hidden_size) to feed into linear layers
        y = outputs.view(-1, self.hidden_size)
        # Raw outputs
        pi = self.fc_pi(y)
        mu = self.fc_mu(y)
        sigma = self.fc_sigma(y)
        # Reshape to (seq_len, batch, num_gaussians, latent_dim)
        seq_len, batch = inputs.size(0), inputs.size(1)
        pi = pi.view(seq_len, batch, self.num_gaussians, self.latent_dim)
        mu = mu.view(seq_len, batch, self.num_gaussians, self.latent_dim)
        # Use softplus to ensure sigma is positive and add small epsilon
        sigma = F.softplus(sigma.view(seq_len, batch, self.num_gaussians, self.latent_dim)) + 1e-6
        # Use log softmax for mixture weights
        log_pi = F.log_softmax(pi, dim=2)
        return log_pi, mu, sigma, hidden

    def mdn_loss(self, log_pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor):
        """
        Compute the negative log-likelihood of target under the mixture model.

        Args:
            log_pi: Log mixture weights (seq_len, batch, M, latent_dim)
            mu: Means of Gaussians (seq_len, batch, M, latent_dim)
            sigma: Standard deviations (seq_len, batch, M, latent_dim)
            target: Target latent vectors (seq_len, batch, latent_dim)

        Returns:
            Scalar negative log-likelihood averaged over sequence and batch.
        """
        seq_len, batch, num_gaussians, latent_dim = mu.size()
        # Expand target to match mixture dims: (seq_len, batch, M, latent_dim)
        target_exp = target.unsqueeze(2).expand(-1, -1, num_gaussians, -1)
        # Construct normal distributions
        dist = Normal(mu, sigma)
        # Compute log-prob for each component: (seq_len, batch, M, latent_dim)
        log_prob = dist.log_prob(target_exp)
        # Combine with log mixture weights using log-sum-exp across mixture components
        log_prob = torch.logsumexp(log_pi + log_prob, dim=2)
        # Sum over latent dimensions before averaging
        log_prob = log_prob.sum(dim=2)
        # Negative log-likelihood
        nll = -log_prob.mean()
        return nll

    def sample(self, log_pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor):
        """
        Sample a latent vector from the mixture distribution at the last time step.

        Args:
            log_pi: Log mixture weights (seq_len, batch, M, latent_dim)
            mu: Means (seq_len, batch, M, latent_dim)
            sigma: Standard deviations (seq_len, batch, M, latent_dim)

        Returns:
            Sampled latent vector of shape (batch, latent_dim)
        """
        # Use the last time step for sampling
        log_pi_last = log_pi[-1]  # (batch, M, latent_dim)
        mu_last = mu[-1]
        sigma_last = sigma[-1]
        # Convert log_pi to probabilities
        pi_last = torch.exp(log_pi_last)
        batch, num_gaussians, latent_dim = pi_last.size()
        samples = []
        for b in range(batch):
            dim_samples = []
            for d in range(latent_dim):
                # Mixture probabilities for dimension d
                probs = pi_last[b, :, d]
                comp_idx = torch.distributions.Categorical(probs=probs).sample()
                comp_mu = mu_last[b, comp_idx, d]
                comp_sigma = sigma_last[b, comp_idx, d]
                eps = torch.randn_like(comp_mu)
                dim_samples.append(comp_mu + comp_sigma * eps)
            samples.append(torch.stack(dim_samples, dim=0))
        return torch.stack(samples, dim=0)