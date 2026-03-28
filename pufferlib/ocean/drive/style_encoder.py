"""Causal Style Encoder for online driving style inference.

A GRU-based encoder that processes ego observations sequentially (past-only)
and outputs a style latent z at each timestep. The encoder only sees past
observations, avoiding future information leaking.

Key design:
  - Lightweight single-layer GRU (hidden=64)
  - Per-timestep operation for compatibility with RL rollout loop
  - Outputs (mu, log_var) for reparameterization trick
  - KL divergence regularizes z toward N(0, I) prior
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalStyleEncoder(nn.Module):
    """Infers driving style latent z from past ego observations using a GRU.

    At each timestep, takes the current ego observation (without z) and the
    hidden state from the previous timestep, and outputs a new z.

    Args:
        input_dim: dimensionality of ego observations (without z)
        z_dim: dimensionality of the style latent vector
        hidden_size: GRU hidden state size
    """

    def __init__(self, input_dim, z_dim=4, hidden_size=64):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_size = hidden_size

        # Project ego obs to GRU input size
        self.proj = nn.Linear(input_dim, hidden_size)
        # Single-layer GRU
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=1, batch_first=False)
        # Output heads for mu and log_var
        self.fc_mu = nn.Linear(hidden_size, z_dim)
        self.fc_logvar = nn.Linear(hidden_size, z_dim)

    def forward(self, ego_obs, hidden):
        """Run one GRU step and produce style z.

        Args:
            ego_obs: (batch, input_dim) ego observations at current timestep
            hidden: (1, batch, hidden_size) GRU hidden state from previous step

        Returns:
            z: (batch, z_dim) sampled or deterministic style vector
            new_hidden: (1, batch, hidden_size) updated hidden state
            mu: (batch, z_dim) mean of the latent distribution
            log_var: (batch, z_dim) log variance of the latent distribution
        """
        # Project to hidden dim
        x = F.gelu(self.proj(ego_obs))  # (batch, hidden_size)
        x = x.unsqueeze(0)  # (1, batch, hidden_size) for GRU

        gru_out, new_hidden = self.gru(x, hidden)  # (1, batch, hidden), (1, batch, hidden)
        gru_out = gru_out.squeeze(0)  # (batch, hidden_size)

        mu = self.fc_mu(gru_out)          # (batch, z_dim)
        log_var = self.fc_logvar(gru_out)  # (batch, z_dim)

        # Reparameterization trick (only during training)
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu  # deterministic at eval

        return z, new_hidden, mu, log_var

    def forward_sequence(self, ego_obs_seq, hidden=None):
        """Run the encoder over a full sequence of observations.

        Args:
            ego_obs_seq: (batch, seq_len, input_dim) sequence of ego observations
            hidden: (1, batch, hidden_size) initial hidden state (zeros if None)

        Returns:
            z_seq: (batch, seq_len, z_dim) style vectors at each timestep
            final_hidden: (1, batch, hidden_size) final hidden state
            mu_seq: (batch, seq_len, z_dim) means at each timestep
            logvar_seq: (batch, seq_len, z_dim) log variances at each timestep
        """
        batch, seq_len, _ = ego_obs_seq.shape

        if hidden is None:
            hidden = torch.zeros(1, batch, self.hidden_size, device=ego_obs_seq.device)

        z_list = []
        mu_list = []
        logvar_list = []

        for t in range(seq_len):
            z_t, hidden, mu_t, logvar_t = self.forward(ego_obs_seq[:, t, :], hidden)
            z_list.append(z_t)
            mu_list.append(mu_t)
            logvar_list.append(logvar_t)

        z_seq = torch.stack(z_list, dim=1)        # (batch, seq_len, z_dim)
        mu_seq = torch.stack(mu_list, dim=1)      # (batch, seq_len, z_dim)
        logvar_seq = torch.stack(logvar_list, dim=1)  # (batch, seq_len, z_dim)

        return z_seq, hidden, mu_seq, logvar_seq

    @staticmethod
    def kl_divergence(mu, log_var):
        """Compute KL divergence between q(z|x) and N(0, I).

        Args:
            mu: (batch, z_dim) or (batch, seq_len, z_dim)
            log_var: same shape as mu

        Returns:
            kl: scalar, mean KL divergence
        """
        return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    def init_hidden(self, batch_size, device="cpu"):
        """Create initial zero hidden state.

        Args:
            batch_size: number of agents
            device: torch device

        Returns:
            hidden: (1, batch_size, hidden_size)
        """
        return torch.zeros(1, batch_size, self.hidden_size, device=device)
