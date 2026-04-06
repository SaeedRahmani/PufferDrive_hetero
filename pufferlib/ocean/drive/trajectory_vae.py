"""Scene-conditioned Trajectory VAE for learning driving style latent spaces.

This module provides:
  - TrajectoryVAE: a VAE that encodes kinematic feature sequences + scene context into a latent z
  - extract_kinematic_features: computes [speed, accel, ang_speed, ang_accel] per timestep
  - extract_scene_features: extracts scene summary from expert observations
  - train_vae: offline training script for the VAE
  - label_expert_data: labels expert sequences with pre-computed z vectors
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryEncoder(nn.Module):
    """Encodes a kinematic feature sequence + scene context into a latent distribution (mu, log_var).

    Scene conditioning allows the encoder to factor out scene-dependent behavior
    (e.g., slowing for red lights) from true driving style, so z captures only
    the residual style differences.
    """

    def __init__(self, input_dim=4, scene_dim=14, hidden_size=64, z_dim=4):
        super().__init__()
        self.scene_dim = scene_dim
        # Project kinematic features
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )
        # Project scene features
        self.scene_proj = nn.Sequential(
            nn.Linear(scene_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )
        # Bidirectional LSTM takes concatenated kinematic + scene projections
        self.lstm = nn.LSTM(hidden_size * 2, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        # Output size is hidden_size * 2 due to bidirectionality
        self.fc_mu = nn.Linear(hidden_size * 2, z_dim)
        self.fc_logvar = nn.Linear(hidden_size * 2, z_dim)

    def forward(self, x, scene=None):
        """
        Args:
            x: (batch, seq_len, input_dim) kinematic features
            scene: (batch, seq_len, scene_dim) scene context features, or None
        Returns:
            mu: (batch, z_dim)
            log_var: (batch, z_dim)
        """
        h_kin = self.proj(x)  # (batch, seq_len, hidden_size)
        if scene is not None and self.scene_dim > 0:
            h_scene = self.scene_proj(scene)  # (batch, seq_len, hidden_size)
        else:
            h_scene = torch.zeros_like(h_kin)
        h = torch.cat([h_kin, h_scene], dim=-1)  # (batch, seq_len, hidden_size * 2)
        lstm_out, (h_n, c_n) = self.lstm(h)
        # h_n shape: (num_layers * num_directions, batch, hidden_size) = (4, batch, 64)
        # We take the forward and backward hidden states from the last layer (indices 2 and 3)
        h_last_layer = torch.cat((h_n[-2], h_n[-1]), dim=-1)  # (batch, hidden_size * 2)
        return self.fc_mu(h_last_layer), self.fc_logvar(h_last_layer)


class TrajectoryDecoder(nn.Module):
    """Decodes a latent z into a reconstructed kinematic feature sequence."""

    def __init__(self, z_dim=4, hidden_size=64, output_dim=4, seq_len=91):
        super().__init__()
        self.seq_len = seq_len
        self.proj = nn.Sequential(
            nn.Linear(z_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=2, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_dim)
        )

    def forward(self, z):
        """
        Args:
            z: (batch, z_dim)
        Returns:
            recon: (batch, seq_len, output_dim)
        """
        h = self.proj(z)  # (batch, hidden)
        # Repeat z projection across time steps
        h = h.unsqueeze(1).expand(-1, self.seq_len, -1)  # (batch, seq_len, hidden)
        lstm_out, _ = self.lstm(h)  # (batch, seq_len, hidden)
        return self.out(lstm_out)  # (batch, seq_len, output_dim)


class TrajectoryVAE(nn.Module):
    """Scene-conditioned VAE for driving trajectory kinematic features.

    The encoder takes both kinematic features and scene context so that z
    captures only driving style, not scene-dependent behavior.
    The decoder reconstructs kinematic features from z alone.
    """

    def __init__(self, input_dim=4, z_dim=4, hidden_size=64, seq_len=91, scene_dim=14):
        super().__init__()
        self.z_dim = z_dim
        self.scene_dim = scene_dim
        self.encoder = TrajectoryEncoder(input_dim, scene_dim, hidden_size, z_dim)
        self.decoder = TrajectoryDecoder(z_dim, hidden_size, input_dim, seq_len)

    def reparameterize(self, mu, log_var):
        """Sample z using the reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, scene=None):
        """
        Args:
            x: (batch, seq_len, input_dim) kinematic features
            scene: (batch, seq_len, scene_dim) scene context, or None
        Returns:
            recon: (batch, seq_len, input_dim)
            mu: (batch, z_dim)
            log_var: (batch, z_dim)
        """
        mu, log_var = self.encoder(x, scene)
        z = self.reparameterize(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var

    def encode(self, x, scene=None, deterministic=True):
        """Encode a batch of sequences to latent z.

        Args:
            x: (batch, seq_len, input_dim) kinematic features
            scene: (batch, seq_len, scene_dim) scene context, or None
            deterministic: if True, return mu; otherwise sample
        Returns:
            z: (batch, z_dim)
        """
        mu, log_var = self.encoder(x, scene)
        if deterministic:
            return mu
        return self.reparameterize(mu, log_var)

    @staticmethod
    def vae_loss(recon, target, mu, log_var, beta=0.1):
        """Compute VAE loss = reconstruction + beta * KL divergence.

        Args:
            recon: (batch, seq_len, dim) reconstructed features
            target: (batch, seq_len, dim) original features
            mu: (batch, z_dim)
            log_var: (batch, z_dim)
            beta: weight on KL term
        Returns:
            total_loss, recon_loss, kl_loss (all scalars)
        """
        recon_loss = F.mse_loss(recon, target, reduction="mean")
        # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss


def extract_kinematic_features(expert_actions_continuous, dt=0.1):
    """Extract kinematic features from expert continuous actions.

    The expert continuous actions contain (acceleration, steering) per timestep.
    We compute derived kinematic features that characterize driving style:
      - acceleration (directly from actions)
      - steering magnitude (directly from actions)
      - speed (integrated from acceleration, approximate)
      - jerk (derivative of acceleration)

    Args:
        expert_actions_continuous: tensor of shape (n_sequences, seq_len, 2)
            where dim 2 is (acceleration, steering)
        dt: timestep duration in seconds

    Returns:
        features: tensor of shape (n_sequences, seq_len, 4)
            where dim 2 is (acceleration, abs_steering, approx_speed, jerk)
    """
    accel = expert_actions_continuous[:, :, 0]  # (n_seq, seq_len)
    steer = expert_actions_continuous[:, :, 1]  # (n_seq, seq_len)

    # Approximate speed by integrating acceleration with a decay factor (drag)
    # We iterate over seq_len to apply drag vs cumsum which acts perfectly frictionless
    drag_factor = 0.95
    speed = torch.zeros_like(accel)
    for t in range(1, accel.shape[1]):
        # v_t = max(v_{t-1} * drag + accel_{t-1} * dt, 0)
        speed[:, t] = (speed[:, t-1] * drag_factor + accel[:, t-1] * dt).clamp(min=0)

    # Jerk = derivative of acceleration
    jerk = torch.zeros_like(accel)
    jerk[:, 1:] = (accel[:, 1:] - accel[:, :-1]) / dt

    features = torch.stack([
        accel,
        steer.abs(),
        speed,
        jerk,
    ], dim=-1)  # (n_seq, seq_len, 4)

    return features


def extract_scene_features(expert_observations, ego_features, partner_features=7,
                           max_partners=31, road_features=7, max_roads=128):
    """Extract scene summary features from expert observations.

    Computes per-timestep scene summaries by aggregating partner and road
    observations. This allows the VAE encoder to condition on scene context
    and factor out scene-dependent behavior from driving style.

    Args:
        expert_observations: (n_sequences, seq_len, num_obs) full observation tensor
        ego_features: int, number of ego features (including z slots if any)
        partner_features: int, features per partner agent (default 7)
        max_partners: int, max number of partner agents (default 31)
        road_features: int, features per road segment (default 7)
        max_roads: int, max number of road segments (default 128)

    Returns:
        scene: (n_sequences, seq_len, 14) scene summary features:
            [:7] = mean-pooled partner features (over non-zero partners)
            [7:14] = mean-pooled road features (over non-zero road segments)
    """
    n_seq, seq_len, _ = expert_observations.shape

    # Extract partner and road observation blocks
    partner_start = ego_features
    partner_end = partner_start + max_partners * partner_features
    road_start = partner_end
    road_end = road_start + max_roads * road_features

    partner_obs = expert_observations[:, :, partner_start:partner_end]
    partner_obs = partner_obs.reshape(n_seq, seq_len, max_partners, partner_features)

    road_obs = expert_observations[:, :, road_start:road_end]
    road_obs = road_obs.reshape(n_seq, seq_len, max_roads, road_features)

    # Mean-pool over non-zero entries (zero = empty/padding)
    # Partner summary
    partner_mask = (partner_obs.abs().sum(dim=-1, keepdim=True) > 1e-6).float()
    partner_count = partner_mask.sum(dim=2).clamp(min=1)  # (n_seq, seq_len, 1)
    partner_summary = (partner_obs * partner_mask).sum(dim=2) / partner_count  # (n_seq, seq_len, 7)

    # Road summary
    road_mask = (road_obs.abs().sum(dim=-1, keepdim=True) > 1e-6).float()
    road_count = road_mask.sum(dim=2).clamp(min=1)
    road_summary = (road_obs * road_mask).sum(dim=2) / road_count  # (n_seq, seq_len, 7)

    scene = torch.cat([partner_summary, road_summary], dim=-1)  # (n_seq, seq_len, 14)
    return scene


def extract_scene_features(expert_observations, ego_features, partner_features=7,
                           max_partners=31, road_features=7, max_roads=128):
    """Extract scene summary features from expert observations.

    Computes per-timestep scene summaries by aggregating partner and road
    observations. This allows the VAE encoder to condition on scene context
    and factor out scene-dependent behavior from driving style.

    Args:
        expert_observations: (n_sequences, seq_len, num_obs) full observation tensor
        ego_features: int, number of ego features (including z slots if any)
        partner_features: int, features per partner agent (default 7)
        max_partners: int, max number of partner agents (default 31)
        road_features: int, features per road segment (default 7)
        max_roads: int, max number of road segments (default 128)

    Returns:
        scene: (n_sequences, seq_len, 14) scene summary features:
            [:7] = mean-pooled partner features (over non-zero partners)
            [7:14] = mean-pooled road features (over non-zero road segments)
    """
    n_seq, seq_len, _ = expert_observations.shape

    # Extract partner and road observation blocks
    partner_start = ego_features
    partner_end = partner_start + max_partners * partner_features
    road_start = partner_end
    road_end = road_start + max_roads * road_features

    partner_obs = expert_observations[:, :, partner_start:partner_end]
    partner_obs = partner_obs.reshape(n_seq, seq_len, max_partners, partner_features)

    road_obs = expert_observations[:, :, road_start:road_end]
    road_obs = road_obs.reshape(n_seq, seq_len, max_roads, road_features)

    # Mean-pool over non-zero entries (zero = empty/padding)
    # Partner summary
    partner_mask = (partner_obs.abs().sum(dim=-1, keepdim=True) > 1e-6).float()
    partner_count = partner_mask.sum(dim=2).clamp(min=1)  # (n_seq, seq_len, 1)
    partner_summary = (partner_obs * partner_mask).sum(dim=2) / partner_count  # (n_seq, seq_len, 7)

    # Road summary
    road_mask = (road_obs.abs().sum(dim=-1, keepdim=True) > 1e-6).float()
    road_count = road_mask.sum(dim=2).clamp(min=1)
    road_summary = (road_obs * road_mask).sum(dim=2) / road_count  # (n_seq, seq_len, 7)

    scene = torch.cat([partner_summary, road_summary], dim=-1)  # (n_seq, seq_len, 14)
    return scene


def train_vae(
    data_dir="resources/drive/human_demonstrations",
    z_dim=4,
    beta=0.1,
    hidden_size=64,
    epochs=200,
    lr=1e-3,
    batch_size=256,
    bptt_horizon=32,
    output_path=None,
    device="cuda",
    ego_features=194,
    scene_dim=14,
):
    """Train the scene-conditioned trajectory VAE on expert demonstration data.

    Args:
        data_dir: directory containing expert_actions_continuous_h{bptt_horizon}.pt
        z_dim: latent dimension
        beta: KL weight
        hidden_size: encoder/decoder hidden size
        epochs: number of training epochs
        lr: learning rate
        batch_size: training batch size
        bptt_horizon: sequence length used for expert data
        output_path: where to save the trained model (defaults to data_dir/vae_model.pt)
        device: 'cuda' or 'cpu'
        ego_features: number of ego features in observation (for scene extraction)
        scene_dim: scene feature dimensionality (partner_features + road_features)
    """
    if output_path is None:
        output_path = os.path.join(data_dir, "vae_model.pt")

    # Load expert continuous actions
    actions_path = os.path.join(data_dir, f"expert_actions_continuous_h{bptt_horizon}.pt")
    if not os.path.exists(actions_path):
        raise FileNotFoundError(
            f"Expert actions not found at {actions_path}. "
            f"Run training with prep_human_data=True first."
        )

    expert_actions = torch.load(actions_path, map_location="cpu", weights_only=False)
    print(f"Loaded expert actions: {expert_actions.shape}")  # (n_sequences, seq_len, 2)

    # Extract kinematic features
    features = extract_kinematic_features(expert_actions)
    print(f"Extracted kinematic features: {features.shape}")  # (n_sequences, seq_len, 4)

    # Load and extract scene features from expert observations
    obs_path = os.path.join(data_dir, f"expert_observations_h{bptt_horizon}.pt")
    if os.path.exists(obs_path):
        expert_obs = torch.load(obs_path, map_location="cpu", weights_only=False)
        scene_features = extract_scene_features(expert_obs, ego_features=ego_features)
        print(f"Extracted scene features: {scene_features.shape}")  # (n_seq, seq_len, 14)
        # Normalize scene features
        scene_mean = scene_features.mean(dim=(0, 1), keepdim=True)
        scene_std = scene_features.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
        scene_norm = (scene_features - scene_mean) / scene_std
        torch.save({"mean": scene_mean.squeeze(), "std": scene_std.squeeze()},
                    os.path.join(data_dir, "vae_scene_stats.pt"))
    else:
        print(f"[WARNING] Expert observations not found at {obs_path}, training without scene conditioning.")
        scene_norm = None
        scene_dim = 0

    # Normalize kinematic features to zero mean, unit variance
    feat_mean = features.mean(dim=(0, 1), keepdim=True)
    feat_std = features.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
    features_norm = (features - feat_mean) / feat_std

    # Save normalization stats for later use
    torch.save({"mean": feat_mean.squeeze(), "std": feat_std.squeeze()},
               os.path.join(data_dir, "vae_feature_stats.pt"))

    # Create VAE
    input_dim = features.shape[-1]
    seq_len = features.shape[1]
    vae = TrajectoryVAE(input_dim=input_dim, z_dim=z_dim, hidden_size=hidden_size,
                        seq_len=seq_len, scene_dim=scene_dim)
    vae = vae.to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_samples = len(features_norm)
    features_norm = features_norm.to(device)
    if scene_norm is not None:
        scene_norm = scene_norm.to(device)

    print(f"\nTraining VAE: z_dim={z_dim}, beta={beta}, scene_dim={scene_dim}, epochs={epochs}, samples={n_samples}")
    print(f"Output: {output_path}\n")

    anneal_epochs = epochs // 2

    best_loss = float("inf")
    for epoch in range(epochs):
        # KL Annealing: linearly increase beta from 0 to target_beta over anneal_epochs
        current_beta = beta * min(1.0, epoch / max(1, anneal_epochs))
        
        vae.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        n_batches = 0

        # Shuffle
        perm = torch.randperm(n_samples, device=device)
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            batch = features_norm[idx]
            scene_batch = scene_norm[idx] if scene_norm is not None else None

            recon, mu, log_var = vae(batch, scene_batch)
            loss, recon_loss, kl_loss = TrajectoryVAE.vae_loss(recon, batch, mu, log_var, current_beta)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        avg_recon = epoch_recon / n_batches
        avg_kl = epoch_kl / n_batches

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Recon: {avg_recon:.4f} | KL: {avg_kl:.4f} | Beta: {current_beta:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state_dict": vae.state_dict(),
                "z_dim": z_dim,
                "input_dim": input_dim,
                "hidden_size": hidden_size,
                "seq_len": seq_len,
                "beta": beta,
                "scene_dim": scene_dim,
            }, output_path)

    print(f"\nVAE training complete. Best loss: {best_loss:.4f}")
    print(f"Model saved to: {output_path}")
    return output_path


def label_expert_data(
    vae_model_path,
    data_dir="resources/drive/human_demonstrations",
    bptt_horizon=32,
    device="cuda",
    ego_features=194,
):
    """Label expert sequences with pre-computed z vectors from the trained VAE.

    Args:
        vae_model_path: path to trained VAE checkpoint
        data_dir: directory containing expert data
        bptt_horizon: sequence length
        device: 'cuda' or 'cpu'
        ego_features: number of ego features in observation (for scene extraction)

    Returns:
        z_vectors: tensor of shape (n_sequences, z_dim)
    """
    # Load VAE
    checkpoint = torch.load(vae_model_path, map_location=device, weights_only=False)
    scene_dim = checkpoint.get("scene_dim", 0)
    vae = TrajectoryVAE(
        input_dim=checkpoint["input_dim"],
        z_dim=checkpoint["z_dim"],
        hidden_size=checkpoint["hidden_size"],
        seq_len=checkpoint["seq_len"],
        scene_dim=scene_dim,
    )
    vae.load_state_dict(checkpoint["model_state_dict"])
    vae = vae.to(device)
    vae.eval()

    # Load expert continuous actions
    actions_path = os.path.join(data_dir, f"expert_actions_continuous_h{bptt_horizon}.pt")
    expert_actions = torch.load(actions_path, map_location="cpu", weights_only=False)

    # Extract and normalize kinematic features
    features = extract_kinematic_features(expert_actions)

    stats_path = os.path.join(data_dir, "vae_feature_stats.pt")
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    features_norm = (features - stats["mean"]) / stats["std"].clamp(min=1e-6)

    # Extract and normalize scene features if needed
    scene_norm = None
    if scene_dim > 0:
        obs_path = os.path.join(data_dir, f"expert_observations_h{bptt_horizon}.pt")
        if os.path.exists(obs_path):
            expert_obs = torch.load(obs_path, map_location="cpu", weights_only=False)
            scene_features = extract_scene_features(expert_obs, ego_features=ego_features)
            scene_stats_path = os.path.join(data_dir, "vae_scene_stats.pt")
            if os.path.exists(scene_stats_path):
                scene_stats = torch.load(scene_stats_path, map_location="cpu", weights_only=False)
                scene_norm = (scene_features - scene_stats["mean"]) / scene_stats["std"].clamp(min=1e-6)
            else:
                scene_mean = scene_features.mean(dim=(0, 1), keepdim=True)
                scene_std = scene_features.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
                scene_norm = (scene_features - scene_mean) / scene_std

    # Encode all sequences
    z_list = []
    batch_size = 512
    with torch.no_grad():
        for i in range(0, len(features_norm), batch_size):
            batch = features_norm[i:i + batch_size].to(device)
            scene_batch = scene_norm[i:i + batch_size].to(device) if scene_norm is not None else None
            z = vae.encode(batch, scene_batch, deterministic=True)
            z_list.append(z.cpu())

    z_vectors = torch.cat(z_list, dim=0)  # (n_sequences, z_dim)

    # Save
    output_path = os.path.join(data_dir, f"expert_style_z_h{bptt_horizon}.pt")
    torch.save(z_vectors, output_path)

    print(f"Labeled {len(z_vectors)} expert sequences with z_dim={checkpoint['z_dim']}")
    print(f"z stats: mean={z_vectors.mean(0).tolist()}, std={z_vectors.std(0).tolist()}")
    print(f"Saved to: {output_path}")

    return z_vectors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train trajectory VAE or label expert data")
    parser.add_argument("mode", choices=["train", "label"], help="'train' the VAE or 'label' expert data")
    parser.add_argument("--data-dir", default="resources/drive/human_demonstrations")
    parser.add_argument("--z-dim", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bptt-horizon", type=int, default=32)
    parser.add_argument("--vae-model-path", default=None, help="Path to VAE model (for labeling)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ego-features", type=int, default=194, help="Number of ego features in obs (for scene extraction)")
    parser.add_argument("--scene-dim", type=int, default=14, help="Scene feature dimensionality")
    args = parser.parse_args()

    if args.mode == "train":
        train_vae(
            data_dir=args.data_dir,
            z_dim=args.z_dim,
            beta=args.beta,
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            bptt_horizon=args.bptt_horizon,
            device=args.device,
            ego_features=args.ego_features,
            scene_dim=args.scene_dim,
        )
    elif args.mode == "label":
        vae_path = args.vae_model_path or os.path.join(args.data_dir, "vae_model.pt")
        label_expert_data(
            vae_model_path=vae_path,
            data_dir=args.data_dir,
            bptt_horizon=args.bptt_horizon,
            device=args.device,
            ego_features=args.ego_features,
        )
