from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

from pufferlib.ocean.drive.style_encoder import CausalStyleEncoder

import pufferlib
import pufferlib.models

from pufferlib.models import Default as Policy  # noqa: F401
from pufferlib.models import Convolutional as Conv  # noqa: F401

Recurrent = pufferlib.models.LSTMWrapper


class Drive(nn.Module):
    def __init__(self, env, input_size=128, hidden_size=128, style_z_dim=0,
                 style_encoder_hidden_size=64, style_stop_grad_rl=True,
                 style_z_dropout_prob=0.5, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.observation_size = env.single_observation_space.shape[0]
        self.max_partner_objects = env.max_partner_objects
        self.partner_features = env.partner_features
        self.max_road_objects = env.max_road_objects
        self.road_features = env.road_features
        self.road_features_after_onehot = env.road_features + 6  # 6 is the number of one-hot encoded categories

        # Style encoder configuration
        self.style_z_dim = int(style_z_dim)
        self.style_stop_grad_rl = style_stop_grad_rl
        self._stop_grad_active = True  # Default: stop-grad on z (RL mode)
        self._last_mu = None
        self._last_logvar = None
        self._override_z = None  # For eval mode manual override
        self.style_z_dropout_prob = float(style_z_dropout_prob)

        # Ego dimensions: base (what C fills) vs full (base + z)
        self.ego_dim_base = getattr(env, 'ego_features_base', env.ego_features)
        self.ego_dim = env.ego_features

        # Create causal style encoder if enabled (with scene conditioning)
        if self.style_z_dim > 0:
            # Scene dim = partner_features (7) + road_features (7) = 14
            self.scene_dim = self.partner_features + self.road_features
            self.style_encoder = CausalStyleEncoder(
                input_dim=self.ego_dim_base,
                z_dim=self.style_z_dim,
                hidden_size=style_encoder_hidden_size,
                scene_dim=self.scene_dim,
            )

        # When style_z_dim > 0, ego obs is concatenated with z before encoding
        ego_encoder_input = self.ego_dim_base + self.style_z_dim if self.style_z_dim > 0 else self.ego_dim
        self.ego_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(ego_encoder_input, input_size)),
            nn.LayerNorm(input_size),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        self.road_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.road_features_after_onehot, input_size)),
            nn.LayerNorm(input_size),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        self.partner_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.partner_features, input_size)),
            nn.LayerNorm(input_size),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        self.shared_embedding = nn.Sequential(
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(3 * input_size, hidden_size)),
        )
        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)

        if self.is_continuous:
            self.atn_dim = (env.single_action_space.shape[0],) * 2
        else:
            self.atn_dim = env.single_action_space.nvec.tolist()

        self.actor = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, sum(self.atn_dim)), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def _compute_scene_summary(self, partner_obs, road_obs):
        """Compute mean-pooled scene summary features for the style encoder.

        Args:
            partner_obs: (batch, partner_dim) raw partner observations
            road_obs: (batch, road_dim) raw road observations

        Returns:
            scene_features: (batch, partner_features + road_features) = (batch, 14)
        """
        batch = partner_obs.shape[0]
        # Reshape to (batch, max_partners, features)
        partners = partner_obs.view(batch, self.max_partner_objects, self.partner_features)
        roads = road_obs.view(batch, self.max_road_objects, self.road_features)

        # Mean-pool over non-zero entries
        partner_mask = (partners.abs().sum(dim=-1, keepdim=True) > 1e-6).float()
        partner_count = partner_mask.sum(dim=1).clamp(min=1)
        partner_summary = (partners * partner_mask).sum(dim=1) / partner_count

        road_mask = (roads.abs().sum(dim=-1, keepdim=True) > 1e-6).float()
        road_count = road_mask.sum(dim=1).clamp(min=1)
        road_summary = (roads * road_mask).sum(dim=1) / road_count

        return torch.cat([partner_summary, road_summary], dim=-1)

    def encode_observations(self, observations, state=None):
        ego_dim = self.ego_dim
        partner_dim = self.max_partner_objects * self.partner_features
        road_dim = self.max_road_objects * self.road_features

        # Split observations
        ego_obs_full = observations[:, :ego_dim]
        partner_obs = observations[:, ego_dim : ego_dim + partner_dim]
        road_obs = observations[:, ego_dim + partner_dim : ego_dim + partner_dim + road_dim]

        # Style encoder: compute z from base ego obs + scene features (without z slots)
        if self.style_z_dim > 0:
            ego_obs_base = ego_obs_full[:, :self.ego_dim_base]  # without z

            # Compute scene summary features for encoder conditioning
            scene_features = self._compute_scene_summary(partner_obs, road_obs)

            if self._override_z is not None:
                # Manual override mode (eval with explicit z)
                z = self._override_z
                if z.shape[0] != ego_obs_base.shape[0]:
                    z = z[:ego_obs_base.shape[0]]
                self._last_mu = z
                self._last_logvar = torch.zeros_like(z)
            else:
                # Get hidden state from state dict if available
                encoder_hidden = None
                if state is not None:
                    encoder_hidden = state.get('style_encoder_hidden', None)

                if encoder_hidden is None:
                    # Determine correct batch size for hidden state
                    if state is not None and "seq_B" in state:
                        hidden_batch = state["seq_B"]  # BPTT: use B, not B*TT
                    else:
                        hidden_batch = ego_obs_base.shape[0]
                    encoder_hidden = self.style_encoder.init_hidden(
                        hidden_batch, ego_obs_base.device
                    )

                if state is not None and "seq_B" in state and "seq_TT" in state:
                    # BPTT Training Pass -> process sequentially
                    B, TT = state["seq_B"], state["seq_TT"]
                    ego_obs_seq = ego_obs_base.view(B, TT, -1)
                    scene_seq = scene_features.view(B, TT, -1)
                    
                    z_seq, new_hidden, mu_seq, logvar_seq = self.style_encoder.forward_sequence(
                        ego_obs_seq, hidden=encoder_hidden, scene_features_seq=scene_seq
                    )
                    
                    # Store variables flattened
                    z = z_seq.view(B * TT, -1)
                    self._last_mu = mu_seq.view(B * TT, -1)
                    self._last_logvar = logvar_seq.view(B * TT, -1)
                else:
                    # Rollout / Eval Pass -> process single step
                    z, new_hidden, mu, logvar = self.style_encoder(
                        ego_obs_base, encoder_hidden, scene_features
                    )
                    self._last_mu = mu
                    self._last_logvar = logvar

                # Store updated hidden in state
                if state is not None:
                    state['style_encoder_hidden'] = new_hidden

            # Apply stop-gradient during RL if configured
            if self._stop_grad_active and self.style_stop_grad_rl:
                z = z.detach()

            # Apply z-dropout: randomly zero out z for robustness
            if self.training and self.style_z_dropout_prob > 0:
                drop_mask = (torch.rand(z.shape[0], 1, device=z.device) < self.style_z_dropout_prob)
                z = z * (~drop_mask).float()

            # Build full ego obs with z
            ego_obs = torch.cat([ego_obs_base, z], dim=-1)  # (batch, ego_dim_base + z_dim)
        else:
            ego_obs = ego_obs_full

        partner_objects = partner_obs.view(-1, self.max_partner_objects, self.partner_features)

        road_objects = road_obs.view(-1, self.max_road_objects, self.road_features)
        road_continuous = road_objects[:, :, : self.road_features - 1]
        road_categorical = road_objects[:, :, self.road_features - 1]
        road_categorical = road_categorical.clamp(0, 6)
        road_onehot = F.one_hot(road_categorical.long(), num_classes=7)
        road_objects = torch.cat([road_continuous, road_onehot], dim=2)
        ego_features = self.ego_encoder(ego_obs)
        partner_features, _ = self.partner_encoder(partner_objects).max(dim=1)
        road_features, _ = self.road_encoder(road_objects).max(dim=1)

        concat_features = torch.cat([ego_features, road_features, partner_features], dim=1)

        # Pass through shared embedding
        embedding = F.relu(self.shared_embedding(concat_features))
        return embedding

    def decode_actions(self, flat_hidden):
        if self.is_continuous:
            parameters = self.actor(flat_hidden)
            loc, scale = torch.split(parameters, self.atn_dim, dim=1)
            std = torch.nn.functional.softplus(scale) + 1e-4
            action = torch.distributions.Normal(loc, std)
        else:
            action = self.actor(flat_hidden)
            action = torch.split(action, self.atn_dim, dim=1)

        value = self.value_fn(flat_hidden)

        return action, value

    def get_style_kl_loss(self):
        """Return the KL divergence from the last forward pass.

        Returns:
            kl_loss: scalar tensor, or 0 if no style encoder or no data.
        """
        if self.style_z_dim == 0 or self._last_mu is None:
            return torch.tensor(0.0)
        return CausalStyleEncoder.kl_divergence(self._last_mu, self._last_logvar)

    def set_style_override(self, z):
        """Override the style z with a fixed value (for eval).

        Args:
            z: tensor of shape (num_agents, z_dim) or None to disable override
        """
        self._override_z = z

    def set_stop_grad(self, active):
        """Control whether stop-gradient is applied to z.

        Args:
            active: if True, z is detached during forward pass (RL mode)
                   if False, gradients flow through z (expert training mode)
        """
        self._stop_grad_active = active
