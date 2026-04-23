"""
models/eq_motion.py
-------------------
EqMotion: Equivariant Multi-agent Motion Prediction (Xu et al., CVPR 2023).

Key idea
--------
The model is built around an SE(2)-equivariant message-passing network.
Each layer computes:
  1. Edge features from relative positions and node features.
  2. Updated node features by aggregating neighbour messages.
  3. Coordinate updates (equivariant shift) from edge features.

The trajectory is decoded by iteratively predicting future positions
with the equivariant network, conditioned on the past trajectory.

This is a self-contained, clean re-implementation that captures the
structure of the original code found in vita-epfl/s-attack/certified/
baselines/EqMotion, but is written from scratch for clarity.

Architecture
------------
  Encoder: GRU over past T_obs positions → node features h_i ∈ R^{nf}
  EqMotion layers (n_layers): equivariant message passing over the
      dynamic agent graph → updated h_i and positional updates Δx_i
  Decoder: linear projection of h_i → T_pred future displacements

Reference: "EqMotion: Equivariant Multi-agent Motion Prediction with
           Invariant Interaction Reasoning" Xu et al., CVPR 2023
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Equivariant layer building blocks
# ---------------------------------------------------------------------------

class EdgeModel(nn.Module):
    """Compute edge messages from node features and relative coordinates."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * node_dim + 1, hidden_dim),   # 1 extra: distance scalar
            nn.SiLU(),
            nn.Linear(hidden_dim, edge_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        h_i: torch.Tensor,     # [E, node_dim]
        h_j: torch.Tensor,     # [E, node_dim]
        dist: torch.Tensor,    # [E, 1]  ||x_i - x_j||
    ) -> torch.Tensor:          # [E, edge_dim]
        return self.net(torch.cat([h_i, h_j, dist], dim=-1))


class NodeModel(nn.Module):
    """Aggregate edge messages → updated node features."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )

    def forward(
        self,
        h: torch.Tensor,            # [N, node_dim]
        agg_msgs: torch.Tensor,     # [N, edge_dim]  sum over neighbours
    ) -> torch.Tensor:              # [N, node_dim]
        return self.net(torch.cat([h, agg_msgs], dim=-1))


class CoordModel(nn.Module):
    """Equivariant coordinate update: weight relative positions by scalar."""

    def __init__(self, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        msgs: torch.Tensor,     # [N, N, edge_dim]  edge features
        rel: torch.Tensor,      # [N, N, 2]          relative positions x_j - x_i
    ) -> torch.Tensor:          # [N, 2]             Δx per agent
        weights = self.net(msgs)        # [N, N, 1]
        # Zero out self-edges
        eye = torch.eye(msgs.shape[0], device=msgs.device).unsqueeze(-1)
        weights = weights * (1 - eye)
        delta = (weights * rel).sum(dim=1)  # [N, 2]
        return delta


# ---------------------------------------------------------------------------
# Single EqMotion layer
# ---------------------------------------------------------------------------

class EqMotionLayer(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.edge_model = EdgeModel(node_dim, edge_dim, hidden_dim)
        self.node_model = NodeModel(node_dim, edge_dim, hidden_dim)
        self.coord_model = CoordModel(edge_dim, hidden_dim)

    def forward(
        self,
        h: torch.Tensor,    # [B, N, node_dim]
        x: torch.Tensor,    # [B, N, 2]
    ):
        B, N, D = h.shape

        # Relative positions and distances
        rel = x.unsqueeze(2) - x.unsqueeze(1)          # [B, N, N, 2]
        dist = rel.norm(dim=-1, keepdim=True)           # [B, N, N, 1]

        # Edge features  [B, N, N, edge_dim]
        h_i = h.unsqueeze(2).expand(B, N, N, D)
        h_j = h.unsqueeze(1).expand(B, N, N, D)
        dist_flat = dist.reshape(B * N * N, 1)
        hi_flat = h_i.reshape(B * N * N, D)
        hj_flat = h_j.reshape(B * N * N, D)

        msgs_flat = self.edge_model(hi_flat, hj_flat, dist_flat)   # [B*N*N, edge_dim]
        edge_dim = msgs_flat.shape[-1]
        msgs = msgs_flat.reshape(B, N, N, edge_dim)

        # Aggregate messages: sum over neighbours (j)
        agg = msgs.sum(dim=2)           # [B, N, edge_dim]

        # Node update
        h_new = self.node_model(
            h.reshape(B * N, D),
            agg.reshape(B * N, edge_dim),
        ).reshape(B, N, D)
        h_new = h + h_new               # residual

        # Coordinate update (equivariant)
        delta = self.coord_model(
            msgs.reshape(B * N, N, edge_dim),
            rel.reshape(B * N, N, 2),
        ).reshape(B, N, 2)
        x_new = x + delta

        return h_new, x_new


# ---------------------------------------------------------------------------
# Full EqMotion model
# ---------------------------------------------------------------------------

class EqMotion(nn.Module):
    """
    Parameters
    ----------
    obs_length    : int    number of observed frames
    pred_length   : int    number of frames to predict
    node_dim      : int    hidden size per agent
    edge_dim      : int    edge feature size
    hidden_dim    : int    MLP hidden size inside layers
    n_layers      : int    number of EqMotion message-passing layers
    """

    def __init__(
        self,
        obs_length: int = 9,
        pred_length: int = 12,
        node_dim: int = 64,
        edge_dim: int = 64,
        hidden_dim: int = 64,
        n_layers: int = 4,
    ):
        super().__init__()
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.node_dim = node_dim

        # Encode past trajectory: velocities over T_obs → node feature
        self.traj_encoder = nn.GRU(
            input_size=2,
            hidden_size=node_dim,
            num_layers=1,
            batch_first=True,
        )

        # Velocity encoder as node feature supplement
        self.vel_encoder = nn.Sequential(
            nn.Linear(2, node_dim),
            nn.SiLU(),
        )

        # EqMotion layers
        self.layers = nn.ModuleList([
            EqMotionLayer(node_dim, edge_dim, hidden_dim)
            for _ in range(n_layers)
        ])

        # Decode future positions: node feature → T_pred × 2 displacements
        self.decoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, pred_length * 2),
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        observed: torch.Tensor,     # [B, T_obs, N, 2]
        num_valid: torch.Tensor | None = None,  # [B]  valid agent count
    ) -> torch.Tensor:              # [B, T_pred, N, 2]
        """
        Parameters
        ----------
        observed  : [B, T_obs, N, 2]  past positions (may contain zeros for padding)
        num_valid : [B]  number of real agents per scene (unused in forward,
                         kept for API compatibility)

        Returns
        -------
        pred : [B, T_pred, N, 2]  predicted future positions
        """
        B, T, N, _ = observed.shape
        device = observed.device

        # Replace NaN with 0 for padding
        obs = torch.nan_to_num(observed, nan=0.0)

        # Velocities:  Δx_t = x_t - x_{t-1}
        vel = torch.zeros_like(obs)
        vel[:, 1:] = obs[:, 1:] - obs[:, :-1]
        vel[:, 0] = vel[:, 1]

        # Encode per-agent trajectory with GRU
        # Reshape to [B*N, T, 2]
        vel_flat = vel.permute(0, 2, 1, 3).reshape(B * N, T, 2)  # [B*N, T, 2]
        _, h_gru = self.traj_encoder(vel_flat)                    # h: [1, B*N, node_dim]
        h = h_gru.squeeze(0).reshape(B, N, self.node_dim)         # [B, N, node_dim]

        # Current position (last observed)
        x = obs[:, -1, :, :]                       # [B, N, 2]

        # EqMotion message-passing layers
        for layer in self.layers:
            h, x = layer(h, x)

        # Decode: h → future displacements
        deltas = self.decoder(h.reshape(B * N, self.node_dim))      # [B*N, T_pred*2]
        deltas = deltas.reshape(B, N, self.pred_length, 2)
        deltas = deltas.permute(0, 2, 1, 3)                         # [B, T_pred, N, 2]

        # Convert displacements to absolute positions  (cumulative sum)
        last_pos = obs[:, -1:, :, :]                                # [B, 1, N, 2]
        pred = last_pos + torch.cumsum(deltas, dim=1)               # [B, T_pred, N, 2]

        return pred


# ---------------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------------

class EqMotionModel:
    """Load weights and run inference per scene.

    Internally EqMotion works with up to N_MAX = 3 agents (primary + 2 closest
    neighbours). Scenes with fewer agents are zero-padded; the output for
    padding agents is ignored.
    """

    N_MAX = 3   # maximum agents passed to the model (matches original code)

    def __init__(
        self,
        weights_path: str,
        obs_length: int = 9,
        pred_length: int = 12,
        node_dim: int = 64,
        edge_dim: int = 64,
        hidden_dim: int = 64,
        n_layers: int = 4,
        device: str = "cpu",
    ):
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.device = torch.device(device)

        self.model = EqMotion(
            obs_length=obs_length,
            pred_length=pred_length,
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        ).to(self.device)

        ckpt = torch.load(weights_path, map_location=self.device)
        state = ckpt.get("state_dict", ckpt)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

    # ------------------------------------------------------------------
    @staticmethod
    def _drop_missing(scene_np):
        """Drop agents with any NaN in the observed frames."""
        import numpy as np
        # scene_np: [T, N, 2]
        T, N, _ = scene_np.shape
        keep = []
        for n in range(N):
            if not any(np.isnan(scene_np[t, n, 0]) for t in range(min(9, T))):
                keep.append(n)
        return scene_np[:, keep, :]

    @staticmethod
    def _keep_closest(scene_np, max_n: int):
        """Keep at most max_n agents (ego + closest neighbours)."""
        import numpy as np
        if scene_np.shape[1] <= max_n:
            return scene_np
        dist_sq = np.nansum(np.square(scene_np - scene_np[:, 0:1]), axis=2)  # [T, N]
        min_dist = np.nanmin(dist_sq, axis=0)                                  # [N]
        idx = np.argsort(min_dist)[:max_n]
        return scene_np[:, idx, :]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, scene: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Return ego prediction [T_pred, 1, 2].

        Parameters
        ----------
        scene : [T_full, N, 2]
        goal  : [N, 2]  (unused by EqMotion)
        """
        import numpy as np

        scene_np = scene.cpu().numpy()  # [T, N, 2]

        # Pre-processing to match original code
        scene_np = self._drop_missing(scene_np)
        scene_np = self._keep_closest(scene_np, self.N_MAX)

        T, N_real, _ = scene_np.shape
        N_pad = self.N_MAX

        # Pad to N_MAX agents
        if N_real < N_pad:
            pad = np.zeros((T, N_pad - N_real, 2))
            scene_np = np.concatenate([scene_np, pad], axis=1)

        obs_np = scene_np[: self.obs_length]                    # [T_obs, N_pad, 2]
        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
        obs_t = obs_t.permute(1, 0, 2).unsqueeze(0)            # [1, T_obs, N, 2] → wait

        # EqMotion expects [B, T_obs, N, 2]
        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self.device)  # [T_obs, N, 2]
        obs_in = obs_t.unsqueeze(0)                              # [1, T_obs, N, 2]

        pred = self.model(obs_in)                                # [1, T_pred, N, 2]

        # Return ego (agent 0) only
        return pred[0, :, 0:1, :]                               # [T_pred, 1, 2]
