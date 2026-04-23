"""
models/autobot.py
-----------------
AutoBot-Joint: Autonomous Agent Behaviour Transformer for joint prediction
of all agents in a scene.

Architecture
------------
  Encoder
    • Per-agent temporal self-attention: each agent's T_obs-frame history
      is encoded independently with a Transformer encoder layer.
    • Social cross-attention: agents attend to each other's temporal
      encodings, building a scene-level representation.
    • Both stages are stacked L_enc times.

  Decoder
    • c latent mode queries (one per predicted mode) are decoded with
      alternating temporal and social cross-attention layers (L_dec times).
    • A lightweight MLP maps each query to bivariate-Gaussian parameters
      (μ_x, μ_y, σ_x, σ_y, ρ) at every future time step.

Reference: "Latent Variable Sequential Set Transformers for Joint
Multi-Agent Motion Prediction" (Girgis et al., ICLR 2022 / AutoBot)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional encoding  (sinusoidal, added to temporal embeddings)
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0).transpose(0, 1)   # [max_len, 1, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : [T, B, D]
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Output head: hidden → bivariate Gaussian parameters
# ---------------------------------------------------------------------------

class GaussianOutputHead(nn.Module):
    """MLP: d_k → (μ_x, μ_y, σ_x, σ_y, ρ) per time step."""

    def __init__(self, d_k: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_k, d_k), nn.ReLU(),
            nn.Linear(d_k, d_k), nn.ReLU(),
            nn.Linear(d_k, 5),
        )
        self.min_std = 0.01

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [T, B, d_k]  →  out : [T, B, 5]"""
        T, B, _ = x.shape
        raw = self.net(x.reshape(T * B, -1)).reshape(T, B, 5)
        mu = raw[..., :2]
        sig_x = F.softplus(raw[..., 2:3]) + self.min_std
        sig_y = F.softplus(raw[..., 3:4]) + self.min_std
        rho = torch.tanh(raw[..., 4:5]) * 0.9
        return torch.cat([mu, sig_x, sig_y, rho], dim=-1)


# ---------------------------------------------------------------------------
# AutoBot-Joint core module
# ---------------------------------------------------------------------------

class AutoBotJoint(nn.Module):
    """
    Parameters
    ----------
    k_attr      : int   input feature dim per agent per step (2 for x,y)
    d_k         : int   model hidden dimension
    M           : int   max number of other agents to consider
    c           : int   number of latent prediction modes
    T           : int   prediction horizon (future steps)
    L_enc       : int   number of encoder layers
    L_dec       : int   number of decoder layers
    num_heads   : int   attention heads
    tx_hidden   : int   feedforward dim inside Transformer layers
    dropout     : float dropout rate
    """

    def __init__(
        self,
        k_attr: int = 2,
        d_k: int = 128,
        M: int = 5,
        c: int = 1,
        T: int = 12,
        L_enc: int = 2,
        L_dec: int = 2,
        num_heads: int = 16,
        tx_hidden: int = 384,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_k = d_k
        self.M = M
        self.c = c
        self.T = T
        self.L_enc = L_enc
        self.L_dec = L_dec

        # ------ Input projection ------
        self.agent_encoder = nn.Linear(k_attr, d_k)
        self.pos_enc = PositionalEncoding(d_k, dropout=dropout)

        # ------ Encoder: temporal + social layers ------
        self.temporal_layers = nn.ModuleList()
        self.social_layers = nn.ModuleList()
        for _ in range(L_enc):
            self.temporal_layers.append(
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_k, num_heads, tx_hidden, dropout, batch_first=False),
                    num_layers=2,
                )
            )
            self.social_layers.append(
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_k, num_heads, tx_hidden, dropout, batch_first=False),
                    num_layers=1,
                )
            )

        # ------ Mode queries (learnable) ------
        self.mode_queries = nn.Parameter(torch.randn(c, 1, d_k))  # [c, 1, d_k]

        # ------ Decoder: temporal + social cross-attention ------
        self.dec_temporal = nn.ModuleList()
        self.dec_social = nn.ModuleList()
        for _ in range(L_dec):
            self.dec_temporal.append(
                nn.MultiheadAttention(d_k, num_heads, dropout=dropout)
            )
            self.dec_social.append(
                nn.MultiheadAttention(d_k, num_heads, dropout=dropout)
            )

        # Future temporal positional encoding for decoder queries
        dec_pe = torch.zeros(T, d_k)
        pos = torch.arange(0, T).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_k, 2).float() * (-math.log(10000.0) / d_k))
        dec_pe[:, 0::2] = torch.sin(pos * div)
        dec_pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("dec_pe", dec_pe)  # [T, d_k]

        self.output_head = GaussianOutputHead(d_k)
        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    def _encode(self, agents: torch.Tensor) -> torch.Tensor:
        """Encode all agents' observed trajectories.

        Parameters
        ----------
        agents : [T_obs, B*(M+1), k_attr]  – flattened (batch × agents) sequences

        Returns
        -------
        memory : [B*(M+1), d_k]  – per-agent context vectors (last time step)
        """
        T_obs, BM, _ = agents.shape

        # Project input coordinates
        x = self.agent_encoder(agents)          # [T_obs, BM, d_k]
        x = self.pos_enc(x)

        # Alternate temporal and social attention L_enc times
        for t_layer, s_layer in zip(self.temporal_layers, self.social_layers):
            # Temporal: each agent attends over its own time steps
            x = t_layer(x)                     # [T_obs, BM, d_k]

            # Social: at each time step, agents attend to each other
            # Reshape to [BM, T_obs, d_k] → transpose for attention → [T_obs, BM, d_k]
            x_social = x.permute(1, 0, 2)      # [BM, T_obs, d_k]
            # treat time-dim as the "sequence" for social attention
            x_social = x_social.reshape(1, BM, -1)  # naive: collapse time → social step
            # Proper social: run social on last time step's repr
            last = x[-1:, :, :]                # [1, BM, d_k]
            last_s = s_layer(last)             # [1, BM, d_k]
            x = x + last_s                     # residual broadcast

        return x[-1]                           # [BM, d_k]  last time step

    # ------------------------------------------------------------------
    def forward(
        self,
        observed: torch.Tensor,     # [T_obs, N, 2]  (single scene, N agents)
    ) -> torch.Tensor:
        """
        Returns
        -------
        pred : [c, T_pred, N, 2]  predicted mean positions per mode
        """
        T_obs, N, _ = observed.shape
        device = observed.device

        # Pad or truncate to M+1 agents  (index 0 = ego, 1..M = neighbours)
        agents = observed[:, : self.M + 1, :]       # [T_obs, ≤M+1, 2]
        # Pad missing agents with zeros
        if agents.shape[1] < self.M + 1:
            pad = torch.zeros(T_obs, self.M + 1 - agents.shape[1], 2, device=device)
            agents = torch.cat([agents, pad], dim=1)  # [T_obs, M+1, 2]

        # Replace NaN with 0
        agents = torch.nan_to_num(agents, nan=0.0)

        # Flatten agents into batch dim for encoder
        # [T_obs, M+1, 2] → [T_obs, M+1, 2]  (B=1 here, so BM = M+1)
        enc_in = agents                           # [T_obs, M+1, 2]

        # Encode
        context = self._encode(enc_in)            # [M+1, d_k]

        # Decoder: expand mode queries over future time steps
        # mode_queries: [c, 1, d_k] → [c, M+1, d_k]
        queries = self.mode_queries.expand(self.c, self.M + 1, self.d_k)  # [c, M+1, d_k]

        # Add future positional encoding
        dec_pe_exp = self.dec_pe.unsqueeze(1).unsqueeze(0)  # [1, T, 1, d_k]
        # Build [T, c*(M+1), d_k] for decoding
        dec_seq = queries.reshape(self.c * (self.M + 1), self.d_k)  # [c*(M+1), d_k]
        dec_seq = dec_seq.unsqueeze(0).expand(self.T, -1, -1)       # [T, c*(M+1), d_k]

        memory = context.unsqueeze(0).expand(self.T, -1, -1)        # [T, M+1, d_k]

        for t_attn, s_attn in zip(self.dec_temporal, self.dec_social):
            dec_seq, _ = t_attn(dec_seq, memory, memory)
            dec_seq, _ = s_attn(dec_seq, dec_seq, dec_seq)

        # dec_seq : [T, c*(M+1), d_k]
        # Reshape to [T, c, M+1, d_k]
        out = dec_seq.reshape(self.T, self.c, self.M + 1, self.d_k)

        # Gaussian parameters: just take the mean (μ_x, μ_y)
        # full output: [T, c, M+1, 5]
        T_, c_, MA, _ = out.shape
        params = self.output_head(
            out.reshape(T_, c_ * MA, self.d_k)
        ).reshape(T_, c_, MA, 5)

        # Cumulative sum to convert velocity predictions → absolute positions
        # (we predict displacements; starting point = last observed position)
        last_obs = agents[-1, : self.M + 1, :]     # [M+1, 2]
        deltas = params[..., :2]                    # [T, c, M+1, 2]
        pred = last_obs.unsqueeze(0).unsqueeze(0) + torch.cumsum(deltas, dim=0)

        return pred                                 # [T_pred, c, M+1, 2]


# ---------------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------------

class AutobotModel:
    """Load weights, run inference per scene."""

    def __init__(
        self,
        weights_path: str,
        obs_length: int = 9,
        pred_length: int = 12,
        d_k: int = 128,
        M: int = 5,
        c: int = 1,
        L_enc: int = 2,
        L_dec: int = 2,
        num_heads: int = 16,
        tx_hidden: int = 384,
        dropout: float = 0.1,
        device: str = "cpu",
    ):
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.device = torch.device(device)

        self.model = AutoBotJoint(
            k_attr=2,
            d_k=d_k,
            M=M,
            c=c,
            T=pred_length,
            L_enc=L_enc,
            L_dec=L_dec,
            num_heads=num_heads,
            tx_hidden=tx_hidden,
            dropout=dropout,
        ).to(self.device)

        ckpt = torch.load(weights_path, map_location=self.device)
        state = ckpt.get("state_dict", ckpt)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def predict(self, scene: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Return ego prediction for mode 0: [T_pred, 1, 2]."""
        obs = scene[: self.obs_length].to(self.device)
        pred = self.model(obs)          # [T_pred, c, M+1, 2]
        # mode 0, ego agent (index 0)
        return pred[:, 0, 0:1, :]      # [T_pred, 1, 2]
