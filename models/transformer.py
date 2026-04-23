"""
models/transformer.py
---------------------
Social Transformer (ST) for multi-agent trajectory prediction.

Architecture
------------
  1. **Input projection**: linear(2 → d_model) per (agent, time-step).
  2. **Temporal encoding**: sinusoidal positional encoding over observed frames.
  3. **Transformer encoder**: each agent's observation sequence is encoded
     with L stacked Transformer encoder layers (self-attention in time ×
     cross-attention in social dimension).
  4. **Decoder**: learned query tokens for each future time step attend to
     the encoded agent memories via cross-attention, then predict Δx, Δy.

Social attention
  At each encoder layer the model first applies self-attention over the
  T_obs frames of each agent independently (temporal), then applies
  self-attention across agents at every time step (social). This is the
  classic "factored attention" design used in Social-Transmotion and
  related models.

Input/Output
  - Input:  scene  [T_obs, N, 2]  observed positions (world coords)
  - Output: pred   [T_pred, N, 2] predicted future positions

Reference style: Social-Transmotion (Saadatnejad et al., ICLR 2024) /
                 AgentFormer (Yuan et al., ICCV 2021)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Utility: sinusoidal positional encoding
# ---------------------------------------------------------------------------

def sinusoidal_pe(length: int, d_model: int, device: torch.device) -> torch.Tensor:
    """Return [length, d_model] sinusoidal positional encoding."""
    pe = torch.zeros(length, d_model, device=device)
    pos = torch.arange(0, length, device=device).float().unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, device=device).float()
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


# ---------------------------------------------------------------------------
# Factored attention block (temporal + social)
# ---------------------------------------------------------------------------

class FactoredAttentionLayer(nn.Module):
    """One block of temporal self-attention followed by social self-attention.

    Parameters
    ----------
    d_model   : int   model width
    num_heads : int   attention heads
    ff_dim    : int   feedforward hidden size
    dropout   : float dropout probability
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 8,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        # --- Temporal self-attention ---
        self.temporal_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # --- Social self-attention ---
        self.social_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # --- Feedforward ---
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [T, N, d_model]

        Returns
        -------
        x : [T, N, d_model]
        """
        T, N, D = x.shape

        # -- Temporal self-attention: each agent over its own T frames --
        # Reshape to [T, N, D] → process each agent separately as batch
        x_t = x.permute(1, 0, 2)           # [N, T, D]
        x_t_in = x_t.reshape(N, T, D)
        x_t_q = x_t_in.permute(1, 0, 2)   # [T, N, D]  (MHA expects seq-first)
        attn_out, _ = self.temporal_attn(x_t_q, x_t_q, x_t_q)
        x = self.norm1(x + self.drop(attn_out))   # [T, N, D]

        # -- Social self-attention: agents attend to each other per frame --
        x_s = x                             # [T, N, D]  reuse
        # Treat N as the sequence dimension (agents) per time step
        x_s_q = x_s                        # [T, N, D]  → permute for MHA
        x_s_perm = x_s_q.permute(1, 0, 2)  # [N, T, D]
        attn_out_s, _ = self.social_attn(x_s_perm, x_s_perm, x_s_perm)
        attn_out_s = attn_out_s.permute(1, 0, 2)   # [T, N, D]
        x = self.norm2(x + self.drop(attn_out_s))

        # -- Feedforward --
        x = self.norm3(x + self.ff(x))

        return x


# ---------------------------------------------------------------------------
# Decoder block with cross-attention to encoder memory
# ---------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,      # [T_pred, N, D]  decoder queries
        memory: torch.Tensor,   # [T_obs,  N, D]  encoder output
    ) -> torch.Tensor:
        # Self-attention over future queries
        a, _ = self.self_attn(tgt, tgt, tgt)
        tgt = self.norm1(tgt + self.drop(a))
        # Cross-attention: queries attend to encoder memory
        a, _ = self.cross_attn(tgt, memory, memory)
        tgt = self.norm2(tgt + self.drop(a))
        tgt = self.norm3(tgt + self.ff(tgt))
        return tgt


# ---------------------------------------------------------------------------
# Full Social Transformer model
# ---------------------------------------------------------------------------

class SocialTransformer(nn.Module):
    """
    Parameters
    ----------
    obs_length  : int   observed frames
    pred_length : int   frames to predict
    d_model     : int   model width
    num_heads   : int   attention heads
    ff_dim      : int   feedforward hidden dim
    L_enc       : int   number of encoder layers
    L_dec       : int   number of decoder layers
    dropout     : float dropout
    """

    def __init__(
        self,
        obs_length: int = 9,
        pred_length: int = 12,
        d_model: int = 128,
        num_heads: int = 8,
        ff_dim: int = 256,
        L_enc: int = 3,
        L_dec: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.d_model = d_model

        # Input projection: (x, y) → d_model
        self.input_proj = nn.Linear(2, d_model)

        # Encoder stack
        self.encoder_layers = nn.ModuleList([
            FactoredAttentionLayer(d_model, num_heads, ff_dim, dropout)
            for _ in range(L_enc)
        ])

        # Decoder queries (learnable; one per future step)
        self.dec_queries = nn.Parameter(torch.randn(pred_length, 1, d_model))

        # Decoder stack
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, ff_dim, dropout)
            for _ in range(L_dec)
        ])

        # Output head: d_model → 2 (Δx, Δy)
        self.out_proj = nn.Linear(d_model, 2)

    # ------------------------------------------------------------------
    def forward(
        self,
        observed: torch.Tensor,     # [T_obs, N, 2]
    ) -> torch.Tensor:              # [T_pred, N, 2]
        T, N, _ = observed.shape
        device = observed.device

        # Replace NaN with 0 (padding)
        obs = torch.nan_to_num(observed, nan=0.0)

        # --- Input projection + positional encoding ---
        x = self.input_proj(obs)                           # [T, N, D]
        pe = sinusoidal_pe(T, self.d_model, device)        # [T, D]
        x = x + pe.unsqueeze(1)                            # [T, N, D]

        # --- Encoder ---
        memory = x
        for layer in self.encoder_layers:
            memory = layer(memory)                         # [T_obs, N, D]

        # --- Decoder ---
        # Expand learnable queries over N agents
        tgt = self.dec_queries.expand(self.pred_length, N, self.d_model)   # [T_pred, N, D]
        future_pe = sinusoidal_pe(self.pred_length, self.d_model, device)  # [T_pred, D]
        tgt = tgt + future_pe.unsqueeze(1)

        for layer in self.decoder_layers:
            tgt = layer(tgt, memory)                       # [T_pred, N, D]

        # --- Output: predict displacements → cumsum → absolute positions ---
        deltas = self.out_proj(tgt)                        # [T_pred, N, 2]
        last_pos = obs[-1]                                 # [N, 2]
        pred = last_pos.unsqueeze(0) + torch.cumsum(deltas, dim=0)   # [T_pred, N, 2]

        return pred


# ---------------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------------

class TransformerModel:
    """Load weights, run inference per scene."""

    def __init__(
        self,
        weights_path: str | None = None,
        obs_length: int = 9,
        pred_length: int = 12,
        d_model: int = 128,
        num_heads: int = 8,
        ff_dim: int = 256,
        L_enc: int = 3,
        L_dec: int = 2,
        dropout: float = 0.1,
        device: str = "cpu",
    ):
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.device = torch.device(device)

        self.model = SocialTransformer(
            obs_length=obs_length,
            pred_length=pred_length,
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            L_enc=L_enc,
            L_dec=L_dec,
            dropout=dropout,
        ).to(self.device)

        if weights_path is not None:
            ckpt = torch.load(weights_path, map_location=self.device)
            state = ckpt.get("state_dict", ckpt)
            self.model.load_state_dict(state)

        self.model.eval()

    @torch.no_grad()
    def predict(self, scene: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Return ego prediction [T_pred, 1, 2]."""
        obs = scene[: self.obs_length].to(self.device)
        pred = self.model(obs)          # [T_pred, N, 2]
        return pred[:, 0:1, :]         # [T_pred, 1, 2]
