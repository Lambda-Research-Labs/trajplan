"""
models/social_lstm.py
---------------------
Social LSTM (Alahi et al., CVPR 2016) for pedestrian trajectory prediction.

Each pedestrian has its own LSTM. Interaction is captured via an occupancy
grid pooling layer that shares hidden states across spatially nearby agents.

Architecture
------------
  1. Encode observed positions (T_obs frames) with per-agent LSTMs,
     pooling neighbours' hidden states at each step via a social pooling map.
  2. Decode future positions (T_pred frames) autoregressively.

Social pooling
  - A flat grid of size (grid_size × grid_size) cells, each of width
    neighbourhood_size / grid_size metres, is centred on each agent.
  - Neighbour hidden states that fall inside a cell are max-pooled.
  - The pooled grid is flattened and embedded → social vector.
  - The social vector is concatenated to the position embedding before
    the LSTM cell.

Reference: "Social LSTM: Human Trajectory Prediction in Crowded Spaces"
           Alahi, Goel, Ramanathan, Robicquet, Fei-Fei, Savarese, CVPR 2016
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Social pooling occupancy grid
# ---------------------------------------------------------------------------

class SocialPooling(nn.Module):
    """Grid-based social pooling of neighbour hidden states.

    For each agent i:
      - Build a (grid_size × grid_size) grid centred on agent i's position.
      - For every other agent j, find which cell their position falls into.
      - Max-pool agent j's hidden state into that cell.
      - Embed the resulting flattened grid → social vector [pool_dim].
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        grid_size: int = 4,
        neighbourhood_size: float = 2.0,
        pool_dim: int = 128,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.neighbourhood_size = neighbourhood_size
        self.pool_dim = pool_dim

        self.embedding = nn.Sequential(
            nn.Linear(grid_size * grid_size * hidden_dim, pool_dim),
            nn.ReLU(),
        )

    @property
    def out_dim(self) -> int:
        return self.pool_dim

    def forward(
        self,
        pos: torch.Tensor,          # [N, 2]  current positions
        hidden: torch.Tensor,       # [N, H]  agent hidden states
    ) -> torch.Tensor:              # [N, pool_dim]
        N = pos.shape[0]
        G = self.grid_size
        H = self.hidden_dim
        cell_size = self.neighbourhood_size / G
        device = pos.device
        zero_hidden = hidden.new_zeros(H)
        pooled_grids = []

        for i in range(N):
            cell_hidden: dict[tuple[int, int], list[torch.Tensor]] = {}
            if not torch.isfinite(pos[i]).all():
                pooled_grids.append(zero_hidden.repeat(G * G).view(G * G, H))
                continue
            for j in range(N):
                if i == j:
                    continue
                if not torch.isfinite(pos[j]).all():
                    continue
                if not torch.isfinite(hidden[j]).all():
                    continue
                rel = pos[j] - pos[i]       # [2]
                if not torch.isfinite(rel).all():
                    continue
                # Check inside neighbourhood
                if (rel.abs() > self.neighbourhood_size / 2).any():
                    continue
                gx = int((rel[0] + self.neighbourhood_size / 2) / cell_size)
                gy = int((rel[1] + self.neighbourhood_size / 2) / cell_size)
                gx = max(0, min(gx, G - 1))
                gy = max(0, min(gy, G - 1))
                cell_hidden.setdefault((gx, gy), []).append(hidden[j])

            pooled_cells = []
            for gx in range(G):
                for gy in range(G):
                    values = cell_hidden.get((gx, gy))
                    if values:
                        pooled_cells.append(torch.stack(values, dim=0).max(dim=0).values)
                    else:
                        pooled_cells.append(zero_hidden)
            pooled_grids.append(torch.stack(pooled_cells, dim=0))

        grid_flat = torch.stack(pooled_grids, dim=0).reshape(N, -1)  # [N, G*G*H]
        return self.embedding(grid_flat)    # [N, pool_dim]


# ---------------------------------------------------------------------------
# Social LSTM model
# ---------------------------------------------------------------------------

class SocialLSTM(nn.Module):
    """Social LSTM encoder-decoder.

    Parameters
    ----------
    embedding_dim      : int   input coordinate embedding size
    hidden_dim         : int   LSTM hidden size
    grid_size          : int   social pooling grid cells per side
    neighbourhood_size : float social pooling radius (metres)
    pool_dim           : int   social vector embedding size
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        grid_size: int = 4,
        neighbourhood_size: float = 2.0,
        pool_dim: int = 128,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.pooling = SocialPooling(
            hidden_dim=hidden_dim,
            grid_size=grid_size,
            neighbourhood_size=neighbourhood_size,
            pool_dim=pool_dim,
        )

        # Input: embedded velocity + social vector
        lstm_in = embedding_dim + pool_dim
        self.encoder = nn.LSTMCell(lstm_in, hidden_dim)
        self.decoder = nn.LSTMCell(lstm_in, hidden_dim)

        self.pos_embedding = nn.Sequential(
            nn.Linear(2, embedding_dim),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(hidden_dim, 2)   # predict Δx, Δy

    def _impute_observed(self, observed: torch.Tensor) -> torch.Tensor:
        """Backfill leading gaps, then forward-fill later missing positions."""
        obs = observed.clone()
        T, N, _ = obs.shape

        for n in range(N):
            finite = torch.isfinite(obs[:, n]).all(dim=1)
            if not finite.any():
                obs[:, n] = 0.0
                continue

            first_valid = finite.nonzero(as_tuple=False)[0, 0].item()
            obs[:first_valid, n] = obs[first_valid, n]

            last = obs[first_valid, n]
            for t in range(first_valid + 1, T):
                if finite[t]:
                    last = obs[t, n]
                else:
                    obs[t, n] = last

        return obs

    # ------------------------------------------------------------------
    def _step(
        self,
        lstm_cell: nn.LSTMCell,
        hs: torch.Tensor,           # [N, H]
        cs: torch.Tensor,           # [N, H]
        pos: torch.Tensor,          # [N, 2]  current position
        prev_pos: torch.Tensor,     # [N, 2]  previous position
    ):
        vel = pos - prev_pos                    # [N, 2]
        emb = self.pos_embedding(vel)           # [N, D]
        social = self.pooling(pos, hs)          # [N, pool_dim]
        x = torch.cat([emb, social], dim=1)    # [N, D+pool_dim]
        hs_new, cs_new = lstm_cell(x, (hs, cs))
        delta = self.output_layer(hs_new)       # [N, 2]
        next_pos = pos + delta
        return hs_new, cs_new, next_pos

    # ------------------------------------------------------------------
    def forward(
        self,
        observed: torch.Tensor,     # [T_obs, N, 2]
        n_predict: int = 12,
    ):
        """Encode observed trajectory, decode future positions.

        Missing observations are imputed per agent by backfilling the first
        valid position and then forward-filling later gaps.

        Returns
        -------
        pred_positions : [n_predict, N, 2]
        """
        T, N, _ = observed.shape
        device = observed.device

        # Fill missing observations per agent so encoder/pooling always sees
        # finite positions, even when a neighbour first appears after frame 0.
        obs = self._impute_observed(observed)

        # Initialise hidden / cell states
        hs = torch.zeros(N, self.hidden_dim, device=device)
        cs = torch.zeros(N, self.hidden_dim, device=device)

        # --- Encoder ---
        for t in range(1, T):
            hs, cs, _ = self._step(self.encoder, hs, cs, obs[t], obs[t - 1])

        # --- Decoder ---
        pred_positions = []
        prev = obs[-1]
        curr = obs[-1]

        for _ in range(n_predict):
            hs, cs, next_pos = self._step(self.decoder, hs, cs, curr, prev)
            pred_positions.append(next_pos)
            prev = curr.detach()
            curr = next_pos.detach()

        return torch.stack(pred_positions)      # [n_predict, N, 2]


# ---------------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------------

class SocialLSTMModel:
    """Load weights and run inference per scene."""

    def __init__(
        self,
        weights_path: str,
        obs_length: int = 9,
        pred_length: int = 12,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        grid_size: int = 4,
        neighbourhood_size: float = 2.0,
        pool_dim: int = 128,
        device: str = "cpu",
    ):
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.device = torch.device(device)

        self.model = SocialLSTM(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            grid_size=grid_size,
            neighbourhood_size=neighbourhood_size,
            pool_dim=pool_dim,
        ).to(self.device)

        ckpt = torch.load(weights_path, map_location=self.device)
        state = ckpt.get("state_dict", ckpt)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def predict(self, scene: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Return ego prediction [T_pred, 1, 2]."""
        obs = scene[: self.obs_length].to(self.device)
        pred = self.model(obs, n_predict=self.pred_length)   # [T_pred, N, 2]
        return pred[:, 0:1, :]
