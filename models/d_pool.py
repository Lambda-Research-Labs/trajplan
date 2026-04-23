"""
models/d_pool.py
----------------
D-Pool (Distance-Pooling LSTM) model for trajectory prediction.

Architecture
------------
Encoder-decoder LSTM where each agent's hidden state is updated
with an interaction vector produced by NN_LSTM pooling:
  - the n nearest neighbours (by Euclidean distance) are selected;
  - their relative positions + relative velocities are embedded and
    passed through a per-scene LSTM to produce the interaction vector;
  - the interaction vector is concatenated to the input embedding
    before each LSTM cell step.

References
----------
TrajNet++ social force baselines codebase (VITA-EPFL/s-attack, DPool folder).
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Small utility modules
# ---------------------------------------------------------------------------

class InputEmbedding(nn.Module):
    """Embed a 2-D velocity vector, appending a 2-D tag slot (zeros).

    The tag slots are overwritten externally to signal 'start of encode' /
    'start of decode', providing the model a phase signal.
    """

    def __init__(self, input_dim: int = 2, embedding_dim: int = 64, scale: float = 4.0):
        super().__init__()
        self.scale = scale
        # Reserve 2 dims for the tag; learn the rest
        self.linear = nn.Sequential(
            nn.Linear(input_dim, embedding_dim - 2),
            nn.ReLU(),
        )
        self.embedding_dim = embedding_dim

    def forward(self, vel: torch.Tensor) -> torch.Tensor:
        emb = self.linear(vel * self.scale)                     # [N, D-2]
        tag = torch.zeros(emb.size(0), 2, device=emb.device)   # [N, 2]
        return torch.cat([emb, tag], dim=1)                     # [N, D]


class Hidden2Normal(nn.Module):
    """Project hidden state to bivariate-Gaussian parameters (mu_x, mu_y, σ_x, σ_y, ρ)."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 5)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out = self.linear(h)
        out[:, 2] = 0.01 + 0.2 * torch.sigmoid(out[:, 2])   # σ_x  ∈ (0.01, 0.21)
        out[:, 3] = 0.01 + 0.2 * torch.sigmoid(out[:, 3])   # σ_y
        out[:, 4] = 0.70 * torch.sigmoid(out[:, 4])          # ρ    ∈ (0, 0.70)
        return out


# ---------------------------------------------------------------------------
# NN_LSTM pooling module
# ---------------------------------------------------------------------------

class NN_LSTM(nn.Module):
    """Interaction encoder: top-n neighbours → LSTM → interaction vector.

    For each agent at each step:
      1. Find the n closest neighbours (by current Euclidean distance).
      2. Embed (rel_pos ‖ rel_vel) for each of them → [n, 4] → [n, D/n].
      3. Flatten → [D] and step a per-agent LSTM.
      4. Project hidden state → interaction vector [out_dim].

    Parameters
    ----------
    n        : number of nearest neighbours to pool
    hidden_dim : LSTM hidden size  (internal to pooling module)
    out_dim  : size of output interaction vector
    """

    def __init__(self, n: int = 4, hidden_dim: int = 256, out_dim: int = 32):
        super().__init__()
        self.n = n
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        # Embed (rel_pos[2] + rel_vel[2]) for each of the n neighbours
        self.embedding = nn.Sequential(
            nn.Linear(4, out_dim // n),
            nn.ReLU(),
        )
        self.pool_lstm = nn.LSTMCell(out_dim, hidden_dim)
        self.hidden2pool = nn.Linear(hidden_dim, out_dim)

        # Persistent hidden/cell state – reset at the start of each scene
        self._h: list[torch.Tensor] = []
        self._c: list[torch.Tensor] = []
        self.track_mask: torch.Tensor | None = None

    # ------------------------------------------------------------------
    def reset(self, num_agents: int, device: torch.device):
        """Reset the pooling LSTM state for a new scene."""
        self._h = [torch.zeros(self.hidden_dim, device=device) for _ in range(num_agents)]
        self._c = [torch.zeros(self.hidden_dim, device=device) for _ in range(num_agents)]

    # ------------------------------------------------------------------
    def _nearest(
        self,
        xy: torch.Tensor,          # [2]  current agent position
        other_xy: torch.Tensor,    # [M, 2]  other agents' positions
        rel_vel: torch.Tensor,     # [M, 2]  relative velocities
    ) -> torch.Tensor:
        """Return concatenated (rel_pos ‖ rel_vel) for the n nearest neighbours → [4n]."""
        rel_pos = other_xy - xy    # [M, 2]
        M = rel_pos.shape[0]

        if M >= self.n:
            dists = torch.norm(rel_pos, dim=1)
            _, idx = torch.topk(-dists, self.n)
            near_pos = rel_pos[idx]
            near_vel = rel_vel[idx]
        else:
            # Pad with zeros if fewer than n neighbours
            near_pos = torch.zeros(self.n, 2, device=xy.device)
            near_vel = torch.zeros(self.n, 2, device=xy.device)
            near_pos[:M] = rel_pos
            near_vel[:M] = rel_vel

        return torch.cat([near_pos, near_vel], dim=1).view(-1)  # [4n]

    # ------------------------------------------------------------------
    def forward(
        self,
        _hidden: torch.Tensor,  # unused – hidden states live in self._h
        obs1: torch.Tensor,     # [N, 2]  previous positions
        obs2: torch.Tensor,     # [N, 2]  current positions
    ) -> torch.Tensor:
        """Return interaction vector [N, out_dim]."""
        assert self.track_mask is not None, "call reset() before forward()"

        N = obs2.shape[0]

        # If only the primary pedestrian is visible, return zeros
        if not self.track_mask[1:].any():
            return torch.zeros(N, self.out_dim, device=obs1.device)

        # Stack hidden/cell for visible agents
        h_stack = torch.stack([h for m, h in zip(self.track_mask, self._h) if m])
        c_stack = torch.stack([c for m, c in zip(self.track_mask, self._c) if m])

        # Relative velocities:  vel_j - vel_i  for all i,j
        vel = obs2 - obs1                       # [N, 2]
        rel_vel_all = vel.unsqueeze(0) - vel.unsqueeze(1)  # [N, N, 2]

        # Build per-agent nearest-n feature
        eye = torch.eye(N, dtype=torch.bool, device=obs1.device)
        grid = torch.stack([
            self._nearest(
                obs2[i],
                obs2[~eye[i]],          # exclude self
                rel_vel_all[i][~eye[i]],
            )
            for i in range(N)
        ])                              # [N, 4n]

        # Embed each neighbour separately
        grid = grid.view(N, self.n, 4)              # [N, n, 4]
        grid = self.embedding(grid).view(N, -1)     # [N, out_dim]

        # LSTM step
        h_stack, c_stack = self.pool_lstm(grid, (h_stack, c_stack))
        out = self.hidden2pool(h_stack)             # [N, out_dim]

        # Write back hidden states
        mask_idx = [i for i, m in enumerate(self.track_mask) if m]
        for i, h, c in zip(mask_idx, h_stack, c_stack):
            self._h[i] = h
            self._c[i] = c

        return out


# ---------------------------------------------------------------------------
# Main LSTM model
# ---------------------------------------------------------------------------

class DPoolLSTM(nn.Module):
    """Encoder-decoder LSTM with D-Pool interaction pooling.

    Parameters
    ----------
    embedding_dim : int    embedding size for input coordinates
    hidden_dim    : int    LSTM hidden state size
    pool          : NN_LSTM  pooling module (can be None → no interaction)
    goal_flag     : bool   if True, concatenate goal-direction embedding
    goal_dim      : int    embedding size for goal direction
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        pool: NN_LSTM | None = None,
        goal_flag: bool = False,
        goal_dim: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.pool = pool
        self.goal_flag = goal_flag
        self.goal_dim = goal_dim

        self.input_embedding = InputEmbedding(2, embedding_dim, scale=4.0)
        self.goal_embedding = InputEmbedding(2, goal_dim, scale=4.0)

        pool_dim = pool.out_dim if pool is not None else 0
        goal_rep = goal_dim if goal_flag else 0
        lstm_in = embedding_dim + goal_rep + pool_dim

        self.encoder = nn.LSTMCell(lstm_in, hidden_dim)
        self.decoder = nn.LSTMCell(lstm_in, hidden_dim)
        self.hidden2normal = Hidden2Normal(hidden_dim)

    # ------------------------------------------------------------------
    def _step(
        self,
        lstm_cell: nn.LSTMCell,
        hidden_states: list,
        cell_states: list,
        obs1: torch.Tensor,     # [N, 2]
        obs2: torch.Tensor,     # [N, 2]
        goals: torch.Tensor,    # [N, 2]
        batch_split: torch.Tensor,
    ):
        """One encoder or decoder step.

        Returns updated (hidden_states, cell_states, normals [N, 5]).
        """
        N = obs2.shape[0]
        NAN = float("nan")

        # Mask: ignore agents missing from current frame
        present = ~(torch.isnan(obs1[:, 0]) | torch.isnan(obs2[:, 0]))  # [N]

        # Stack hidden/cell for present agents
        h = torch.stack([h for m, h in zip(present, hidden_states) if m])
        c = torch.stack([c for m, c in zip(present, cell_states) if m])

        # Velocity embedding
        vel = (obs2 - obs1)[present]              # [N_present, 2]
        emb = self.input_embedding(vel)           # [N_present, D]

        # Optional goal embedding
        if self.goal_flag:
            diff = obs2[present] - goals[present]
            norms = torch.norm(diff, dim=1, keepdim=True).clamp(min=1e-6)
            goal_dir = diff / norms
            emb = torch.cat([emb, self.goal_embedding(goal_dir)], dim=1)

        # Interaction pooling (per scene in batch)
        if self.pool is not None:
            pool_out_parts = []
            for start, end in zip(batch_split[:-1], batch_split[1:]):
                seg_mask = present[start:end]
                seg_obs1 = obs1[start:end][seg_mask]
                seg_obs2 = obs2[start:end][seg_mask]

                # Provide full-scene track_mask to the pooling module
                full_mask = torch.zeros(N, dtype=torch.bool, device=obs1.device)
                full_mask[start:end] = seg_mask
                self.pool.track_mask = full_mask

                pool_out_parts.append(self.pool(None, seg_obs1, seg_obs2))

            pooled = torch.cat(pool_out_parts, dim=0)   # [N_present, pool_out]
            emb = torch.cat([emb, pooled], dim=1)

        # LSTM step
        h, c = lstm_cell(emb, (h, c))
        normals_present = self.hidden2normal(h)   # [N_present, 5]

        # Unmask: write back hidden states and expand normals
        normals = torch.full((N, 5), NAN, device=obs1.device)
        idx = [i for i, m in enumerate(present) if m]
        for k, (i, h_i, c_i, n_i) in enumerate(zip(idx, h, c, normals_present)):
            hidden_states[i] = h_i
            cell_states[i] = c_i
            normals[i] = n_i

        return hidden_states, cell_states, normals

    # ------------------------------------------------------------------
    def forward(
        self,
        observed: torch.Tensor,              # [T_obs, N, 2]
        goals: torch.Tensor,                 # [N, 2]
        batch_split: torch.Tensor,           # [batch+1] long
        n_predict: int = 12,
    ):
        """Run encoder over obs frames, then decoder for n_predict steps.

        Returns
        -------
        normals   : [T_obs-1 + n_predict, N, 5]  velocity distributions
        positions : [T_obs-1 + n_predict, N, 2]  absolute positions
        """
        N = observed.shape[1]
        device = observed.device

        # Initialise hidden / cell states
        hs = [torch.zeros(self.hidden_dim, device=device) for _ in range(N)]
        cs = [torch.zeros(self.hidden_dim, device=device) for _ in range(N)]

        if self.pool is not None:
            self.pool.reset(N, device=device)

        normals_list = []
        positions = []

        # --- Encoder ---
        for obs1, obs2 in zip(observed[:-1], observed[1:]):
            hs, cs, normal = self._step(self.encoder, hs, cs, obs1, obs2, goals, batch_split)
            normals_list.append(normal)
            positions.append(obs2 + normal[:, :2])   # mean displacement only

        # --- Decoder ---
        prev2 = observed[-2].clone()
        prev1 = observed[-1].clone()

        for _ in range(n_predict):
            hs, cs, normal = self._step(self.decoder, hs, cs, prev2, prev1, goals, batch_split)
            next_pos = prev1 + normal[:, :2]
            normals_list.append(normal)
            positions.append(next_pos.detach())
            prev2 = prev1.detach()
            prev1 = next_pos.detach()

        return torch.stack(normals_list), torch.stack(positions)


# ---------------------------------------------------------------------------
# Wrapper matching the repo's predictor interface
# ---------------------------------------------------------------------------

class DPoolModel:
    """High-level wrapper: load weights, run inference per scene."""

    def __init__(
        self,
        weights_path: str,
        obs_length: int = 9,
        pred_length: int = 12,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        pool_hidden_dim: int = 256,
        pool_dim: int = 256,
        n_neighbours: int = 4,
        goals: bool = False,
        goal_dim: int = 64,
        device: str = "cpu",
    ):
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.device = torch.device(device)

        pool = NN_LSTM(n=n_neighbours, hidden_dim=pool_hidden_dim, out_dim=pool_dim)

        self.model = DPoolLSTM(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            pool=pool,
            goal_flag=goals,
            goal_dim=goal_dim,
        )

        checkpoint = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"], strict=False)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, scene: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Predict ego trajectory for a single scene.

        Parameters
        ----------
        scene : [T, N, 2]  full scene (obs + gt, but only obs is used)
        goal  : [N, 2]

        Returns
        -------
        pred_ego : [T_pred, 1, 2]  predicted ego positions
        """
        obs = scene[: self.obs_length].to(self.device)
        goal = goal.to(self.device)
        batch_split = torch.tensor([0, obs.shape[1]], device=self.device, dtype=torch.long)

        _, positions = self.model(obs, goal, batch_split, n_predict=self.pred_length)

        # Last pred_length frames, ego agent only
        return positions[-self.pred_length :, 0:1, :]
