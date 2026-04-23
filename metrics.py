"""
metrics.py
----------
Standard trajectory-prediction evaluation metrics.

All functions operate on torch.Tensor inputs and return Python scalars
(or tensors for batch variants), making them trivially composable.

Shapes used throughout
-----------------------
pred  : [T_pred, N_agents, 2]   predicted positions (world coords, metres)
gt    : [T_pred, N_agents, 2]   ground-truth future positions
  - axis 0 : future time steps
  - axis 1 : agents  (agent 0 = primary / ego pedestrian)
  - axis 2 : (x, y)
"""

from __future__ import annotations

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Point-displacement helpers
# ---------------------------------------------------------------------------

def _l2_per_step(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Per-step L2 distance for the primary agent (index 0).

    Parameters
    ----------
    pred : [T, N, 2]
    gt   : [T, N, 2]

    Returns
    -------
    dist : [T]  – L2 in metres at each future frame for agent 0
    """
    diff = pred[:, 0, :] - gt[:, 0, :]          # [T, 2]
    return torch.norm(diff, dim=1)               # [T]


# ---------------------------------------------------------------------------
# ADE – Average Displacement Error
# ---------------------------------------------------------------------------

def ade(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Mean L2 error over all predicted time steps for the primary agent.

    Parameters
    ----------
    pred : [T_pred, N, 2]
    gt   : [T_pred, N, 2]  (ground-truth future, same length as pred)

    Returns
    -------
    ade  : float  metres
    """
    # Align lengths in case pred is longer than gt
    t_common = min(pred.shape[0], gt.shape[0])
    return _l2_per_step(pred[:t_common], gt[:t_common]).mean().item()


# ---------------------------------------------------------------------------
# FDE – Final Displacement Error
# ---------------------------------------------------------------------------

def fde(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """L2 error at the *last* predicted time step for the primary agent.

    Parameters
    ----------
    pred : [T_pred, N, 2]
    gt   : [T_pred, N, 2]

    Returns
    -------
    fde  : float  metres
    """
    t_common = min(pred.shape[0], gt.shape[0])
    return _l2_per_step(pred[:t_common], gt[:t_common])[-1].item()


# ---------------------------------------------------------------------------
# Collision metric
# ---------------------------------------------------------------------------

def collision(
    pred: torch.Tensor,
    gt: torch.Tensor,
    threshold: float = 0.2,
) -> int:
    """Check whether the predicted ego trajectory collides with any ground-truth neighbour.

    A collision occurs when the distance between the predicted ego position
    and the ground-truth position of *any* neighbour (agent index >= 1) is
    less than *threshold* metres at *any* future step.

    Parameters
    ----------
    pred      : [T_pred, N, 2]  – prediction for ALL agents (or just ego [T,1,2])
    gt        : [T_pred, N, 2]  – ground-truth future for ALL agents
    threshold : float  metres

    Returns
    -------
    1 if a collision is detected, 0 otherwise.
    """
    if gt.shape[1] < 2:
        # No neighbours in scene – no collision possible
        return 0

    ego_pred = pred[:, 0:1, :]        # [T, 1, 2]
    neighbours_gt = gt[:, 1:, :]      # [T, N-1, 2]

    dists = torch.norm(ego_pred - neighbours_gt, dim=2)   # [T, N-1]
    return int((dists < threshold).any().item())


# ---------------------------------------------------------------------------
# Batch-level aggregation
# ---------------------------------------------------------------------------

def compute_metrics(
    predictions: list[torch.Tensor],
    ground_truths: list[torch.Tensor],
    collision_threshold: float = 0.2,
) -> dict:
    """Aggregate ADE, FDE, and collision rate over a list of scenes.

    Parameters
    ----------
    predictions   : list of [T_pred, N_i, 2] tensors
    ground_truths : list of [T_pred, N_i, 2] tensors
    collision_threshold : float  metres

    Returns
    -------
    dict with keys 'ade', 'fde', 'col'  (col = collision rate in [0, 1])
    """
    ades, fdes, cols = [], [], []
    for pred, gt in zip(predictions, ground_truths):
        ades.append(ade(pred, gt))
        fdes.append(fde(pred, gt))
        cols.append(collision(pred, gt, threshold=collision_threshold))

    return {
        "ade": float(np.mean(ades)),
        "fde": float(np.mean(fdes)),
        "col": float(np.mean(cols)),
    }
