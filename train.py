from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from dataset import TrajectoryDataset, collate_fn
from metrics import compute_metrics


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def l2_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mean L2 loss over all time steps for the primary agent.

    Parameters
    ----------
    pred : [T_pred, N, 2]   predicted positions
    gt   : [T_pred, N, 2]   ground-truth future positions

    Returns
    -------
    loss : scalar
    """
    ego_pred = pred[:, 0, :]   # [T_pred, 2]
    ego_gt   = gt[:, 0, :]     # [T_pred, 2]
    return torch.norm(ego_pred - ego_gt, dim=1).mean()


def fde_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Final displacement error loss for the primary agent."""
    ego_pred = pred[:, 0, :]
    ego_gt = gt[:, 0, :]
    valid = (~torch.isnan(ego_gt).any(dim=-1)) & torch.isfinite(ego_pred).all(dim=-1)
    if not valid.any():
        return pred.new_zeros(())

    final_idx = valid.nonzero(as_tuple=False)[-1, 0]
    return torch.norm(ego_pred[final_idx] - ego_gt[final_idx], dim=0)


def ego_nll_loss(params: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood for the primary agent only."""
    ego_params = params[:, 0:1, :]
    ego_gt = gt[:, 0:1, :]
    return masked_nll_loss(ego_params, ego_gt)


def masked_l2_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mean L2 loss over all valid agent positions."""
    mask = (~torch.isnan(gt).any(dim=-1)) & torch.isfinite(pred).all(dim=-1)
    if not mask.any():
        return pred.new_zeros(())

    gt_safe = torch.nan_to_num(gt, nan=0.0)
    diff = torch.norm(pred - gt_safe, dim=-1)
    return diff[mask].mean()


def nll_loss(params: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood under a bivariate Gaussian.

    Parameters
    ----------
    params : [T_pred, N, 5]  (μ_x, μ_y, σ_x, σ_y, ρ) per step per agent
    gt     : [T_pred, N, 2]  ground-truth positions

    Returns
    -------
    loss : scalar
    """
    mu_x  = params[:, :, 0]
    mu_y  = params[:, :, 1]
    sig_x = params[:, :, 2].clamp(min=1e-4)
    sig_y = params[:, :, 3].clamp(min=1e-4)
    rho   = params[:, :, 4].clamp(-0.99, 0.99)

    dx = (gt[:, :, 0] - mu_x) / sig_x
    dy = (gt[:, :, 1] - mu_y) / sig_y

    z = dx**2 + dy**2 - 2 * rho * dx * dy
    denom = 1 - rho**2

    nll = (
        0.5 * z / denom
        + torch.log(sig_x)
        + torch.log(sig_y)
        + 0.5 * torch.log(denom)
        + torch.log(torch.tensor(2 * 3.14159265, device=params.device))
    )
    return nll.mean()


def masked_nll_loss(params: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood under a bivariate Gaussian for valid targets only."""
    mask = (~torch.isnan(gt).any(dim=-1)) & torch.isfinite(params).all(dim=-1)
    if not mask.any():
        return params.new_zeros(())

    gt_safe = torch.nan_to_num(gt, nan=0.0)
    mu_x  = params[:, :, 0]
    mu_y  = params[:, :, 1]
    sig_x = params[:, :, 2].clamp(min=1e-4)
    sig_y = params[:, :, 3].clamp(min=1e-4)
    rho   = params[:, :, 4].clamp(-0.99, 0.99)

    dx = (gt_safe[:, :, 0] - mu_x) / sig_x
    dy = (gt_safe[:, :, 1] - mu_y) / sig_y

    z = dx**2 + dy**2 - 2 * rho * dx * dy
    denom = 1 - rho**2

    nll = (
        0.5 * z / denom
        + torch.log(sig_x)
        + torch.log(sig_y)
        + 0.5 * torch.log(denom)
        + torch.log(torch.tensor(2 * 3.14159265, device=params.device))
    )
    return nll[mask].mean()


def collision_penalty_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    threshold: float = 0.2,
) -> torch.Tensor:
    """Soft collision penalty between predicted ego and ground-truth neighbours."""
    if gt.shape[1] < 2:
        return pred.new_zeros(())

    ego_pred = pred[:, 0:1, :]
    neighbours_gt = gt[:, 1:, :]
    mask = (~torch.isnan(neighbours_gt).any(dim=-1)) & torch.isfinite(ego_pred).all(dim=-1)
    if not mask.any():
        return pred.new_zeros(())

    dists = torch.norm(ego_pred - torch.nan_to_num(neighbours_gt, nan=0.0), dim=2)
    penalties = torch.relu(threshold - dists)
    return penalties[mask].mean()


# ---------------------------------------------------------------------------
# Build model, optimiser, and select a forward pass adaptor
# ---------------------------------------------------------------------------

def build_model(args) -> tuple[nn.Module, callable]:
    """Return (nn.Module, forward_fn).

    forward_fn(model, scene, goal, obs_len, pred_len)
        → (pred [T_pred, N, 2],  loss_tensor)
    """
    obs_len  = args.obs_length
    pred_len = args.pred_length

    # ------------------------------------------------------------------ D-Pool
    if args.model == "d_pool":
        from models.d_pool import DPoolLSTM, NN_LSTM

        pool = NN_LSTM(n=4, hidden_dim=256, out_dim=256)
        model = DPoolLSTM(
            embedding_dim=64, hidden_dim=128, pool=pool,
            goal_flag=False,
        )

        def forward_fn(model, scene, goal, obs_len, pred_len):
            obs = scene[:obs_len]
            gt  = scene[obs_len:obs_len + pred_len]
            N = obs.shape[1]
            batch_split = torch.tensor([0, N], device=obs.device, dtype=torch.long)
            normals, positions = model(obs, goal, batch_split, n_predict=pred_len)
            pred = positions[-pred_len:]
            params = normals[-pred_len:]
            loss = (
                args.dpool_nll_weight * ego_nll_loss(params, gt)
                + args.dpool_l2_weight * l2_loss(pred, gt)
                + args.fde_weight * fde_loss(pred, gt)
                + args.dpool_collision_weight * collision_penalty_loss(
                    pred, gt, threshold=args.collision_threshold
                )
            )
            return pred, loss

    # ---------------------------------------------------------------- Social LSTM
    elif args.model == "social_lstm":
        from models.social_lstm import SocialLSTM

        model = SocialLSTM(
            embedding_dim=64, hidden_dim=128,
            grid_size=4, neighbourhood_size=2.0, pool_dim=128,
        )

        def forward_fn(model, scene, goal, obs_len, pred_len):
            obs  = scene[:obs_len]
            gt   = scene[obs_len:obs_len + pred_len]
            pred = model(obs, n_predict=pred_len)   # [T_pred, N, 2]
            loss = l2_loss(pred, gt) + args.fde_weight * fde_loss(pred, gt)
            return pred, loss

    # ------------------------------------------------------------------ AutoBot
    elif args.model == "autobot":
        from models.autobot import AutoBotJoint

        model = AutoBotJoint(
            k_attr=2, d_k=128, M=5, c=1, T=pred_len,
            L_enc=2, L_dec=2, num_heads=8, tx_hidden=256, dropout=0.1,
        )

        def forward_fn(model, scene, goal, obs_len, pred_len):
            obs = scene[:obs_len]
            gt  = scene[obs_len:obs_len + pred_len]
            # pred: [T_pred, c, M+1, 2]
            pred_all = model(obs)
            pred = pred_all[:, 0, :, :]                # mode 0  [T_pred, M+1, 2]
            # Align N dims
            n_common = min(pred.shape[1], gt.shape[1])
            pred_common = pred[:, :n_common]
            gt_common = gt[:, :n_common]
            loss = l2_loss(pred_common, gt_common) + args.fde_weight * fde_loss(pred_common, gt_common)
            return pred[:, 0:1, :], loss               # return ego only

    # ---------------------------------------------------------------- EqMotion
    elif args.model == "eq_motion":
        from models.eq_motion import EqMotion

        model = EqMotion(
            obs_length=obs_len, pred_length=pred_len,
            node_dim=64, edge_dim=64, hidden_dim=64, n_layers=4,
        )

        def forward_fn(model, scene, goal, obs_len, pred_len):
            obs = scene[:obs_len]       # [T_obs, N, 2]
            gt  = scene[obs_len:obs_len + pred_len]

            # EqMotion expects [B, T_obs, N, 2]
            obs_in = obs.unsqueeze(0)
            pred = model(obs_in)[0]     # [T_pred, N, 2]

            n_common = min(pred.shape[1], gt.shape[1])
            pred_common = pred[:, :n_common]
            gt_common = gt[:, :n_common]
            loss = l2_loss(pred_common, gt_common) + args.fde_weight * fde_loss(pred_common, gt_common)
            return pred[:, 0:1, :], loss

    # -------------------------------------------------------------- Transformer
    elif args.model == "transformer":
        from models.transformer import SocialTransformer

        model = SocialTransformer(
            obs_length=obs_len, pred_length=pred_len,
            d_model=128, num_heads=8, ff_dim=256,
            L_enc=3, L_dec=2, dropout=0.1,
        )

        def forward_fn(model, scene, goal, obs_len, pred_len):
            obs  = scene[:obs_len]
            gt   = scene[obs_len:obs_len + pred_len]
            pred = model(obs)           # [T_pred, N, 2]
            n_common = min(pred.shape[1], gt.shape[1])
            pred_common = pred[:, :n_common]
            gt_common = gt[:, :n_common]
            loss = l2_loss(pred_common, gt_common) + args.fde_weight * fde_loss(pred_common, gt_common)
            return pred[:, 0:1, :], loss

    else:
        raise ValueError(f"Unknown model: {args.model!r}")

    return model, forward_fn


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    forward_fn: callable,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    obs_len: int,
    pred_len: int,
    device: torch.device,
    clip_grad: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for _, scenes, goals in tqdm(loader, desc="  train", leave=False):
        optimiser.zero_grad()
        scene_losses = []

        for scene, goal in zip(scenes, goals):
            scene = scene.to(device)
            goal  = goal.to(device)

            # Skip scenes that are too short
            if scene.shape[0] < obs_len + pred_len:
                continue

            _, loss = forward_fn(model, scene, goal, obs_len, pred_len)
            scene_losses.append(loss)

        if not scene_losses:
            continue

        batch_loss = torch.stack(scene_losses).mean()
        batch_loss.backward()
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimiser.step()

        total_loss += batch_loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    forward_fn: callable,
    loader: DataLoader,
    obs_len: int,
    pred_len: int,
    device: torch.device,
    collision_threshold: float = 0.2,
) -> dict:
    model.eval()
    preds, gts = [], []

    for _, scenes, goals in tqdm(loader, desc="  eval ", leave=False):
        for scene, goal in zip(scenes, goals):
            scene = scene.to(device)
            goal  = goal.to(device)

            if scene.shape[0] < obs_len + pred_len:
                continue

            pred, _ = forward_fn(model, scene, goal, obs_len, pred_len)
            gt = scene[obs_len:obs_len + pred_len]

            preds.append(pred.cpu())
            gts.append(gt.cpu())

    if not preds:
        return {"ade": float("inf"), "fde": float("inf"), "col": 0.0}

    return compute_metrics(preds, gts, collision_threshold=collision_threshold)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train a trajectory prediction model.")

    # Data
    parser.add_argument("--data_root",   default="Data",
                        help="Path to TrajNet++ data root")
    parser.add_argument("--sample",      type=float, default=1.0,
                        help="Fraction of scenes to load (1.0 = all)")
    parser.add_argument("--val_split",   type=float, default=0.1,
                        help="Fraction of the training set reserved for validation")

    # Model
    parser.add_argument("--model",       default="transformer",
                        choices=["d_pool", "social_lstm", "autobot",
                                 "eq_motion", "transformer"])
    parser.add_argument("--obs_length",  type=int, default=9)
    parser.add_argument("--pred_length", type=int, default=12)

    # Training
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--weight_decay",type=float, default=5e-4)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--clip_grad",   type=float, default=1.0)
    parser.add_argument("--patience",    type=int,   default=50,
                        help="Early-stopping patience (epochs without improvement)")
    parser.add_argument("--scheduler_patience", type=int, default=2,
                        help="ReduceLROnPlateau patience")
    parser.add_argument("--scheduler_factor", type=float, default=0.5,
                        help="ReduceLROnPlateau decay factor")
    parser.add_argument("--collision_threshold", type=float, default=0.2,
                        help="Collision distance threshold in metres")
    parser.add_argument("--dpool_nll_weight", type=float, default=0.2,
                        help="Weight for D-Pool Gaussian NLL loss")
    parser.add_argument("--dpool_l2_weight", type=float, default=1.0,
                        help="Weight for D-Pool L2 loss")
    parser.add_argument("--fde_weight", type=float, default=0.5,
                        help="Weight for final displacement error loss")
    parser.add_argument("--dpool_collision_weight", type=float, default=0.05,
                        help="Weight for D-Pool collision penalty")

    # Misc
    parser.add_argument("--device",      default=None,
                        help="e.g. cuda:0 or cpu (auto-detected if omitted)")
    parser.add_argument("--save_path",   default="checkpoints/model.pth",
                        help="Where to save the best model checkpoint")
    parser.add_argument("--log_interval",type=int,   default=1,
                        help="Print metrics every N epochs")
    parser.add_argument("--seed",        type=int,   default=42,
                        help="Random seed used for the train/validation split")

    args = parser.parse_args()

    if (
        args.dpool_nll_weight < 0
        or args.dpool_l2_weight < 0
        or args.fde_weight < 0
        or args.dpool_collision_weight < 0
    ):
        raise ValueError("Loss weights must be non-negative.")
    if args.collision_threshold <= 0:
        raise ValueError("--collision_threshold must be positive.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Datasets
    print("Loading datasets …")
    full_train_dataset = TrajectoryDataset(
        data_root=args.data_root,
        data_part="train",
        sample=args.sample,
    )
    if not 0.0 < args.val_split < 1.0:
        raise ValueError("--val_split must be in the interval (0, 1).")

    val_size = max(1, int(len(full_train_dataset) * args.val_split))
    if val_size >= len(full_train_dataset):
        val_size = len(full_train_dataset) - 1
    train_size = len(full_train_dataset) - val_size

    split_generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=split_generator,
    )
    print(f"  Train scenes : {len(train_dataset)}")
    print(f"  Val   scenes : {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True,  collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn,
    )

    # Model
    model, forward_fn = build_model(args)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}  |  parameters: {n_params:,}")

    # Optimiser + scheduler
    optimiser = Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )

    # Checkpoint dir
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    # Training loop
    best_ade = float("inf")
    patience_cnt = 0
    history = []

    print(f"\nStarting training for {args.epochs} epochs …\n")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, forward_fn, train_loader,
            optimiser, args.obs_length, args.pred_length,
            device, args.clip_grad,
        )
        val_metrics = evaluate(
            model, forward_fn, val_loader,
            args.obs_length, args.pred_length, device,
            collision_threshold=args.collision_threshold,
        )

        scheduler.step(val_metrics["ade"])

        ade_val = val_metrics["ade"]
        fde_val = val_metrics["fde"]
        col_val = val_metrics["col"]
        dt = time.time() - t0

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "ade": ade_val, "fde": fde_val, "col": col_val,
        })

        if epoch % args.log_interval == 0:
            print(
                f"Epoch {epoch:3d}/{args.epochs}  "
                f"loss={train_loss:.4f}  "
                f"ADE={ade_val:.4f}  FDE={fde_val:.4f}  "
                f"Col={col_val * 100:.1f}%  "
                f"({dt:.1f}s)"
            )

        # Save best
        if ade_val < best_ade:
            best_ade = ade_val
            patience_cnt = 0
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimiser": optimiser.state_dict(),
                    "ade": best_ade,
                    "args": vars(args),
                },
                args.save_path,
            )
            print(f"  ✓ Saved best model (ADE={best_ade:.4f}) → {args.save_path}")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"\nEarly stopping after {epoch} epochs (no improvement for {args.patience} epochs).")
                break

    print(f"\nTraining complete. Best validation ADE: {best_ade:.4f}")
    return history


if __name__ == "__main__":
    main()
