"""
ppo_finetune.py
---------------
PPO-style offline RL fine-tuning for trajectory-prediction backbones.

This script treats each scene like a one-step RLHF-style episode:
  1. A pretrained backbone conditions on the observed trajectory.
  2. The policy samples a future ego trajectory from a Gaussian centered on
     the backbone's predicted mean trajectory.
  3. A reward model scores the sampled trajectory with negative FDE and a
     soft collision penalty against ground-truth neighbours.
  4. PPO updates the backbone, the policy variance, and a value head.

Supported backbones
-------------------
  d_pool, social_lstm, autobot, eq_motion, transformer

Example
-------
    python ppo_finetune.py \
        --model autobot \
        --checkpoint checkpoints/autobot_bb.pth \
        --data_root Data \
        --save_path checkpoints/autobot_ppo.pth \
        --device cuda
"""

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
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from dataset import TrajectoryDataset, collate_fn
from metrics import compute_metrics
from models.autobot import AutoBotJoint
from models.d_pool import DPoolLSTM, NN_LSTM
from models.eq_motion import EqMotion
from models.social_lstm import SocialLSTM
from models.transformer import SocialTransformer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_backbone(args) -> tuple[nn.Module, callable]:
    """Create a raw backbone model plus a mean-prediction adapter."""
    obs_len = args.obs_length
    pred_len = args.pred_length

    if args.model == "d_pool":
        pool = NN_LSTM(n=4, hidden_dim=256, out_dim=256)
        model = DPoolLSTM(
            embedding_dim=64,
            hidden_dim=128,
            pool=pool,
            goal_flag=False,
        )

        def predict_mean(model, scene, goal):
            obs = scene[:obs_len]
            batch_split = torch.tensor([0, obs.shape[1]], device=obs.device, dtype=torch.long)
            _, positions = model(obs, goal, batch_split, n_predict=pred_len)
            return positions[-pred_len:, 0:1, :]

    elif args.model == "social_lstm":
        model = SocialLSTM(
            embedding_dim=64,
            hidden_dim=128,
            grid_size=4,
            neighbourhood_size=2.0,
            pool_dim=128,
        )

        def predict_mean(model, scene, goal):
            obs = scene[:obs_len]
            pred = model(obs, n_predict=pred_len)
            return pred[:, 0:1, :]

    elif args.model == "autobot":
        model = AutoBotJoint(
            k_attr=2,
            d_k=128,
            M=5,
            c=1,
            T=pred_len,
            L_enc=2,
            L_dec=2,
            num_heads=8,
            tx_hidden=256,
            dropout=0.1,
        )

        def predict_mean(model, scene, goal):
            obs = scene[:obs_len]
            pred = model(obs)
            return pred[:, 0, 0:1, :]

    elif args.model == "eq_motion":
        model = EqMotion(
            obs_length=obs_len,
            pred_length=pred_len,
            node_dim=64,
            edge_dim=64,
            hidden_dim=64,
            n_layers=4,
        )

        def predict_mean(model, scene, goal):
            obs = scene[:obs_len]
            pred = model(obs.unsqueeze(0))[0]
            return pred[:, 0:1, :]

    elif args.model == "transformer":
        model = SocialTransformer(
            obs_length=obs_len,
            pred_length=pred_len,
            d_model=128,
            num_heads=8,
            ff_dim=256,
            L_enc=3,
            L_dec=2,
            dropout=0.1,
        )

        def predict_mean(model, scene, goal):
            obs = scene[:obs_len]
            pred = model(obs)
            return pred[:, 0:1, :]

    else:
        raise ValueError(f"Unknown model: {args.model!r}")

    return model, predict_mean


def load_backbone_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys when loading checkpoint: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys when loading checkpoint: {unexpected}")


class BackbonePolicy(nn.Module):
    """Policy that samples ego trajectories around a backbone mean prediction."""

    def __init__(
        self,
        backbone: nn.Module,
        predict_mean_fn: callable,
        pred_length: int,
        init_log_std: float,
    ):
        super().__init__()
        self.backbone = backbone
        self.predict_mean_fn = predict_mean_fn
        self.log_std = nn.Parameter(torch.full((pred_length, 1, 2), init_log_std))

    def mean_prediction(self, scene: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.predict_mean_fn(self.backbone, scene, goal)

    def _std(self) -> torch.Tensor:
        return self.log_std.clamp(min=-5.0, max=2.0).exp()

    def log_prob(self, mean: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        std = self._std().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(action).sum()

    def entropy(self, mean: torch.Tensor) -> torch.Tensor:
        std = self._std().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        return dist.entropy().sum()

    def sample_action(
        self,
        scene: torch.Tensor,
        goal: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.mean_prediction(scene, goal)
        if deterministic:
            action = mean
        else:
            std = self._std().expand_as(mean)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
        log_prob = self.log_prob(mean, action)
        entropy = self.entropy(mean)
        return action, log_prob, entropy, mean


def value_feature_dim(obs_length: int, pred_length: int) -> int:
    return obs_length * 2 + pred_length * 2 + 4


def build_value_features(
    scene: torch.Tensor,
    mean_pred: torch.Tensor,
    obs_length: int,
) -> torch.Tensor:
    """Construct a compact scene feature vector for the value head."""
    obs = torch.nan_to_num(scene[:obs_length], nan=0.0)
    ego_obs = obs[:, 0, :].reshape(-1)
    ego_pred = mean_pred[:, 0, :].reshape(-1)

    last_obs = obs[-1]
    if last_obs.shape[0] > 1:
        rel = last_obs[1:] - last_obs[0:1]
        social_mean = rel.mean(dim=0)
        social_min = rel.norm(dim=-1).min().unsqueeze(0)
        neighbour_count = torch.tensor([float(rel.shape[0])], device=scene.device)
    else:
        social_mean = torch.zeros(2, device=scene.device)
        social_min = torch.zeros(1, device=scene.device)
        neighbour_count = torch.zeros(1, device=scene.device)

    return torch.cat([ego_obs, ego_pred, social_mean, social_min, neighbour_count], dim=0)


class ValueHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def final_displacement(action: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    ego_action = action[:, 0, :]
    ego_gt = gt[:, 0, :]
    valid = (~torch.isnan(ego_gt).any(dim=-1)) & torch.isfinite(ego_action).all(dim=-1)
    if not valid.any():
        return action.new_zeros(())

    final_idx = valid.nonzero(as_tuple=False)[-1, 0]
    return torch.norm(ego_action[final_idx] - ego_gt[final_idx], dim=0)


def average_displacement(action: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    ego_action = action[:, 0, :]
    ego_gt = gt[:, 0, :]
    valid = (~torch.isnan(ego_gt).any(dim=-1)) & torch.isfinite(ego_action).all(dim=-1)
    if not valid.any():
        return action.new_zeros(())

    return torch.norm(ego_action[valid] - ego_gt[valid], dim=-1).mean()


def trajectory_smoothness_penalty(
    action: torch.Tensor,
) -> torch.Tensor:
    ego_action = action[:, 0, :]
    valid = torch.isfinite(ego_action).all(dim=-1)
    ego_action = ego_action[valid]
    if ego_action.shape[0] < 3:
        return action.new_zeros(())

    velocity = ego_action[1:] - ego_action[:-1]
    acceleration = velocity[1:] - velocity[:-1]
    return torch.norm(acceleration, dim=-1).mean()


def compute_reward(
    action: torch.Tensor,
    gt: torch.Tensor,
    fde_weight: float,
    ade_weight: float,
    smoothness_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    fde = final_displacement(action, gt)
    ade = average_displacement(action, gt)
    smoothness = trajectory_smoothness_penalty(action)
    reward = -(fde_weight * fde + ade_weight * torch.tanh(ade) + smoothness_weight * smoothness)
    return reward, fde, ade, smoothness


def scene_future(scene: torch.Tensor, obs_length: int, pred_length: int) -> torch.Tensor:
    return scene[obs_length: obs_length + pred_length]


@torch.no_grad()
def collect_rollouts(
    policy: BackbonePolicy,
    value_head: ValueHead,
    loader: DataLoader,
    args,
    device: torch.device,
) -> tuple[dict, dict]:
    rollouts = {
        "scenes": [],
        "goals": [],
        "actions": [],
        "old_log_probs": [],
        "old_values": [],
        "rewards": [],
    }
    reward_vals, fde_vals, ade_vals, smoothness_vals = [], [], [], []

    collected = 0
    for _, scenes, goals in loader:
        for scene, goal in zip(scenes, goals):
            if scene.shape[0] < args.obs_length + args.pred_length:
                continue

            scene = scene.to(device)
            goal = goal.to(device)
            action, log_prob, _, mean = policy.sample_action(scene, goal, deterministic=False)
            value_features = build_value_features(scene, mean, args.obs_length)
            value = value_head(value_features)
            gt = scene_future(scene, args.obs_length, args.pred_length)
            reward, fde, ade, smoothness = compute_reward(
                action,
                gt,
                fde_weight=args.reward_fde_weight,
                ade_weight=args.reward_ade_weight,
                smoothness_weight=args.reward_smoothness_weight,
            )

            rollouts["scenes"].append(scene.detach().cpu())
            rollouts["goals"].append(goal.detach().cpu())
            rollouts["actions"].append(action.detach().cpu())
            rollouts["old_log_probs"].append(log_prob.detach().cpu())
            rollouts["old_values"].append(value.detach().cpu())
            rollouts["rewards"].append(reward.detach().cpu())

            reward_vals.append(reward.item())
            fde_vals.append(fde.item())
            ade_vals.append(ade.item())
            smoothness_vals.append(smoothness.item())
            collected += 1

            if collected >= args.rollout_scenes:
                break
        if collected >= args.rollout_scenes:
            break

    stats = {
        "reward": float(np.mean(reward_vals)) if reward_vals else float("-inf"),
        "fde_reward_term": float(np.mean(fde_vals)) if fde_vals else float("inf"),
        "ade_reward_term": float(np.mean(ade_vals)) if ade_vals else float("inf"),
        "smoothness_reward_term": float(np.mean(smoothness_vals)) if smoothness_vals else 0.0,
        "num_scenes": collected,
    }
    return rollouts, stats


def ppo_update(
    policy: BackbonePolicy,
    value_head: ValueHead,
    optimizer: torch.optim.Optimizer,
    rollouts: dict,
    args,
    device: torch.device,
) -> dict:
    if not rollouts["scenes"]:
        return {"actor_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    old_log_probs = torch.stack(rollouts["old_log_probs"]).to(device)
    old_values = torch.stack(rollouts["old_values"]).to(device)
    returns = torch.stack(rollouts["rewards"]).to(device)
    advantages = returns - old_values
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    actor_losses, value_losses, entropies = [], [], []

    for _ in range(args.ppo_epochs):
        indices = torch.randperm(len(rollouts["scenes"]))
        for start in range(0, len(indices), args.minibatch_size):
            batch_indices = indices[start:start + args.minibatch_size]
            optimizer.zero_grad()

            batch_actor, batch_value, batch_entropy = [], [], []
            for idx in batch_indices.tolist():
                scene = rollouts["scenes"][idx].to(device)
                goal = rollouts["goals"][idx].to(device)
                action = rollouts["actions"][idx].to(device)

                mean = policy.mean_prediction(scene, goal)
                log_prob = policy.log_prob(mean, action)
                entropy = policy.entropy(mean)
                value_features = build_value_features(scene, mean, args.obs_length)
                value = value_head(value_features)

                ratio = torch.exp(log_prob - old_log_probs[idx])
                unclipped = ratio * advantages[idx]
                clipped = torch.clamp(
                    ratio,
                    1.0 - args.clip_epsilon,
                    1.0 + args.clip_epsilon,
                ) * advantages[idx]

                batch_actor.append(-torch.min(unclipped, clipped))
                batch_value.append(F.mse_loss(value, returns[idx]))
                batch_entropy.append(entropy)

            actor_loss = torch.stack(batch_actor).mean()
            value_loss = torch.stack(batch_value).mean()
            entropy_loss = torch.stack(batch_entropy).mean()
            loss = actor_loss + args.value_coef * value_loss - args.entropy_coef * entropy_loss
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(policy.parameters()) + list(value_head.parameters()),
                args.max_grad_norm,
            )
            optimizer.step()

            actor_losses.append(actor_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy_loss.item())

    return {
        "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
        "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
        "entropy": float(np.mean(entropies)) if entropies else 0.0,
    }


@torch.no_grad()
def evaluate_policy(
    policy: BackbonePolicy,
    loader: DataLoader,
    args,
    device: torch.device,
) -> dict:
    rewards, fdes, ades_reward, smoothness_vals = [], [], [], []
    preds, gts = [], []
    seen = 0

    for _, scenes, goals in loader:
        for scene, goal in zip(scenes, goals):
            if scene.shape[0] < args.obs_length + args.pred_length:
                continue

            scene = scene.to(device)
            goal = goal.to(device)
            action, _, _, _ = policy.sample_action(scene, goal, deterministic=True)
            gt = scene_future(scene, args.obs_length, args.pred_length)
            reward, fde, ade_reward, smoothness = compute_reward(
                action,
                gt,
                fde_weight=args.reward_fde_weight,
                ade_weight=args.reward_ade_weight,
                smoothness_weight=args.reward_smoothness_weight,
            )

            rewards.append(reward.item())
            fdes.append(fde.item())
            ades_reward.append(ade_reward.item())
            smoothness_vals.append(smoothness.item())
            preds.append(action.cpu())
            gts.append(gt.cpu())
            seen += 1

            if args.eval_scenes > 0 and seen >= args.eval_scenes:
                break
        if args.eval_scenes > 0 and seen >= args.eval_scenes:
            break

    if not preds:
        return {
            "reward": float("-inf"),
            "fde_reward_term": float("inf"),
            "ade_reward_term": float("inf"),
            "smoothness_reward_term": 0.0,
            "ade": float("inf"),
            "fde": float("inf"),
            "col_rate": 0.0,
        }

    metrics = compute_metrics(preds, gts, collision_threshold=args.collision_threshold)
    return {
        "reward": float(np.mean(rewards)),
        "fde_reward_term": float(np.mean(fdes)),
        "ade_reward_term": float(np.mean(ades_reward)),
        "smoothness_reward_term": float(np.mean(smoothness_vals)),
        "ade": metrics["ade"],
        "fde": metrics["fde"],
        "col_rate": metrics["col"],
    }


def save_checkpoint(
    policy: BackbonePolicy,
    value_head: ValueHead,
    optimizer: torch.optim.Optimizer,
    args,
    save_path: str,
    step: int,
    best_reward: float,
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(
        {
            "step": step,
            "state_dict": policy.backbone.state_dict(),
            "policy_log_std": policy.log_std.detach().cpu(),
            "value_head": value_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_reward": best_reward,
            "args": vars(args),
        },
        save_path,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="PPO fine-tune a trajectory backbone with offline rewards.")

    parser.add_argument("--model", required=True,
                        choices=["d_pool", "social_lstm", "autobot", "eq_motion", "transformer"])
    parser.add_argument("--checkpoint", required=True,
                        help="Path to the supervised pretrained checkpoint.")
    parser.add_argument("--data_root", default="Data")
    parser.add_argument("--save_path", required=True,
                        help="Where to save the PPO-finetuned backbone checkpoint.")
    parser.add_argument("--device", default=None)

    parser.add_argument("--obs_length", type=int, default=9)
    parser.add_argument("--pred_length", type=int, default=12)
    parser.add_argument("--sample", type=float, default=1.0)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Scene batch size for rollout/eval data loading.")

    parser.add_argument("--ppo_iterations", type=int, default=10)
    parser.add_argument("--rollout_scenes", type=int, default=256,
                        help="Number of scenes sampled per PPO iteration.")
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--minibatch_size", type=int, default=32)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=1e-3)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--init_log_std", type=float, default=-1.5,
                        help="Initial policy log-std for trajectory sampling.")

    parser.add_argument("--reward_fde_weight", type=float, default=1.0)
    parser.add_argument("--reward_ade_weight", type=float, default=1.0)
    parser.add_argument("--reward_smoothness_weight", type=float, default=0.2)
    parser.add_argument("--collision_threshold", type=float, default=0.2)
    parser.add_argument("--eval_scenes", type=int, default=256,
                        help="How many validation scenes to score each iteration (-1 = all).")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    set_seed(args.seed)

    print(f"Device: {device}")
    print("Loading dataset …")
    full_dataset = TrajectoryDataset(
        data_root=args.data_root,
        data_part="train",
        sample=args.sample,
    )

    if not 0.0 < args.val_split < 1.0:
        raise ValueError("--val_split must be in the interval (0, 1).")

    val_size = max(1, int(len(full_dataset) * args.val_split))
    if val_size >= len(full_dataset):
        val_size = len(full_dataset) - 1
    train_size = len(full_dataset) - val_size

    split_generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=split_generator,
    )
    print(f"  Train scenes : {len(train_dataset)}")
    print(f"  Val   scenes : {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    backbone, predict_mean_fn = build_backbone(args)
    backbone = backbone.to(device)
    load_backbone_checkpoint(backbone, args.checkpoint, device)

    policy = BackbonePolicy(
        backbone=backbone,
        predict_mean_fn=predict_mean_fn,
        pred_length=args.pred_length,
        init_log_std=args.init_log_std,
    ).to(device)
    value_head = ValueHead(value_feature_dim(args.obs_length, args.pred_length)).to(device)

    optimizer = Adam(
        list(policy.parameters()) + list(value_head.parameters()),
        lr=args.lr,
    )

    print(f"Model: {args.model}")
    print(f"PPO iterations: {args.ppo_iterations}")
    print(
        "Reward: -( "
        f"{args.reward_fde_weight} * FDE + "
        f"{args.reward_ade_weight} * tanh(ADE) + "
        f"{args.reward_smoothness_weight} * smoothness )"
    )

    best_reward = float("-inf")
    for step in range(1, args.ppo_iterations + 1):
        t0 = time.time()
        rollouts, rollout_stats = collect_rollouts(policy, value_head, train_loader, args, device)
        update_stats = ppo_update(policy, value_head, optimizer, rollouts, args, device)
        val_stats = evaluate_policy(policy, val_loader, args, device)
        dt = time.time() - t0

        if step % args.log_interval == 0:
            print(
                f"Iter {step:3d}/{args.ppo_iterations}  "
                f"rollout_reward={rollout_stats['reward']:.4f}  "
                f"val_reward={val_stats['reward']:.4f}  "
                f"val_ADE={val_stats['ade']:.4f}  "
                f"val_FDE={val_stats['fde']:.4f}  "
                f"val_Col={val_stats['col_rate'] * 100:.1f}%  "
                f"actor={update_stats['actor_loss']:.4f}  "
                f"value={update_stats['value_loss']:.4f}  "
                f"entropy={update_stats['entropy']:.4f}  "
                f"({dt:.1f}s)"
            )

        if val_stats["reward"] > best_reward:
            best_reward = val_stats["reward"]
            save_checkpoint(policy, value_head, optimizer, args, args.save_path, step, best_reward)
            print(f"  ✓ Saved best PPO checkpoint (reward={best_reward:.4f}) → {args.save_path}")

    print(f"\nPPO fine-tuning complete. Best validation reward: {best_reward:.4f}")


if __name__ == "__main__":
    main()
