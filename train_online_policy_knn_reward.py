"""
train_online_policy_knn_reward.py
---------------------------------
Train the same online PPO residual policy as train_online_policy.py, but use
Euclidean nearest-k neighbor distances for the collision penalty term instead
of PedPy-derived Voronoi neighbors.

All other parameters and behavior are kept the same as the original script.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import random_split

sys.path.insert(0, str(Path(__file__).parent))

from dataset import TrajectoryDataset
from online_simulator import FrozenScenePrior, KNNRewardPriorEnv
from ppo_finetune import build_backbone, load_backbone_checkpoint, set_seed
from train_online_policy import (
    ResidualActorCritic,
    collect_rollouts,
    evaluate_policy,
    ppo_update,
    save_checkpoint,
    valid_scene_indices,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an online PPO residual policy with Euclidean kNN collision reward."
    )
    parser.add_argument("--model", required=True,
                        choices=["d_pool", "social_lstm", "autobot", "eq_motion", "transformer"])
    parser.add_argument("--prior_checkpoint", required=True,
                        help="Supervised or PPO-finetuned backbone checkpoint.")
    parser.add_argument("--data_root", default="Data")
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--device", default=None)

    parser.add_argument("--obs_length", type=int, default=9)
    parser.add_argument("--pred_length", type=int, default=12)
    parser.add_argument("--sample", type=float, default=1.0)
    parser.add_argument("--val_split", type=float, default=0.1)

    parser.add_argument("--ppo_iterations", type=int, default=20)
    parser.add_argument("--rollout_episodes", type=int, default=64)
    parser.add_argument("--eval_episodes", type=int, default=32)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--minibatch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=1e-3)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-4)

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dt", type=float, default=0.4)
    parser.add_argument("--max_speed", type=float, default=2.0)
    parser.add_argument("--residual_limit", type=float, default=0.75)
    parser.add_argument("--neighbor_k", type=int, default=4)
    parser.add_argument("--collision_radius", type=float, default=0.3)
    parser.add_argument("--world_margin", type=float, default=2.0)
    parser.add_argument("--reward_progress_weight", type=float, default=1.0)
    parser.add_argument("--reward_fde_weight", type=float, default=1.0)
    parser.add_argument("--reward_collision_weight", type=float, default=5.0)
    parser.add_argument("--max_prior_agents", type=int, default=6)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--show_progress", action="store_true",
                        help="Show per-episode progress bars for rollout and evaluation.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    set_seed(args.seed)

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
    train_scene_indices = valid_scene_indices(train_dataset, args.obs_length, args.pred_length)
    val_scene_indices = valid_scene_indices(val_dataset, args.obs_length, args.pred_length)

    if not train_scene_indices:
        raise RuntimeError("No valid training scenes for the requested observation/prediction lengths.")

    backbone, predict_mean_fn = build_backbone(args)
    backbone = backbone.to(device)
    load_backbone_checkpoint(backbone, args.prior_checkpoint, device)

    prior = FrozenScenePrior(
        backbone=backbone,
        predict_mean_fn=predict_mean_fn,
        obs_length=args.obs_length,
        pred_length=args.pred_length,
        dt=args.dt,
        max_prior_agents=args.max_prior_agents,
        device=str(device),
    )

    env = KNNRewardPriorEnv(
        prior=prior,
        obs_length=args.obs_length,
        pred_length=args.pred_length,
        dt=args.dt,
        max_speed=args.max_speed,
        residual_limit=args.residual_limit,
        neighbor_k=args.neighbor_k,
        collision_radius=args.collision_radius,
        world_margin=args.world_margin,
        progress_weight=args.reward_progress_weight,
        terminal_fde_weight=args.reward_fde_weight,
        collision_weight=args.reward_collision_weight,
        device=str(device),
    )

    dummy_scene = train_dataset[train_scene_indices[0]][1]
    dummy_obs = env.reset(dummy_scene)
    policy = ResidualActorCritic(
        obs_dim=dummy_obs.shape[-1],
        hidden_dim=args.hidden_dim,
        action_dim=2,
    ).to(device)
    optimizer = Adam(policy.parameters(), lr=args.lr)

    print(f"Device: {device}")
    print(f"Model prior: {args.model}")
    print(f"Train scenes: {len(train_scene_indices)} | Val scenes: {len(val_scene_indices)}")
    print("Collision reward: Euclidean nearest-k neighbors")
    print(f"PPO iterations: {args.ppo_iterations}")

    best_reward = float("-inf")
    for step in range(1, args.ppo_iterations + 1):
        t0 = time.time()
        rollouts, rollout_stats = collect_rollouts(
            policy=policy,
            env=env,
            dataset=train_dataset,
            scene_indices=train_scene_indices,
            args=args,
            device=device,
        )
        update_stats = ppo_update(policy, optimizer, rollouts, args, device)
        val_stats = evaluate_policy(policy, env, val_dataset, val_scene_indices, args, device)
        dt = time.time() - t0

        if step % args.log_interval == 0:
            print(
                f"Iter {step:3d}/{args.ppo_iterations}  "
                f"rollout_reward={rollout_stats['reward']:.4f}  "
                f"rollout_FDE={rollout_stats['fde']:.4f}  "
                f"rollout_Col={rollout_stats['collision_rate'] * 100:.1f}%  "
                f"val_reward={val_stats['reward']:.4f}  "
                f"val_FDE={val_stats['fde']:.4f}  "
                f"val_Col={val_stats['collision_rate'] * 100:.1f}%  "
                f"policy={update_stats['policy_loss']:.4f}  "
                f"value={update_stats['value_loss']:.4f}  "
                f"entropy={update_stats['entropy']:.4f}  "
                f"({dt:.1f}s)"
            )

        if val_stats["reward"] > best_reward:
            best_reward = val_stats["reward"]
            save_checkpoint(policy, optimizer, args, args.save_path, step, best_reward)
            print(f"  ✓ Saved best online policy (reward={best_reward:.4f}) → {args.save_path}")

    print(f"\nOnline PPO complete. Best validation reward: {best_reward:.4f}")


if __name__ == "__main__":
    main()
