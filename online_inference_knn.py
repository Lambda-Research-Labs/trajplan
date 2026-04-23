"""
online_inference_knn.py
-----------------------
Evaluate an online PPO residual policy on trajectory data using the same
ADE/FDE/collision metrics as offline inference, while matching the
Euclidean nearest-k collision-reward environment used by
`train_online_policy_knn_reward.py`.

The online policy is rolled out autoregressively from the observed frames:
  1. Load the frozen trajectory prior backbone.
  2. Load the trained residual PPO policy checkpoint.
  3. Reset the kNN-reward environment on each dataset scene.
  4. Roll the policy out for pred_length steps.
  5. Compare the predicted future against the dataset ground truth.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from dataset import TrajectoryDataset
from online_inference import (
    evaluate_online_policy,
    resolve_policy_metadata,
    valid_indices,
    write_results,
)
from online_simulator import FrozenScenePrior, KNNRewardPriorEnv
from ppo_finetune import build_backbone, load_backbone_checkpoint, set_seed
from train_online_policy import ResidualActorCritic


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an online PPO residual policy using the kNN-reward environment."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["d_pool", "social_lstm", "autobot", "eq_motion", "transformer"],
    )
    parser.add_argument(
        "--policy_checkpoint",
        required=True,
        help="Path to the online PPO residual policy checkpoint.",
    )
    parser.add_argument(
        "--prior_checkpoint",
        default=None,
        help="Path to the frozen backbone checkpoint. If omitted, try to read it from the policy checkpoint metadata.",
    )
    parser.add_argument("--data_root", default="Data")
    parser.add_argument(
        "--data_part",
        default="train",
        choices=["train", "test", "secret"],
        help="Use a split with future ground truth available. Usually 'train'.",
    )
    parser.add_argument("--obs_length", type=int, default=9)
    parser.add_argument("--pred_length", type=int, default=12)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sample", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=-1)

    parser.add_argument("--hidden_dim", type=int, default=None)
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
    parser.add_argument("--show_progress", action="store_true")
    parser.add_argument("--output_file", default=None, help="Optional TSV output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    policy_checkpoint = resolve_policy_metadata(args)
    set_seed(args.seed)

    dataset = TrajectoryDataset(
        data_root=args.data_root,
        data_part=args.data_part,
        sample=args.sample,
    )
    indices = valid_indices(dataset, args.obs_length, args.pred_length)
    if not indices:
        raise RuntimeError(
            f"No valid scenes found in split '{args.data_part}' for "
            f"obs_length={args.obs_length}, pred_length={args.pred_length}."
        )

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

    dummy_scene = dataset[indices[0]][1]
    dummy_obs = env.reset(dummy_scene)
    policy = ResidualActorCritic(
        obs_dim=dummy_obs.shape[-1],
        hidden_dim=args.hidden_dim,
        action_dim=2,
    ).to(device)
    state_dict = policy_checkpoint.get("state_dict", policy_checkpoint)
    policy.load_state_dict(state_dict, strict=True)
    policy.eval()

    print(f"Device     : {device}")
    print(f"Model      : {args.model}")
    print(f"Prior      : {args.prior_checkpoint}")
    print(f"Policy     : {args.policy_checkpoint}")
    print("Reward env : Euclidean nearest-k collision penalty")
    print(f"Data part  : {args.data_part}")
    print(f"Scenes     : {len(indices) if args.max_samples < 0 else min(len(indices), args.max_samples)}")

    results = evaluate_online_policy(policy, env, dataset, indices, args, device)
    if not results:
        print("No valid scenes evaluated.")
        return

    ades = np.array([row["ade"] for row in results], dtype=np.float64)
    fdes = np.array([row["fde"] for row in results], dtype=np.float64)
    cols = np.array([row["col"] for row in results], dtype=np.float64)

    print("\n" + "=" * 52)
    print(f"  Model         : online_{args.model}_knn")
    print(f"  Scenes eval'd : {len(results)}")
    print(f"  ADE   (↓)     : {ades.mean():.4f} m")
    print(f"  FDE   (↓)     : {fdes.mean():.4f} m")
    print(f"  Col.  (↓)     : {cols.mean() * 100:.2f} %")
    print("=" * 52)

    if args.output_file:
        write_results(args.output_file, results)
        print(f"\nResults saved → {args.output_file}")


if __name__ == "__main__":
    main()
