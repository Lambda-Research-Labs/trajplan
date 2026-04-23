"""
validate_online_val.py
----------------------
Validate an online PPO residual policy on the validation split carved from
`Data/train` using the exact same split logic as `train.py`.

Example
-------
    python validate_online_val.py \
        --model transformer \
        --policy_checkpoint checkpoints/transformer_online.pth \
        --prior_checkpoint checkpoints/transformer.pth \
        --data_root Data \
        --device cuda
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import random_split

sys.path.insert(0, str(Path(__file__).parent))

from dataset import TrajectoryDataset
from online_inference import (
    evaluate_online_policy,
    resolve_policy_metadata,
    valid_indices,
)
from online_simulator import FrozenScenePrior, PedPyPriorEnv
from ppo_finetune import build_backbone, load_backbone_checkpoint, set_seed
from train_online_policy import ResidualActorCritic


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate an online PPO model on the train.py validation split."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["d_pool", "social_lstm", "autobot", "eq_motion", "transformer"],
    )
    parser.add_argument("--policy_checkpoint", required=True)
    parser.add_argument("--prior_checkpoint", default=None)
    parser.add_argument("--data_root", default="Data")
    parser.add_argument("--obs_length", type=int, default=9)
    parser.add_argument("--pred_length", type=int, default=12)
    parser.add_argument("--sample", type=float, default=1.0)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
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

    parser.add_argument("--show_progress", action="store_true")
    parser.add_argument("--output_file", default=None, help="Optional TSV output path.")
    return parser.parse_args()


def build_val_dataset(args):
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
    _, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=split_generator,
    )
    return val_dataset


def subset_indices(indices: list[int], max_samples: int, seed: int):
    if 0 < max_samples < len(indices):
        rng = random.Random(seed)
        rng.shuffle(indices)
        indices = indices[:max_samples]
    return indices


def write_results(path: str, results: list[dict]):
    rows = [
        "scene_id\tade\tfde\tcol",
        *[
            f"{row['scene_id']}\t{row['ade']:.6f}\t{row['fde']:.6f}\t{row['col']:.6f}"
            for row in results
        ],
    ]
    Path(path).write_text("\n".join(rows) + "\n")


def main():
    args = parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    policy_checkpoint = resolve_policy_metadata(args)
    set_seed(args.seed)

    print(f"Device     : {device}")
    print(f"Model      : {args.model}")
    print(f"Prior      : {args.prior_checkpoint}")
    print(f"Policy     : {args.policy_checkpoint}")
    print("Split      : train.py validation split from Data/train")

    print("\nLoading validation split …")
    val_dataset = build_val_dataset(args)
    indices = valid_indices(val_dataset, args.obs_length, args.pred_length)
    indices = subset_indices(indices, args.max_samples, args.seed)
    print(f"  Val scenes total : {len(val_dataset)}")
    print(f"  Val scenes valid : {len(indices)}")

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
    env = PedPyPriorEnv(
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

    dummy_scene = val_dataset[indices[0]][1]
    dummy_obs = env.reset(dummy_scene)
    policy = ResidualActorCritic(
        obs_dim=dummy_obs.shape[-1],
        hidden_dim=args.hidden_dim,
        action_dim=2,
    ).to(device)
    state_dict = policy_checkpoint.get("state_dict", policy_checkpoint)
    policy.load_state_dict(state_dict, strict=True)
    policy.eval()

    print("\nRunning validation …")
    results = evaluate_online_policy(policy, env, val_dataset, indices, args, device)
    if not results:
        print("No valid scenes evaluated.")
        return

    ades = np.array([row["ade"] for row in results], dtype=np.float64)
    fdes = np.array([row["fde"] for row in results], dtype=np.float64)
    cols = np.array([row["col"] for row in results], dtype=np.float64)

    print("\n" + "=" * 52)
    print(f"  Model         : online_{args.model}")
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
