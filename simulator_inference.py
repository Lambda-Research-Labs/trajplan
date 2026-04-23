"""
simulator_inference.py
----------------------
Evaluate the PedPy-backed simulator without any learned model by using a
hand-crafted heuristic prior derived only from observed waypoints.

Default behavior is fair for trajectory prediction: goals are estimated from
the observed trajectory only, so future labels are not used to generate the
rollout. An optional oracle-goal mode is also provided for analysis.

Example
-------
    python simulator_inference.py \
        --data_root Data \
        --data_part train \
        --mode social_force \
        --device cpu \
        --show_progress
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from dataset import TrajectoryDataset
from metrics import ade as calc_ade, collision as calc_collision, fde as calc_fde
from online_simulator import (
    HeuristicScenePrior,
    PedPyPriorEnv,
    _extract_goals,
    _filter_active_agents,
    _forward_fill_scene,
)
from ppo_finetune import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a model-free simulator baseline on offline trajectory data."
    )
    parser.add_argument("--data_root", default="Data")
    parser.add_argument(
        "--data_part",
        default="train",
        choices=["train", "test", "secret"],
        help="Use a split with ground-truth future available. Usually 'train'.",
    )
    parser.add_argument("--obs_length", type=int, default=9)
    parser.add_argument("--pred_length", type=int, default=12)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sample", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=-1)

    parser.add_argument(
        "--mode",
        default="constant_velocity",
        choices=["constant_velocity", "goal_directed", "social_force"],
        help="Heuristic simulator prior.",
    )
    parser.add_argument(
        "--goal_mode",
        default="observed_velocity",
        choices=["observed_velocity", "oracle_endpoint"],
        help="How simulator goals are set. 'observed_velocity' is fair; 'oracle_endpoint' peeks at the future.",
    )
    parser.add_argument("--history_window", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.4)
    parser.add_argument("--max_speed", type=float, default=2.0)
    parser.add_argument("--neighbor_k", type=int, default=4)
    parser.add_argument("--collision_radius", type=float, default=0.3)
    parser.add_argument("--world_margin", type=float, default=2.0)
    parser.add_argument("--reward_progress_weight", type=float, default=1.0)
    parser.add_argument("--reward_fde_weight", type=float, default=1.0)
    parser.add_argument("--reward_collision_weight", type=float, default=5.0)
    parser.add_argument("--repulsion_radius", type=float, default=0.8)
    parser.add_argument("--repulsion_weight", type=float, default=0.35)
    parser.add_argument("--goal_weight", type=float, default=0.5)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show_progress", action="store_true")
    parser.add_argument("--output_file", default=None)
    return parser.parse_args()


def valid_indices(dataset, obs_length: int, pred_length: int):
    indices = []
    for idx in range(len(dataset)):
        _, scene, _ = dataset[idx]
        if scene.shape[0] >= obs_length + pred_length:
            indices.append(idx)
    return indices


def estimate_goals_from_observation(
    scene: torch.Tensor,
    obs_length: int,
    pred_length: int,
    dt: float,
    goal_mode: str,
) -> torch.Tensor:
    if goal_mode == "oracle_endpoint":
        return _extract_goals(scene)

    history = scene[:obs_length]
    if history.shape[0] < 2:
        return history[-1].clone()

    window = min(3, history.shape[0] - 1)
    velocity = (history[-window:] - history[-window - 1:-1]).mean(dim=0) / dt
    return history[-1] + velocity * dt * pred_length


@torch.no_grad()
def rollout_scene(env: PedPyPriorEnv, scene: torch.Tensor, goals: torch.Tensor, device: torch.device):
    env.reset(scene)
    env.goals = goals.to(device)
    predictions = []
    done = False

    while not done:
        zero_actions = torch.zeros_like(env.positions)
        _, _, done, _ = env.step(zero_actions)
        predictions.append(env.positions.detach().clone())

    return torch.stack(predictions, dim=0)


@torch.no_grad()
def evaluate(dataset, indices, args, env, device: torch.device):
    if 0 < args.max_samples < len(indices):
        generator = torch.Generator().manual_seed(args.seed)
        order = torch.randperm(len(indices), generator=generator).tolist()
        indices = [indices[i] for i in order[: args.max_samples]]

    results = []
    progress = tqdm(
        indices,
        total=len(indices),
        desc="  eval",
        leave=False,
        disable=not args.show_progress,
    )

    for idx in progress:
        scene_id, raw_scene, _ = dataset[idx]
        scene, _ = _filter_active_agents(raw_scene.to(device), args.obs_length)
        scene = _forward_fill_scene(scene)
        gt = scene[args.obs_length: args.obs_length + args.pred_length]
        goals = estimate_goals_from_observation(
            scene,
            obs_length=args.obs_length,
            pred_length=args.pred_length,
            dt=args.dt,
            goal_mode=args.goal_mode,
        )

        pred_full = rollout_scene(env, scene, goals, device)
        pred_ego = pred_full[:, 0:1, :]
        gt_ego = gt[:, 0:1, :]

        ade_v = calc_ade(pred_ego, gt_ego)
        fde_v = calc_fde(pred_ego, gt_ego)
        col_v = calc_collision(pred_ego, gt, threshold=args.collision_radius)

        results.append({"scene_id": scene_id, "ade": ade_v, "fde": fde_v, "col": col_v})
        progress.set_postfix(
            scenes=len(results),
            ade=f"{np.mean([r['ade'] for r in results]):.3f}",
            fde=f"{np.mean([r['fde'] for r in results]):.3f}",
            col=f"{np.mean([r['col'] for r in results]) * 100:.1f}%",
        )

    progress.close()
    return results


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
    set_seed(args.seed)

    dataset = TrajectoryDataset(
        data_root=args.data_root,
        data_part=args.data_part,
        sample=args.sample,
        device=str(device),
    )
    indices = valid_indices(dataset, args.obs_length, args.pred_length)
    if not indices:
        raise RuntimeError(
            f"No valid scenes found in split '{args.data_part}' for "
            f"obs_length={args.obs_length}, pred_length={args.pred_length}."
        )

    prior = HeuristicScenePrior(
        dt=args.dt,
        pred_length=args.pred_length,
        max_speed=args.max_speed,
        history_window=args.history_window,
        mode=args.mode,
        repulsion_radius=args.repulsion_radius,
        repulsion_weight=args.repulsion_weight,
        goal_weight=args.goal_weight,
    )
    env = PedPyPriorEnv(
        prior=prior,
        obs_length=args.obs_length,
        pred_length=args.pred_length,
        dt=args.dt,
        max_speed=args.max_speed,
        residual_limit=0.0,
        neighbor_k=args.neighbor_k,
        collision_radius=args.collision_radius,
        world_margin=args.world_margin,
        progress_weight=args.reward_progress_weight,
        terminal_fde_weight=args.reward_fde_weight,
        collision_weight=args.reward_collision_weight,
        device=str(device),
    )

    print(f"Device     : {device}")
    print(f"Mode       : {args.mode}")
    print(f"Goal mode  : {args.goal_mode}")
    print(f"Data part  : {args.data_part}")
    print(f"Scenes     : {len(indices) if args.max_samples < 0 else min(len(indices), args.max_samples)}")

    results = evaluate(dataset, indices, args, env, device)
    if not results:
        print("No valid scenes evaluated.")
        return

    ades = np.array([row["ade"] for row in results], dtype=np.float64)
    fdes = np.array([row["fde"] for row in results], dtype=np.float64)
    cols = np.array([row["col"] for row in results], dtype=np.float64)

    print("\n" + "=" * 52)
    print(f"  Baseline      : simulator_{args.mode}")
    print(f"  Goal mode     : {args.goal_mode}")
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
