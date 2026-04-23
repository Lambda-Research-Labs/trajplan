"""
visualize_online_scene.py
-------------------------
Plot a sample scene for an online PPO residual policy using the observed ego
trajectory, predicted ego rollout, and ground-truth ego future.

Example
-------
    python visualize_online_scene.py \
        --model transformer \
        --policy_checkpoint checkpoints/online_policy_smoke.pth \
        --prior_checkpoint checkpoints/transformer.pth \
        --data_root Data \
        --scene_index 0 \
        --output_file outputs/online_transformer_scene.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dataset import TrajectoryDataset
from metrics import ade as calc_ade, fde as calc_fde
from online_inference import (
    parse_args_defaults,
    process_scene_for_metrics,
    resolve_policy_metadata,
    rollout_scene,
    valid_indices,
)
from online_simulator import FrozenScenePrior, PedPyPriorEnv
from ppo_finetune import build_backbone, load_backbone_checkpoint, set_seed
from train_online_policy import ResidualActorCritic


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize one scene for an online PPO residual policy."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["d_pool", "social_lstm", "autobot", "eq_motion", "transformer"],
    )
    parser.add_argument("--policy_checkpoint", required=True)
    parser.add_argument("--prior_checkpoint", default=None)
    parser.add_argument("--data_root", default="Data")
    parser.add_argument(
        "--data_part",
        default="train",
        choices=["train", "test", "secret"],
        help="Use train when you want ground-truth future trajectories.",
    )
    parser.add_argument("--scene_index", type=int, default=None)
    parser.add_argument("--obs_length", type=int, default=9)
    parser.add_argument("--pred_length", type=int, default=12)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sample", type=float, default=1.0)

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
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def select_scene(dataset, scene_index, obs_length, pred_length):
    valid = valid_indices(dataset, obs_length, pred_length)
    if not valid:
        raise RuntimeError(
            "No scenes in this split contain both observed and future frames "
            f"for obs_length={obs_length}, pred_length={pred_length}."
        )
    if scene_index is None:
        return valid[0]
    if scene_index < 0 or scene_index >= len(dataset):
        raise IndexError(
            f"scene_index={scene_index} is out of range for dataset size {len(dataset)}."
        )
    if scene_index not in valid:
        raise ValueError(
            f"scene_index={scene_index} does not have enough frames for "
            f"{obs_length}+{pred_length} steps."
        )
    return scene_index


def resolve_runtime_args(args):
    policy_checkpoint = resolve_policy_metadata(args)
    defaults = parse_args_defaults()
    for name, default in defaults.items():
        if getattr(args, name, default) == default and name in policy_checkpoint.get("args", {}):
            setattr(args, name, policy_checkpoint["args"][name])
    return policy_checkpoint


def _plot_traj(ax, traj, color, linewidth, label, linestyle="-"):
    xy = traj.detach().cpu().numpy()
    ax.plot(
        xy[:, 0],
        xy[:, 1],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
    )
    ax.scatter(xy[-1, 0], xy[-1, 1], color=color, s=24)


def render_scene(scene_id, obs, gt, pred, output_file, show_plot):
    obs = obs.detach().cpu()
    gt = gt.detach().cpu()
    pred = pred.detach().cpu()

    ade_val = calc_ade(pred, gt)
    fde_val = calc_fde(pred, gt)

    fig, ax = plt.subplots(figsize=(8, 8))
    _plot_traj(ax, obs[:, 0, :], color="#2563eb", linewidth=2.6, label="Observed")
    _plot_traj(ax, gt[:, 0, :], color="#16a34a", linewidth=2.6, label="Actual")
    _plot_traj(ax, pred[:, 0, :], color="#dc2626", linewidth=2.6, label="Predicted")
    ax.scatter(obs[0, 0, 0].item(), obs[0, 0, 1].item(), color="#1d4ed8", s=48, marker="o", label="Start")

    ax.set_title(f"Scene {scene_id} | ADE={ade_val:.3f} m | FDE={fde_val:.3f} m")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    ax.axis("equal")
    fig.tight_layout()

    if output_file is not None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    if show_plot:
        plt.show()

    plt.close(fig)


def main():
    args = parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    policy_checkpoint = resolve_runtime_args(args)
    set_seed(args.seed)

    dataset = TrajectoryDataset(
        data_root=args.data_root,
        data_part=args.data_part,
        sample=args.sample,
    )
    scene_index = select_scene(dataset, args.scene_index, args.obs_length, args.pred_length)
    scene_id, scene, _ = dataset[scene_index]

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

    dummy_obs = env.reset(scene)
    policy = ResidualActorCritic(
        obs_dim=dummy_obs.shape[-1],
        hidden_dim=args.hidden_dim,
        action_dim=2,
    ).to(device)
    state_dict = policy_checkpoint.get("state_dict", policy_checkpoint)
    policy.load_state_dict(state_dict, strict=True)
    policy.eval()

    obs, gt = process_scene_for_metrics(scene, args.obs_length, args.pred_length)
    pred = rollout_scene(policy, env, scene, device)[:, 0:1, :]

    output_file = args.output_file
    if output_file is None:
        output_file = f"outputs/online_{args.model}_scene_{scene_id}.png"

    render_scene(
        scene_id=scene_id,
        obs=obs[:, 0:1, :],
        gt=gt[:, 0:1, :],
        pred=pred,
        output_file=output_file,
        show_plot=args.show,
    )


if __name__ == "__main__":
    main()
