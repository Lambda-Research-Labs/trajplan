"""
visualize_scene.py
------------------
Plot a sample trajectory scene with observed, predicted, and ground-truth
future trajectories.

Example
-------
    python visualize_scene.py \
        --model d_pool \
        --weights checkpoints/d_pool_bb.pth \
        --data_root Data \
        --scene_index 0 \
        --output_file outputs/sample_scene.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dataset import TrajectoryDataset
from metrics import ade as calc_ade, fde as calc_fde, collision as calc_collision
from models import get_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize one trajectory scene with observed, predicted, and actual futures."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["d_pool", "social_lstm", "autobot", "eq_motion", "transformer"],
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to model weights (.pth / .state file).",
    )
    parser.add_argument(
        "--data_root",
        default="Data",
        help="Path to the dataset root.",
    )
    parser.add_argument(
        "--data_part",
        default="train",
        choices=["train", "test", "secret"],
        help="Dataset split to visualize. Use train when you want ground truth futures.",
    )
    parser.add_argument("--scene_index", type=int, default=None)
    parser.add_argument(
        "--obs_length",
        type=int,
        default=9,
    )
    parser.add_argument(
        "--pred_length",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--device",
        default=None,
        help="e.g. cuda:0 or cpu (auto-detected if omitted)",
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="PNG path to save the rendered figure.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also open an interactive matplotlib window.",
    )
    return parser.parse_args()


def build_model(args):
    common = dict(
        weights_path=args.weights,
        obs_length=args.obs_length,
        pred_length=args.pred_length,
        device=args.device,
    )

    if args.model == "d_pool":
        return get_model(
            args.model,
            n_neighbours=4,
            embedding_dim=64,
            hidden_dim=128,
            pool_hidden_dim=256,
            pool_dim=256,
            **common,
        )
    if args.model == "social_lstm":
        return get_model(
            args.model,
            embedding_dim=64,
            hidden_dim=128,
            grid_size=4,
            neighbourhood_size=2.0,
            pool_dim=128,
            **common,
        )
    if args.model == "autobot":
        return get_model(
            args.model,
            d_k=128,
            M=5,
            c=1,
            L_enc=2,
            L_dec=2,
            num_heads=8,
            tx_hidden=256,
            dropout=0.1,
            **common,
        )
    if args.model == "eq_motion":
        return get_model(
            args.model,
            node_dim=64,
            edge_dim=64,
            hidden_dim=64,
            n_layers=4,
            **common,
        )
    return get_model(
        args.model,
        d_model=128,
        num_heads=8,
        ff_dim=256,
        L_enc=3,
        L_dec=2,
        dropout=0.0,
        **common,
    )


def select_scene(dataset, scene_index, obs_length, pred_length):
    valid_indices = [
        idx
        for idx in range(len(dataset))
        if dataset[idx][1].shape[0] >= obs_length + pred_length
    ]
    if not valid_indices:
        raise RuntimeError(
            "No scenes in this split contain both observed and future frames "
            f"for obs_length={obs_length}, pred_length={pred_length}."
        )

    if scene_index is None:
        return valid_indices[0]

    if scene_index < 0 or scene_index >= len(dataset):
        raise IndexError(
            f"scene_index={scene_index} is out of range for dataset size {len(dataset)}."
        )

    if scene_index not in valid_indices:
        raise ValueError(
            f"scene_index={scene_index} does not have enough frames for "
            f"{obs_length}+{pred_length} steps."
        )

    return scene_index


def _plot_agent(ax, traj, color, alpha, linewidth, label=None, linestyle="-"):
    if torch.isnan(traj).any():
        mask = ~torch.isnan(traj[:, 0])
        traj = traj[mask]
    if traj.numel() == 0:
        return

    xy = traj.cpu().numpy()
    ax.plot(
        xy[:, 0],
        xy[:, 1],
        color=color,
        alpha=alpha,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
    )
    ax.scatter(xy[-1, 0], xy[-1, 1], color=color, alpha=alpha, s=20)


def render_scene(scene_id, scene, pred, obs_length, pred_length, output_file, show_plot):
    scene = scene.detach().cpu()
    pred = pred.detach().cpu()

    obs = scene[:obs_length, 0:1, :]
    gt = scene[obs_length: obs_length + pred_length, 0:1, :]
    pred = pred[:, 0:1, :]

    ade_val = calc_ade(pred, gt)
    fde_val = calc_fde(pred, gt)

    fig, ax = plt.subplots(figsize=(8, 8))

    _plot_agent(ax, obs[:, 0, :], color="#2563eb", alpha=1.0, linewidth=2.6, label="Observed")
    _plot_agent(ax, gt[:, 0, :], color="#16a34a", alpha=1.0, linewidth=2.6, label="Actual")
    _plot_agent(ax, pred[:, 0, :], color="#dc2626", alpha=1.0, linewidth=2.6, label="Predicted")

    ax.scatter(obs[0, 0, 0].item(), obs[0, 0, 1].item(), color="#1d4ed8", s=48, marker="o", label="Start")

    ax.set_title(
        f"Scene {scene_id} | ADE={ade_val:.3f} m | FDE={fde_val:.3f} m"
    )
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

    dataset = TrajectoryDataset(
        data_root=args.data_root,
        data_part=args.data_part,
    )
    scene_index = select_scene(dataset, args.scene_index, args.obs_length, args.pred_length)
    scene_id, scene, goal = dataset[scene_index]

    model = build_model(args)
    pred = model.predict(scene, goal)

    output_file = args.output_file
    if output_file is None:
        output_file = f"outputs/{args.model}_scene_{scene_id}.png"

    render_scene(
        scene_id=scene_id,
        scene=scene,
        pred=pred,
        obs_length=args.obs_length,
        pred_length=args.pred_length,
        output_file=output_file,
        show_plot=args.show,
    )


if __name__ == "__main__":
    main()
