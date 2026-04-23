"""
visualize_pedpy_frame.py
------------------------
Visualize PedPy measurements for one frame from a dataset scene:
  - agent positions
  - walkable area
  - Voronoi cells
  - ego-neighbor links and distances

Example
-------
    python visualize_pedpy_frame.py \
        --data_root Data \
        --data_part train \
        --scene_index 0 \
        --output_file outputs/pedpy_frame.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pedpy
import torch

from dataset import TrajectoryDataset
from online_simulator import _filter_active_agents, _forward_fill_scene
from shapely.geometry import box


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize PedPy Voronoi and neighbor measurements for a sample frame."
    )
    parser.add_argument("--data_root", default="Data")
    parser.add_argument(
        "--data_part",
        default="train",
        choices=["train", "test", "secret"],
    )
    parser.add_argument("--scene_index", type=int, default=0)
    parser.add_argument("--sample", type=float, default=1.0)
    parser.add_argument(
        "--frame_index",
        type=int,
        default=None,
        help="Frame to visualize after preprocessing. Defaults to obs_length - 1.",
    )
    parser.add_argument("--obs_length", type=int, default=9)
    parser.add_argument("--world_margin", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=0.4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def build_walkable_polygon(scene: torch.Tensor, world_margin: float):
    coords = torch.nan_to_num(scene, nan=0.0).reshape(-1, 2)
    xs = coords[:, 0]
    ys = coords[:, 1]
    min_x = float(xs.min().item() - world_margin)
    max_x = float(xs.max().item() + world_margin)
    min_y = float(ys.min().item() - world_margin)
    max_y = float(ys.max().item() + world_margin)

    if max_x - min_x < 1e-3:
        min_x -= 1.0
        max_x += 1.0
    if max_y - min_y < 1e-3:
        min_y -= 1.0
        max_y += 1.0
    return box(min_x, min_y, max_x, max_y)


def trajectory_dataframe(positions: torch.Tensor, frame_idx: int) -> pd.DataFrame:
    pos = positions.detach().cpu()
    return pd.DataFrame(
        {
            "id": list(range(pos.shape[0])),
            "frame": [frame_idx] * pos.shape[0],
            "x": pos[:, 0].tolist(),
            "y": pos[:, 1].tolist(),
        }
    )


def select_scene(dataset, scene_index: int):
    if scene_index < 0 or scene_index >= len(dataset):
        raise IndexError(f"scene_index={scene_index} is out of range for dataset size {len(dataset)}.")
    return dataset[scene_index]


def render(scene_id, frame_idx, positions, walkable_polygon, voronoi, distances, output_file, show_plot):
    positions = positions.detach().cpu()
    fig, ax = plt.subplots(figsize=(8, 8))

    wx, wy = walkable_polygon.exterior.xy
    ax.fill(wx, wy, color="#f8fafc", alpha=1.0, zorder=0)
    ax.plot(wx, wy, color="#cbd5e1", linewidth=1.5, zorder=1, label="Walkable area")

    for _, row in voronoi.iterrows():
        poly = row["polygon"]
        x, y = poly.exterior.xy
        is_ego = int(row["id"]) == 0
        edge = "#2563eb" if is_ego else "#64748b"
        face = "#dbeafe" if is_ego else "#e5e7eb"
        ax.fill(x, y, color=face, alpha=0.28, zorder=2)
        ax.plot(x, y, color=edge, linewidth=1.5 if is_ego else 1.0, zorder=3)

        centroid = poly.centroid
        ax.text(
            centroid.x,
            centroid.y,
            f"id={int(row['id'])}\nd={row['density']:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color=edge,
            zorder=4,
        )

    ego = positions[0]
    for _, row in distances.iterrows():
        if int(row["id"]) != 0:
            continue
        nbr_idx = int(row["neighbor_id"])
        nbr = positions[nbr_idx]
        ax.plot(
            [ego[0].item(), nbr[0].item()],
            [ego[1].item(), nbr[1].item()],
            color="#f59e0b",
            linewidth=2.0,
            alpha=0.85,
            zorder=5,
        )
        mid_x = 0.5 * (ego[0].item() + nbr[0].item())
        mid_y = 0.5 * (ego[1].item() + nbr[1].item())
        ax.text(mid_x, mid_y, f"{row['distance']:.2f} m", fontsize=8, color="#b45309", zorder=6)

    for idx in range(positions.shape[0]):
        is_ego = idx == 0
        color = "#2563eb" if is_ego else "#111111"
        label = "Ego" if is_ego else ("Neighbors" if idx == 1 else None)
        ax.scatter(
            positions[idx, 0].item(),
            positions[idx, 1].item(),
            s=90 if is_ego else 55,
            color=color,
            zorder=7,
            label=label,
        )
        ax.text(
            positions[idx, 0].item() + 0.04,
            positions[idx, 1].item() + 0.04,
            str(idx),
            fontsize=8,
            color=color,
            zorder=8,
        )

    ax.set_title(f"Scene {scene_id} | Frame {frame_idx} | PedPy Voronoi + Ego Neighbors")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")
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

    dataset = TrajectoryDataset(
        data_root=args.data_root,
        data_part=args.data_part,
        sample=args.sample,
        device=str(device),
    )
    scene_id, raw_scene, _ = select_scene(dataset, args.scene_index)

    scene, _ = _filter_active_agents(raw_scene, args.obs_length)
    scene = _forward_fill_scene(scene)
    frame_idx = args.obs_length - 1 if args.frame_index is None else args.frame_index
    if frame_idx < 0 or frame_idx >= scene.shape[0]:
        raise IndexError(f"frame_index={frame_idx} is out of range for scene length {scene.shape[0]}.")

    positions = scene[frame_idx]
    walkable_polygon = build_walkable_polygon(scene, args.world_margin)
    walkable_area = pedpy.WalkableArea(walkable_polygon)
    traj = pedpy.TrajectoryData(
        trajectory_dataframe(positions, frame_idx),
        frame_rate=1.0 / args.dt,
    )
    voronoi = pedpy.compute_individual_voronoi_polygons(
        traj_data=traj,
        walkable_area=walkable_area,
        use_blind_points=True,
    )
    neighbors = pedpy.compute_neighbors(voronoi, as_list=False)
    distances = (
        pedpy.compute_neighbor_distance(traj_data=traj, neighborhood=neighbors)
        if not neighbors.empty
        else pd.DataFrame(columns=["id", "frame", "neighbor_id", "distance"])
    )

    output_file = args.output_file
    if output_file is None:
        output_file = f"outputs/pedpy_scene_{scene_id}_frame_{frame_idx}.png"

    render(
        scene_id=scene_id,
        frame_idx=frame_idx,
        positions=positions,
        walkable_polygon=walkable_polygon,
        voronoi=voronoi,
        distances=distances,
        output_file=output_file,
        show_plot=args.show,
    )


if __name__ == "__main__":
    main()
