"""
make_online_demo_video.py
-------------------------
Create a short demo video for an online PPO residual policy using a synthetic
multi-agent scene. The first obs_length waypoints are hand-crafted estimates;
the future is rolled out by the frozen prior plus the online residual policy.

By default this script uses the transformer prior and transformer online policy.

Example
-------
    python make_online_demo_video.py \
        --model transformer \
        --policy_checkpoint checkpoints/transformer_online.pth \
        --prior_checkpoint checkpoints/transformer.pth \
        --output_file outputs/online_transformer_demo.mp4 \
        --device cuda
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from online_inference import parse_args_defaults, resolve_policy_metadata, rollout_scene
from online_simulator import FrozenScenePrior, PedPyPriorEnv
from ppo_finetune import build_backbone, load_backbone_checkpoint, set_seed
from train_online_policy import ResidualActorCritic


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a synthetic online-policy demo video."
    )
    parser.add_argument(
        "--model",
        default="transformer",
        choices=["d_pool", "social_lstm", "autobot", "eq_motion", "transformer"],
    )
    parser.add_argument(
        "--policy_checkpoint",
        default="checkpoints/transformer_online.pth",
    )
    parser.add_argument(
        "--prior_checkpoint",
        default="checkpoints/transformer.pth",
    )
    parser.add_argument("--obs_length", type=int, default=9)
    parser.add_argument("--pred_length", type=int, default=12)
    parser.add_argument("--total_frames", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num_neighbors", type=int, default=3)
    parser.add_argument(
        "--fixed_neighbors",
        action="store_true",
        help="Keep neighbor future motion scripted and predict only the ego future.",
    )

    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--dt", type=float, default=0.4)
    parser.add_argument("--max_speed", type=float, default=2.0)
    parser.add_argument("--residual_limit", type=float, default=0.75)
    parser.add_argument("--neighbor_k", type=int, default=4)
    parser.add_argument("--collision_radius", type=float, default=0.3)
    parser.add_argument(
        "--visual_collision_radius",
        type=float,
        default=None,
        help="Collision distance used only for coloring/highlighting in the video. Defaults to max(collision_radius, 0.5).",
    )
    parser.add_argument("--world_margin", type=float, default=2.0)
    parser.add_argument("--reward_progress_weight", type=float, default=1.0)
    parser.add_argument("--reward_fde_weight", type=float, default=1.0)
    parser.add_argument("--reward_collision_weight", type=float, default=5.0)
    parser.add_argument("--max_prior_agents", type=int, default=6)

    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", default="outputs/online_transformer_demo.mp4")
    return parser.parse_args()


def apply_total_frames(args):
    if args.total_frames is not None:
        if args.total_frames <= args.obs_length:
            raise ValueError("--total_frames must be greater than --obs_length.")
        args.pred_length = args.total_frames - args.obs_length


def visual_collision_radius(args) -> float:
    if args.visual_collision_radius is not None:
        return args.visual_collision_radius
    return max(args.collision_radius, 0.5)


def resolve_runtime_args(args):
    checkpoint = resolve_policy_metadata(args)
    defaults = parse_args_defaults()
    meta_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    for name, default in defaults.items():
        if getattr(args, name, default) == default and name in meta_args:
            setattr(args, name, meta_args[name])
    return checkpoint


def checkpoint_runtime_args(args, checkpoint) -> argparse.Namespace:
    """Build model-loading args from checkpoint metadata without overriding video horizon."""
    runtime = copy.deepcopy(args)
    meta_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    for name in ["obs_length", "pred_length"]:
        if name in meta_args:
            setattr(runtime, name, meta_args[name])
    return runtime


def _linspace_traj(start_xy, velocity_xy, steps: int, dt: float) -> torch.Tensor:
    pts = [
        torch.tensor(start_xy, dtype=torch.float32) + i * dt * torch.tensor(velocity_xy, dtype=torch.float32)
        for i in range(steps)
    ]
    return torch.stack(pts, dim=0)


def _rand_uniform(generator: torch.Generator, low: float, high: float) -> float:
    return low + (high - low) * torch.rand((), generator=generator).item()


def _neighbor_specs(num_neighbors: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    base_specs = [
        {"start": (-0.6, -3.2), "vel": (0.02, 0.92), "curve_amp": 0.05, "curve_freq": 0.17, "curve_phase": 0.0, "curve_axis": 0},
        {"start": (2.6, 3.0), "vel": (-0.12, -0.85), "curve_amp": 0.06, "curve_freq": 0.16, "curve_phase": 0.8, "curve_axis": 0},
        {"start": (-2.0, 2.2), "vel": (0.95, -0.02), "curve_amp": 0.10, "curve_freq": 0.22, "curve_phase": 0.6, "curve_axis": 1},
        {"start": (3.6, -1.8), "vel": (-0.78, 0.10), "curve_amp": 0.08, "curve_freq": 0.14, "curve_phase": 1.2, "curve_axis": 1},
        {"start": (-3.3, 3.5), "vel": (0.62, -0.42), "curve_amp": 0.07, "curve_freq": 0.19, "curve_phase": 2.0, "curve_axis": 0},
        {"start": (1.2, -4.0), "vel": (0.05, 0.88), "curve_amp": 0.05, "curve_freq": 0.20, "curve_phase": 1.5, "curve_axis": 0},
    ]
    template_order = torch.randperm(len(base_specs), generator=generator).tolist()

    specs = []
    for idx in range(num_neighbors):
        template = base_specs[template_order[idx % len(base_specs)]]
        ring = idx // len(base_specs)
        direction = torch.tensor(template["vel"], dtype=torch.float32)
        direction = direction / direction.norm().clamp(min=1e-6)
        perpendicular = torch.tensor([-direction[1].item(), direction[0].item()], dtype=torch.float32)

        lane_offset = (ring + 1) * 1.35 * (1.0 if idx % 2 == 0 else -1.0)
        jitter = torch.tensor(
            [
                _rand_uniform(generator, -0.22, 0.22),
                _rand_uniform(generator, -0.22, 0.22),
            ],
            dtype=torch.float32,
        )
        start = torch.tensor(template["start"], dtype=torch.float32) + lane_offset * perpendicular + jitter

        speed_scale = _rand_uniform(generator, 0.92, 1.08)
        vel_jitter = torch.tensor(
            [
                _rand_uniform(generator, -0.05, 0.05),
                _rand_uniform(generator, -0.05, 0.05),
            ],
            dtype=torch.float32,
        )
        velocity = torch.tensor(template["vel"], dtype=torch.float32) * speed_scale + vel_jitter

        specs.append(
            {
                "start": (float(start[0].item()), float(start[1].item())),
                "vel": (float(velocity[0].item()), float(velocity[1].item())),
                "curve_amp": template["curve_amp"] * _rand_uniform(generator, 0.85, 1.25),
                "curve_freq": template["curve_freq"] * _rand_uniform(generator, 0.9, 1.1),
                "curve_phase": _rand_uniform(generator, 0.0, 2.0 * np.pi),
                "curve_axis": int(torch.randint(0, 2, (1,), generator=generator).item())
                if idx >= len(base_specs)
                else template["curve_axis"],
            }
        )
    return specs


def build_synthetic_scene(obs_length: int, pred_length: int, dt: float, num_neighbors: int, seed: int) -> torch.Tensor:
    total = obs_length + pred_length

    # Ego approaches a busy crossing corridor with enough nearby context for the
    # residual policy to react to, while neighbors remain well-separated.
    ego = _linspace_traj(start_xy=(-4.0, -0.4), velocity_xy=(1.15, 0.08), steps=total, dt=dt)
    agents = [ego]
    for spec in _neighbor_specs(num_neighbors, seed=seed):
        traj = _linspace_traj(start_xy=spec["start"], velocity_xy=spec["vel"], steps=total, dt=dt)
        time = torch.arange(total, dtype=torch.float32)
        traj[:, spec["curve_axis"]] += spec["curve_amp"] * torch.sin(time * spec["curve_freq"] + spec["curve_phase"])
        agents.append(traj)

    scene = torch.stack(agents, dim=1)  # [T, N, 2]

    # Add a tiny amount of curvature to make the observed history less perfectly linear.
    time = torch.arange(total, dtype=torch.float32)
    scene[:, 0, 1] += 0.12 * torch.sin(time * 0.18)
    return scene


def load_online_policy(args, device: torch.device):
    checkpoint = resolve_runtime_args(args)
    backbone_args = checkpoint_runtime_args(args, checkpoint)

    backbone, predict_mean_fn = build_backbone(backbone_args)
    backbone = backbone.to(device)
    load_backbone_checkpoint(backbone, args.prior_checkpoint, device)

    if backbone_args.pred_length != args.pred_length:
        print(
            "Note: prior checkpoint was trained with "
            f"pred_length={backbone_args.pred_length}; using that horizon for the backbone "
            f"while rolling out a {args.pred_length}-step demo."
        )
    if backbone_args.obs_length != args.obs_length:
        print(
            "Note: prior checkpoint was trained with "
            f"obs_length={backbone_args.obs_length}; expected runtime obs_length={args.obs_length}."
        )

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

    dummy_scene = build_synthetic_scene(args.obs_length, args.pred_length, args.dt, args.num_neighbors, args.seed)
    dummy_obs = env.reset(dummy_scene)
    policy = ResidualActorCritic(
        obs_dim=dummy_obs.shape[-1],
        hidden_dim=args.hidden_dim,
        action_dim=2,
    ).to(device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    policy.load_state_dict(state_dict, strict=True)
    policy.eval()
    return policy, env


def build_demo_trajectories(policy, env, args, device: torch.device):
    scene = build_synthetic_scene(args.obs_length, args.pred_length, args.dt, args.num_neighbors, args.seed)
    if args.fixed_neighbors:
        pred_future = rollout_with_fixed_neighbors(policy, env, scene, device).detach().cpu()
    else:
        pred_future = rollout_scene(policy, env, scene, device).detach().cpu()
    observed = scene[: args.obs_length].detach().cpu()
    full_pred = torch.cat([observed, pred_future], dim=0)
    return full_pred


@torch.no_grad()
def rollout_with_fixed_neighbors(policy, env, scene: torch.Tensor, device: torch.device) -> torch.Tensor:
    scene = scene.to(device)
    env.reset(scene)
    scripted_future = scene[env.obs_length: env.obs_length + env.pred_length]
    predictions = []

    while env.step_idx < env.pred_length:
        obs = env._build_observation()
        action, _, _, _ = policy.act(obs.to(device), deterministic=True)
        residual_actions = torch.zeros_like(env.positions)
        residual_actions[0] = action[0]
        residual_actions = residual_actions.clamp(min=-env.residual_limit, max=env.residual_limit)

        prior_velocities = env.prior.all_prior_velocities(env.history, env.goals)
        candidate_vel = prior_velocities + residual_actions
        speed = torch.norm(candidate_vel, dim=-1, keepdim=True).clamp(min=1e-6)
        candidate_vel = torch.where(
            speed > env.max_speed,
            candidate_vel * (env.max_speed / speed),
            candidate_vel,
        )

        next_positions = env.positions.clone()
        next_positions[0] = env._project_inside_walkable_area(env.positions[0] + env.dt * candidate_vel[0])
        if next_positions.shape[0] > 1:
            next_positions[1:] = scripted_future[env.step_idx, 1:]

        next_velocities = torch.zeros_like(env.velocities)
        next_velocities[0] = candidate_vel[0]
        if next_positions.shape[0] > 1:
            next_velocities[1:] = (next_positions[1:] - env.positions[1:]) / env.dt

        env.step_idx += 1
        env.positions = next_positions
        env.velocities = next_velocities
        env.history = torch.cat([env.history[1:], next_positions.unsqueeze(0)], dim=0)
        env.last_prior_velocities = prior_velocities
        env.neighbor_distances = env._compute_neighbor_distances(env.positions)
        predictions.append(env.positions.detach().clone())

    return torch.stack(predictions, dim=0)


def colliding_agents(positions: torch.Tensor, threshold: float) -> set[int]:
    collided = set()
    if positions.shape[0] < 2:
        return collided

    for i in range(positions.shape[0]):
        for j in range(i + 1, positions.shape[0]):
            dist = torch.norm(positions[i] - positions[j], dim=0).item()
            if dist < threshold:
                collided.add(i)
                collided.add(j)
    return collided


def render_frame(
    frame_idx: int,
    traj: torch.Tensor,
    obs_length: int,
    size: tuple[int, int],
    collision_radius: float,
    fixed_neighbors: bool,
) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    frame_positions = traj[frame_idx]
    collided = colliding_agents(frame_positions, collision_radius)

    for agent_idx in range(traj.shape[1]):
        current = traj[: frame_idx + 1, agent_idx].numpy()
        is_ego = agent_idx == 0
        alpha = 1.0 if is_ego else 0.85
        linewidth = 3.0 if is_ego else 2.0
        color = "#2563eb" if is_ego else "#111111"
        if agent_idx in collided:
            color = "#dc2626"
        label = "Ego" if is_ego else f"Neighbor {agent_idx}"
        ax.plot(current[:, 0], current[:, 1], color=color, linewidth=linewidth, alpha=alpha)
        edgecolor = "#7f1d1d" if agent_idx in collided else color
        ax.scatter(
            current[-1, 0],
            current[-1, 1],
            color=color,
            edgecolors=edgecolor,
            linewidths=2.0 if agent_idx in collided else 0.5,
            s=110 if is_ego else 65,
            label=label,
            zorder=10,
        )

        if frame_idx + 1 >= obs_length:
            obs_tail = traj[:obs_length, agent_idx].numpy()
            ax.plot(obs_tail[:, 0], obs_tail[:, 1], color=color, linestyle="--", linewidth=1.2, alpha=0.35)

    phase = "Observed" if frame_idx < obs_length else "Online Rollout"
    ax.set_title(f"Transformer Prior + Online Residual | {phase} | Frame {frame_idx + 1}/{traj.shape[0]}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")

    pad = 0.8
    x_min = float(traj[:, :, 0].min().item() - pad)
    x_max = float(traj[:, :, 0].max().item() + pad)
    y_min = float(traj[:, :, 1].min().item() - pad)
    y_max = float(traj[:, :, 1].max().item() + pad)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")

    note = (
        "Observed waypoints are hand-crafted.\n"
        f"Neighbors: {'scripted future' if fixed_neighbors else 'model rollout'}.\n"
        "Future ego uses transformer prior + PPO residual.\n"
        f"Collision: {'Yes' if collided else 'No'}"
    )
    ax.text(
        0.02,
        0.02,
        note,
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="#cbd5e1"),
    )

    fig.tight_layout()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[1], fig.canvas.get_width_height()[0], 4)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    return image


def write_video(frames: list[np.ndarray], output_path: Path, fps: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    for frame in frames:
        writer.write(frame)
    writer.release()


def main():
    args = parse_args()
    apply_total_frames(args)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    set_seed(args.seed)

    policy, env = load_online_policy(args, device)
    traj = build_demo_trajectories(policy, env, args, device)

    if traj.shape[0] < 20:
        raise RuntimeError(f"Demo has only {traj.shape[0]} frames; expected at least 20.")

    frames = [
        render_frame(
            frame_idx,
            traj,
            args.obs_length,
            size=(args.width, args.height),
            collision_radius=visual_collision_radius(args),
            fixed_neighbors=args.fixed_neighbors,
        )
        for frame_idx in range(traj.shape[0])
    ]

    output_path = Path(args.output_file)
    write_video(frames, output_path, args.fps)
    print(f"Saved demo video to {output_path}")


if __name__ == "__main__":
    main()
