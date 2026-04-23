"""
online_simulator.py
-------------------
PedPy-backed multi-agent online RL environment that uses a frozen
trajectory-prediction prior to propose socially-aware motion, while a
learned residual policy adapts it online.

PedPy is used for:
  - walkable-area bookkeeping
  - Voronoi-neighborhood computation
  - neighbor-distance extraction for collision-aware rewards and observations
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pedpy
import torch
from shapely.geometry import Point, box
from shapely.ops import nearest_points


@dataclass
class PedPyStepInfo:
    mean_reward: float
    mean_fde: float
    collision_rate: float


def _forward_fill_scene(scene: torch.Tensor) -> torch.Tensor:
    """Forward-fill NaNs along time for each agent."""
    filled = scene.clone()
    T, N, _ = filled.shape
    for n in range(N):
        valid = ~torch.isnan(filled[:, n, 0])
        if not valid.any():
            filled[:, n] = 0.0
            continue

        first_valid = valid.nonzero(as_tuple=False)[0, 0]
        last = filled[first_valid, n].clone()
        filled[:first_valid, n] = last
        for t in range(first_valid, T):
            if torch.isnan(filled[t, n, 0]):
                filled[t, n] = last
            else:
                last = filled[t, n].clone()
    return filled


def _extract_goals(scene: torch.Tensor) -> torch.Tensor:
    """Use the last valid position of each agent as the goal."""
    T, N, _ = scene.shape
    goals = torch.zeros(N, 2, dtype=scene.dtype, device=scene.device)
    for n in range(N):
        valid = (~torch.isnan(scene[:, n, 0])).nonzero(as_tuple=False)
        if len(valid) == 0:
            continue
        goals[n] = scene[valid[-1, 0], n]
    return goals


def _filter_active_agents(scene: torch.Tensor, obs_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep agents present in the last observed frame."""
    last_obs = scene[obs_length - 1]
    keep_mask = ~torch.isnan(last_obs[:, 0])
    if not keep_mask.any():
        keep_mask[0] = True
    return scene[:, keep_mask], keep_mask


class FrozenScenePrior:
    """Frozen prior model used to propose one-step velocities per agent."""

    def __init__(
        self,
        backbone: torch.nn.Module,
        predict_mean_fn: callable,
        obs_length: int,
        pred_length: int,
        dt: float,
        max_prior_agents: int = 6,
        device: str = "cpu",
    ):
        self.backbone = backbone
        self.predict_mean_fn = predict_mean_fn
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.dt = dt
        self.max_prior_agents = max_prior_agents
        self.device = torch.device(device)

        self.backbone.to(self.device)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad_(False)

    def _local_scene(self, history: torch.Tensor, goals: torch.Tensor, agent_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        current = history[-1]
        rel = current - current[agent_idx:agent_idx + 1]
        dist = torch.norm(rel, dim=-1)
        order = torch.argsort(dist)
        if self.max_prior_agents is not None:
            order = order[: self.max_prior_agents]

        if order[0].item() != agent_idx:
            order = torch.cat([
                torch.tensor([agent_idx], device=order.device),
                order[order != agent_idx],
            ])[: self.max_prior_agents]

        local_scene = history[:, order].to(self.device)
        local_goals = goals[order].to(self.device)
        return local_scene, local_goals

    @torch.no_grad()
    def one_step_velocity(self, history: torch.Tensor, goals: torch.Tensor, agent_idx: int) -> torch.Tensor:
        local_scene, local_goals = self._local_scene(history, goals, agent_idx)
        mean_pred = self.predict_mean_fn(self.backbone, local_scene, local_goals)
        current_pos = local_scene[-1, 0]
        next_pos = mean_pred[0, 0]
        return (next_pos - current_pos) / self.dt

    @torch.no_grad()
    def all_prior_velocities(self, history: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        velocities = [
            self.one_step_velocity(history, goals, agent_idx)
            for agent_idx in range(history.shape[1])
        ]
        return torch.stack(velocities, dim=0)


class HeuristicScenePrior:
    """Model-free prior for simulator-only rollouts from observed history."""

    def __init__(
        self,
        dt: float,
        pred_length: int,
        max_speed: float = 2.0,
        history_window: int = 3,
        mode: str = "constant_velocity",
        repulsion_radius: float = 0.8,
        repulsion_weight: float = 0.35,
        goal_weight: float = 0.5,
    ):
        self.dt = dt
        self.pred_length = pred_length
        self.max_speed = max_speed
        self.history_window = max(1, history_window)
        self.mode = mode
        self.repulsion_radius = repulsion_radius
        self.repulsion_weight = repulsion_weight
        self.goal_weight = goal_weight

    def _recent_velocity(self, history: torch.Tensor) -> torch.Tensor:
        window = min(self.history_window, history.shape[0] - 1)
        if window <= 0:
            return torch.zeros(history.shape[1], 2, dtype=history.dtype, device=history.device)
        deltas = history[-window:] - history[-window - 1:-1]
        return deltas.mean(dim=0) / self.dt

    def _social_repulsion(self, positions: torch.Tensor) -> torch.Tensor:
        num_agents = positions.shape[0]
        repulsion = torch.zeros_like(positions)
        if num_agents < 2:
            return repulsion

        for agent_idx in range(num_agents):
            rel = positions[agent_idx:agent_idx + 1] - positions
            dist = torch.norm(rel, dim=-1, keepdim=True).clamp(min=1e-6)
            mask = (dist < self.repulsion_radius) & (dist > 1e-6)
            if mask.any():
                force = (rel / dist) * (self.repulsion_radius - dist).clamp(min=0.0)
                repulsion[agent_idx] = force[mask.expand_as(force)].view(-1, 2).sum(dim=0)
        return repulsion

    def all_prior_velocities(self, history: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        positions = history[-1]
        velocity = self._recent_velocity(history)

        if self.mode == "constant_velocity":
            proposed = velocity
        elif self.mode == "goal_directed":
            goal_dir = goals - positions
            goal_dist = torch.norm(goal_dir, dim=-1, keepdim=True).clamp(min=1e-6)
            goal_vel = goal_dir / goal_dist * velocity.norm(dim=-1, keepdim=True).clamp(min=0.2)
            proposed = (1.0 - self.goal_weight) * velocity + self.goal_weight * goal_vel
        elif self.mode == "social_force":
            goal_dir = goals - positions
            goal_dist = torch.norm(goal_dir, dim=-1, keepdim=True).clamp(min=1e-6)
            goal_vel = goal_dir / goal_dist * velocity.norm(dim=-1, keepdim=True).clamp(min=0.2)
            repulsion = self._social_repulsion(positions)
            proposed = (
                (1.0 - self.goal_weight) * velocity
                + self.goal_weight * goal_vel
                + self.repulsion_weight * repulsion
            )
        else:
            raise ValueError(f"Unknown heuristic prior mode: {self.mode}")

        speed = torch.norm(proposed, dim=-1, keepdim=True).clamp(min=1e-6)
        return torch.where(speed > self.max_speed, proposed * (self.max_speed / speed), proposed)


class PedPyPriorEnv:
    """Custom multi-agent environment using PedPy for social metrics."""

    def __init__(
        self,
        prior: FrozenScenePrior,
        obs_length: int = 9,
        pred_length: int = 12,
        dt: float = 0.4,
        max_speed: float = 2.0,
        residual_limit: float = 0.75,
        neighbor_k: int = 4,
        collision_radius: float = 0.3,
        world_margin: float = 2.0,
        progress_weight: float = 1.0,
        terminal_fde_weight: float = 1.0,
        collision_weight: float = 5.0,
        device: str = "cpu",
    ):
        self.prior = prior
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.dt = dt
        self.max_speed = max_speed
        self.residual_limit = residual_limit
        self.neighbor_k = neighbor_k
        self.collision_radius = collision_radius
        self.world_margin = world_margin
        self.progress_weight = progress_weight
        self.terminal_fde_weight = terminal_fde_weight
        self.collision_weight = collision_weight
        self.device = torch.device(device)
        self.frame_rate = 1.0 / dt

        self.history: torch.Tensor | None = None
        self.goals: torch.Tensor | None = None
        self.positions: torch.Tensor | None = None
        self.velocities: torch.Tensor | None = None
        self.last_prior_velocities: torch.Tensor | None = None
        self.step_idx: int = 0

        self.walkable_polygon = None
        self.walkable_area = None
        self.neighbor_distances = pd.DataFrame(columns=["id", "frame", "neighbor_id", "distance"])

    @property
    def obs_dim(self) -> int:
        return 8 + self.neighbor_k * 5

    def reset(self, scene: torch.Tensor) -> torch.Tensor:
        scene = scene.to(self.device)
        scene, keep_mask = _filter_active_agents(scene, self.obs_length)
        scene = _forward_fill_scene(scene)

        self.history = scene[: self.obs_length].clone()
        self.positions = self.history[-1].clone()
        self.velocities = (self.history[-1] - self.history[-2]) / self.dt
        self.goals = _extract_goals(scene)
        self.step_idx = 0

        self.walkable_polygon = self._build_walkable_polygon(scene)
        self.walkable_area = pedpy.WalkableArea(self.walkable_polygon)
        self.last_prior_velocities = self.prior.all_prior_velocities(self.history, self.goals)
        self.neighbor_distances = self._compute_neighbor_distances(self.positions)
        return self._build_observation()

    def _build_walkable_polygon(self, scene: torch.Tensor):
        valid = scene[~torch.isnan(scene)]
        if valid.numel() == 0:
            return box(-5.0, -5.0, 5.0, 5.0)

        coords = torch.nan_to_num(scene, nan=0.0).reshape(-1, 2)
        xs = coords[:, 0]
        ys = coords[:, 1]
        min_x = float(xs.min().item() - self.world_margin)
        max_x = float(xs.max().item() + self.world_margin)
        min_y = float(ys.min().item() - self.world_margin)
        max_y = float(ys.max().item() + self.world_margin)

        if max_x - min_x < 1e-3:
            min_x -= 1.0
            max_x += 1.0
        if max_y - min_y < 1e-3:
            min_y -= 1.0
            max_y += 1.0

        return box(min_x, min_y, max_x, max_y)

    def _trajectory_dataframe(self, positions: torch.Tensor) -> pd.DataFrame:
        pos_cpu = positions.detach().cpu()
        return pd.DataFrame(
            {
                "id": list(range(pos_cpu.shape[0])),
                "frame": [self.step_idx] * pos_cpu.shape[0],
                "x": pos_cpu[:, 0].tolist(),
                "y": pos_cpu[:, 1].tolist(),
            }
        )

    def _compute_neighbor_distances(self, positions: torch.Tensor) -> pd.DataFrame:
        if positions.shape[0] < 2:
            return pd.DataFrame(columns=["id", "frame", "neighbor_id", "distance"])

        try:
            traj = pedpy.TrajectoryData(self._trajectory_dataframe(positions), frame_rate=self.frame_rate)
            voronoi = pedpy.compute_individual_voronoi_polygons(
                traj_data=traj,
                walkable_area=self.walkable_area,
                use_blind_points=True,
            )
            neighbors = pedpy.compute_neighbors(voronoi, as_list=False)
            if neighbors.empty:
                return pd.DataFrame(columns=["id", "frame", "neighbor_id", "distance"])
            return pedpy.compute_neighbor_distance(traj_data=traj, neighborhood=neighbors)
        except Exception:
            return pd.DataFrame(columns=["id", "frame", "neighbor_id", "distance"])

    def _project_inside_walkable_area(self, point_xy: torch.Tensor) -> torch.Tensor:
        point = Point(float(point_xy[0].item()), float(point_xy[1].item()))
        if self.walkable_polygon.covers(point):
            return point_xy

        projected = nearest_points(self.walkable_polygon, point)[0]
        return torch.tensor([projected.x, projected.y], dtype=point_xy.dtype, device=self.device)

    def _neighbor_rows(self, agent_idx: int) -> pd.DataFrame:
        if self.neighbor_distances.empty:
            return self.neighbor_distances
        rows = self.neighbor_distances[self.neighbor_distances["id"] == agent_idx]
        return rows.sort_values("distance")

    def _build_observation(self) -> torch.Tensor:
        observations = []
        num_agents = self.positions.shape[0]
        time_remaining = torch.tensor(
            [(self.pred_length - self.step_idx) / max(self.pred_length, 1)],
            dtype=self.positions.dtype,
            device=self.device,
        )

        for agent_idx in range(num_agents):
            pos = self.positions[agent_idx]
            vel = self.velocities[agent_idx]
            prior_vel = self.last_prior_velocities[agent_idx]
            goal_vec = self.goals[agent_idx] - pos
            goal_dist = torch.norm(goal_vec, dim=0, keepdim=True)

            features = [vel, prior_vel, goal_vec, goal_dist, time_remaining]

            neighbor_features = []
            rows = self._neighbor_rows(agent_idx)
            for _, row in rows.head(self.neighbor_k).iterrows():
                other_idx = int(row["neighbor_id"])
                rel_pos = self.positions[other_idx] - pos
                rel_vel = self.velocities[other_idx] - vel
                dist = torch.tensor([float(row["distance"])], dtype=self.positions.dtype, device=self.device)
                neighbor_features.append(torch.cat([rel_pos, rel_vel, dist], dim=0))

            while len(neighbor_features) < self.neighbor_k:
                neighbor_features.append(torch.zeros(5, dtype=self.positions.dtype, device=self.device))

            observations.append(torch.cat(features + neighbor_features, dim=0))

        return torch.stack(observations, dim=0)

    def step(self, residual_actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool, PedPyStepInfo]:
        residual_actions = residual_actions.to(self.device).clamp(
            min=-self.residual_limit,
            max=self.residual_limit,
        )
        prior_velocities = self.prior.all_prior_velocities(self.history, self.goals)
        candidate_vel = prior_velocities + residual_actions

        speed = torch.norm(candidate_vel, dim=-1, keepdim=True).clamp(min=1e-6)
        candidate_vel = torch.where(
            speed > self.max_speed,
            candidate_vel * (self.max_speed / speed),
            candidate_vel,
        )

        next_positions = self.positions + self.dt * candidate_vel
        next_positions = torch.stack(
            [self._project_inside_walkable_area(p) for p in next_positions],
            dim=0,
        )

        prev_goal_dist = torch.norm(self.goals - self.positions, dim=-1)
        new_goal_dist = torch.norm(self.goals - next_positions, dim=-1)
        progress_reward = prev_goal_dist - new_goal_dist

        self.step_idx += 1
        self.positions = next_positions
        self.velocities = candidate_vel
        self.history = torch.cat([self.history[1:], next_positions.unsqueeze(0)], dim=0)
        self.last_prior_velocities = prior_velocities
        self.neighbor_distances = self._compute_neighbor_distances(self.positions)

        collision_penalty = torch.zeros(self.positions.shape[0], dtype=self.positions.dtype, device=self.device)
        if not self.neighbor_distances.empty:
            grouped = self.neighbor_distances.groupby("id")["distance"].apply(list).to_dict()
            for agent_idx, distances in grouped.items():
                dist_tensor = torch.tensor(distances, dtype=self.positions.dtype, device=self.device)
                collision_penalty[int(agent_idx)] = torch.relu(self.collision_radius - dist_tensor).sum()

        reward = self.progress_weight * progress_reward - self.collision_weight * collision_penalty
        done = self.step_idx >= self.pred_length
        if done:
            reward = reward - self.terminal_fde_weight * new_goal_dist

        obs = self._build_observation()
        info = PedPyStepInfo(
            mean_reward=float(reward.mean().item()),
            mean_fde=float(new_goal_dist.mean().item()),
            collision_rate=float((collision_penalty > 0).float().mean().item()),
        )
        return obs, reward, done, info


class KNNRewardPriorEnv(PedPyPriorEnv):
    """Same simulator as PedPyPriorEnv, but collision reward uses Euclidean kNN distances."""

    def _knn_collision_penalty(self, positions: torch.Tensor) -> torch.Tensor:
        num_agents = positions.shape[0]
        penalties = torch.zeros(num_agents, dtype=positions.dtype, device=self.device)
        if num_agents < 2 or self.neighbor_k <= 0:
            return penalties

        dists = torch.cdist(positions, positions)
        dists.fill_diagonal_(float("inf"))
        k = min(self.neighbor_k, num_agents - 1)
        nearest_dists, _ = torch.topk(dists, k=k, largest=False, dim=1)
        penalties = torch.relu(self.collision_radius - nearest_dists).sum(dim=1)
        return penalties

    def step(self, residual_actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool, PedPyStepInfo]:
        residual_actions = residual_actions.to(self.device).clamp(
            min=-self.residual_limit,
            max=self.residual_limit,
        )
        prior_velocities = self.prior.all_prior_velocities(self.history, self.goals)
        candidate_vel = prior_velocities + residual_actions

        speed = torch.norm(candidate_vel, dim=-1, keepdim=True).clamp(min=1e-6)
        candidate_vel = torch.where(
            speed > self.max_speed,
            candidate_vel * (self.max_speed / speed),
            candidate_vel,
        )

        next_positions = self.positions + self.dt * candidate_vel
        next_positions = torch.stack(
            [self._project_inside_walkable_area(p) for p in next_positions],
            dim=0,
        )

        prev_goal_dist = torch.norm(self.goals - self.positions, dim=-1)
        new_goal_dist = torch.norm(self.goals - next_positions, dim=-1)
        progress_reward = prev_goal_dist - new_goal_dist

        self.step_idx += 1
        self.positions = next_positions
        self.velocities = candidate_vel
        self.history = torch.cat([self.history[1:], next_positions.unsqueeze(0)], dim=0)
        self.last_prior_velocities = prior_velocities
        self.neighbor_distances = self._compute_neighbor_distances(self.positions)

        collision_penalty = self._knn_collision_penalty(self.positions)

        reward = self.progress_weight * progress_reward - self.collision_weight * collision_penalty
        done = self.step_idx >= self.pred_length
        if done:
            reward = reward - self.terminal_fde_weight * new_goal_dist

        obs = self._build_observation()
        info = PedPyStepInfo(
            mean_reward=float(reward.mean().item()),
            mean_fde=float(new_goal_dist.mean().item()),
            collision_rate=float((collision_penalty > 0).float().mean().item()),
        )
        return obs, reward, done, info
