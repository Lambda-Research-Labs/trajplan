"""
train_online_policy.py
----------------------
Train a shared multi-agent online policy with PPO using a frozen offline prior
from a supervised or PPO-finetuned trajectory backbone.

The simulator uses PedPy to compute Voronoi neighborhoods and neighbor
distances at every step, and the online policy learns residual actions on top
of the frozen prior.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import random_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from dataset import TrajectoryDataset
from online_simulator import FrozenScenePrior, PedPyPriorEnv
from ppo_finetune import build_backbone, load_backbone_checkpoint, set_seed


class ResidualActorCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 128, action_dim: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(obs)
        return self.actor(feat), self.critic(feat).squeeze(-1)

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        mean, value = self.forward(obs)
        std = self.log_std.clamp(min=-5.0, max=2.0).exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        action = mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        mean, value = self.forward(obs)
        std = self.log_std.clamp(min=-5.0, max=2.0).exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


def valid_scene_indices(dataset, obs_length: int, pred_length: int) -> list[int]:
    indices = []
    for idx in range(len(dataset)):
        _, scene, _ = dataset[idx]
        if scene.shape[0] >= obs_length + pred_length:
            indices.append(idx)
    return indices


def compute_gae(
    rewards: list[torch.Tensor],
    values: list[torch.Tensor],
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    T = len(rewards)
    advantages = [torch.zeros_like(rewards[0]) for _ in range(T)]
    gae = torch.zeros_like(rewards[0])
    next_value = torch.zeros_like(rewards[0])

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae
        next_value = values[t]

    adv = torch.stack(advantages, dim=0)
    ret = adv + torch.stack(values, dim=0)
    return adv, ret


def collect_rollouts(
    policy: ResidualActorCritic,
    env: PedPyPriorEnv,
    dataset,
    scene_indices: list[int],
    args,
    device: torch.device,
) -> tuple[dict, dict]:
    obs_buf, act_buf, logp_buf, adv_buf, ret_buf, val_buf = [], [], [], [], [], []
    reward_means, fde_means, collision_rates = [], [], []

    episodes = 0
    scene_order = torch.randperm(len(scene_indices)).tolist()
    progress = tqdm(
        scene_order,
        total=min(len(scene_order), args.rollout_episodes),
        desc="  rollout",
        leave=False,
        disable=not args.show_progress,
    )
    for perm_idx in progress:
        scene_idx = scene_indices[perm_idx]
        _, scene, _ = dataset[scene_idx]
        obs = env.reset(scene)

        ep_obs, ep_actions, ep_logps, ep_values, ep_rewards = [], [], [], [], []
        done = False
        last_info = None

        while not done:
            obs_t = obs.to(device)
            with torch.no_grad():
                action, log_prob, _, value = policy.act(obs_t, deterministic=False)
            next_obs, reward, done, info = env.step(action)

            ep_obs.append(obs_t.cpu())
            ep_actions.append(action.cpu())
            ep_logps.append(log_prob.cpu())
            ep_values.append(value.cpu())
            ep_rewards.append(reward.cpu())

            obs = next_obs
            last_info = info

        advantages, returns = compute_gae(
            rewards=ep_rewards,
            values=ep_values,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )

        obs_buf.append(torch.cat(ep_obs, dim=0))
        act_buf.append(torch.cat(ep_actions, dim=0))
        logp_buf.append(torch.cat(ep_logps, dim=0))
        val_buf.append(torch.cat(ep_values, dim=0))
        adv_buf.append(advantages.reshape(-1))
        ret_buf.append(returns.reshape(-1))

        reward_means.append(last_info.mean_reward if last_info is not None else 0.0)
        fde_means.append(last_info.mean_fde if last_info is not None else float("inf"))
        collision_rates.append(last_info.collision_rate if last_info is not None else 0.0)

        episodes += 1
        progress.set_postfix(
            episodes=episodes,
            reward=f"{reward_means[-1]:.3f}",
            fde=f"{fde_means[-1]:.3f}",
            col=f"{collision_rates[-1] * 100:.1f}%",
        )
        if episodes >= args.rollout_episodes:
            break
    progress.close()

    if not obs_buf:
        empty = torch.empty(0)
        return {
            "obs": empty,
            "actions": empty,
            "log_probs": empty,
            "advantages": empty,
            "returns": empty,
        }, {
            "reward": float("-inf"),
            "fde": float("inf"),
            "collision_rate": 0.0,
            "episodes": 0,
        }

    advantages = torch.cat(adv_buf, dim=0)
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    rollouts = {
        "obs": torch.cat(obs_buf, dim=0),
        "actions": torch.cat(act_buf, dim=0),
        "log_probs": torch.cat(logp_buf, dim=0),
        "advantages": advantages,
        "returns": torch.cat(ret_buf, dim=0),
    }
    stats = {
        "reward": float(np.mean(reward_means)),
        "fde": float(np.mean(fde_means)),
        "collision_rate": float(np.mean(collision_rates)),
        "episodes": episodes,
    }
    return rollouts, stats


def ppo_update(policy: ResidualActorCritic, optimizer, rollouts: dict, args, device: torch.device) -> dict:
    if rollouts["obs"].numel() == 0:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    obs = rollouts["obs"].to(device)
    actions = rollouts["actions"].to(device)
    old_log_probs = rollouts["log_probs"].to(device)
    advantages = rollouts["advantages"].to(device)
    returns = rollouts["returns"].to(device)

    policy_losses, value_losses, entropies = [], [], []
    batch_size = obs.shape[0]

    for _ in range(args.ppo_epochs):
        indices = torch.randperm(batch_size, device=device)
        for start in range(0, batch_size, args.minibatch_size):
            batch_idx = indices[start:start + args.minibatch_size]

            new_log_probs, entropy, values = policy.evaluate_actions(obs[batch_idx], actions[batch_idx])
            ratio = torch.exp(new_log_probs - old_log_probs[batch_idx])
            unclipped = ratio * advantages[batch_idx]
            clipped = torch.clamp(
                ratio,
                1.0 - args.clip_epsilon,
                1.0 + args.clip_epsilon,
            ) * advantages[batch_idx]

            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = nn.functional.mse_loss(values, returns[batch_idx])
            entropy_mean = entropy.mean()

            loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy_mean

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy_mean.item())

    return {
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropies)),
    }


@torch.no_grad()
def evaluate_policy(
    policy: ResidualActorCritic,
    env: PedPyPriorEnv,
    dataset,
    scene_indices: list[int],
    args,
    device: torch.device,
) -> dict:
    reward_means, fde_means, collision_rates = [], [], []

    episodes = 0
    eval_indices = scene_indices[: args.eval_episodes]
    progress = tqdm(
        eval_indices,
        total=len(eval_indices),
        desc="  eval",
        leave=False,
        disable=not args.show_progress,
    )
    for scene_idx in progress:
        _, scene, _ = dataset[scene_idx]
        obs = env.reset(scene)
        done = False
        last_info = None

        while not done:
            obs_t = obs.to(device)
            action, _, _, _ = policy.act(obs_t, deterministic=True)
            obs, _, done, last_info = env.step(action)

        reward_means.append(last_info.mean_reward if last_info is not None else 0.0)
        fde_means.append(last_info.mean_fde if last_info is not None else float("inf"))
        collision_rates.append(last_info.collision_rate if last_info is not None else 0.0)
        episodes += 1
        progress.set_postfix(
            episodes=episodes,
            reward=f"{reward_means[-1]:.3f}",
            fde=f"{fde_means[-1]:.3f}",
            col=f"{collision_rates[-1] * 100:.1f}%",
        )
    progress.close()

    if episodes == 0:
        return {"reward": float("-inf"), "fde": float("inf"), "collision_rate": 0.0}

    return {
        "reward": float(np.mean(reward_means)),
        "fde": float(np.mean(fde_means)),
        "collision_rate": float(np.mean(collision_rates)),
    }


def save_checkpoint(policy: ResidualActorCritic, optimizer, args, path: str, step: int, best_reward: float):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "step": step,
            "state_dict": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_reward": best_reward,
            "prior_checkpoint": args.prior_checkpoint,
            "args": vars(args),
        },
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train a PedPy-backed online PPO residual policy.")
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
