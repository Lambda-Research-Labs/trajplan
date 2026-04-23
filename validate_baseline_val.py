"""
validate_baseline_val.py
------------------------
Validate a supervised baseline model on the validation split carved from
`Data/train` using the exact same split logic as `train.py`.

Example
-------
    python validate_baseline_val.py \
        --model transformer \
        --weights checkpoints/transformer.pth \
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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from dataset import TrajectoryDataset
from inference import load_model
from metrics import ade as calc_ade, collision as calc_collision, fde as calc_fde


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate a baseline model on the train.py validation split."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["d_pool", "social_lstm", "autobot", "eq_motion", "transformer"],
    )
    parser.add_argument("--weights", required=True, help="Path to model weights.")
    parser.add_argument("--data_root", default="Data")
    parser.add_argument("--obs_length", type=int, default=9)
    parser.add_argument("--pred_length", type=int, default=12)
    parser.add_argument("--sample", type=float, default=1.0)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--collision_threshold", type=float, default=0.2)
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


def select_eval_indices(dataset, obs_length: int, pred_length: int, max_samples: int, seed: int):
    indices = [
        idx for idx in range(len(dataset))
        if dataset[idx][1].shape[0] >= obs_length + pred_length
    ]
    if 0 < max_samples < len(indices):
        rng = random.Random(seed)
        rng.shuffle(indices)
        indices = indices[:max_samples]
    return indices


@torch.no_grad()
def evaluate(model, dataset, indices, args):
    results = []
    progress = tqdm(
        indices,
        total=len(indices),
        desc="  eval",
        leave=False,
        disable=not args.show_progress,
    )
    for idx in progress:
        scene_id, scene, goal = dataset[idx]
        pred = model.predict(scene, goal).detach().cpu()
        gt = scene[args.obs_length: args.obs_length + args.pred_length].detach().cpu()

        ade_v = calc_ade(pred, gt[:, :pred.shape[1]])
        fde_v = calc_fde(pred, gt[:, :pred.shape[1]])
        col_v = calc_collision(pred, gt, threshold=args.collision_threshold)

        results.append(
            {
                "scene_id": scene_id,
                "ade": ade_v,
                "fde": fde_v,
                "col": col_v,
            }
        )
        mean_ade = float(np.mean([row["ade"] for row in results]))
        mean_fde = float(np.mean([row["fde"] for row in results]))
        mean_col = float(np.mean([row["col"] for row in results]))
        progress.set_postfix(
            scenes=len(results),
            ade=f"{mean_ade:.3f}",
            fde=f"{mean_fde:.3f}",
            col=f"{mean_col * 100:.1f}%",
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

    print(f"Device     : {args.device}")
    print(f"Model      : {args.model}")
    print(f"Weights    : {args.weights}")
    print("Split      : train.py validation split from Data/train")

    print("\nLoading validation split …")
    val_dataset = build_val_dataset(args)
    indices = select_eval_indices(
        val_dataset,
        obs_length=args.obs_length,
        pred_length=args.pred_length,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    print(f"  Val scenes total : {len(val_dataset)}")
    print(f"  Val scenes valid : {len(indices)}")

    print("Loading model …")
    model = load_model(args)

    print("\nRunning validation …")
    results = evaluate(model, val_dataset, indices, args)
    if not results:
        print("No valid scenes evaluated.")
        return

    ades = np.array([row["ade"] for row in results], dtype=np.float64)
    fdes = np.array([row["fde"] for row in results], dtype=np.float64)
    cols = np.array([row["col"] for row in results], dtype=np.float64)

    print("\n" + "=" * 52)
    print(f"  Model         : {args.model}")
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
