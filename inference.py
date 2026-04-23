"""
inference.py
------------
Evaluate any trained model on TrajNet++ test data and report ADE, FDE,
and Collision Rate.

Usage examples
--------------
    # Evaluate a trained transformer:
    python inference.py \\
        --model transformer \\
        --weights checkpoints/transformer.pth \\
        --data_root DATA_BLOCK/trajdata

    # Evaluate D-Pool (requires pre-trained weights):
    python inference.py \\
        --model d_pool \\
        --weights baselines/weights/DPool/d_pool.state \\
        --data_root DATA_BLOCK/trajdata

    # Use all test scenes:
    python inference.py --model eq_motion \\
        --weights baselines/weights/EqMotion/my_checkpoint.pth \\
        --max_samples -1

Output
------
Prints per-scene metrics and a final summary table to stdout.
Optionally saves results as a tab-separated CSV (--output_file).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from dataset import TrajectoryDataset
from metrics import ade as calc_ade, fde as calc_fde, collision as calc_collision


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(args):
    """Instantiate and return a model wrapper from args."""
    common = dict(
        obs_length=args.obs_length,
        pred_length=args.pred_length,
        device=args.device,
    )

    if args.model == "d_pool":
        from models.d_pool import DPoolModel
        return DPoolModel(
            weights_path=args.weights,
            n_neighbours=4,
            embedding_dim=64,
            hidden_dim=128,
            pool_hidden_dim=256,
            pool_dim=256,
            **common,
        )

    elif args.model == "social_lstm":
        from models.social_lstm import SocialLSTMModel
        return SocialLSTMModel(
            weights_path=args.weights,
            embedding_dim=64,
            hidden_dim=128,
            grid_size=4,
            neighbourhood_size=2.0,
            pool_dim=128,
            **common,
        )

    elif args.model == "autobot":
        from models.autobot import AutobotModel
        return AutobotModel(
            weights_path=args.weights,
            d_k=128, M=5, c=1,
            L_enc=2, L_dec=2,
            num_heads=8, tx_hidden=256, dropout=0.1,
            **common,
        )

    elif args.model == "eq_motion":
        from models.eq_motion import EqMotionModel
        return EqMotionModel(
            weights_path=args.weights,
            node_dim=64, edge_dim=64, hidden_dim=64, n_layers=4,
            **common,
        )

    elif args.model == "transformer":
        from models.transformer import TransformerModel
        return TransformerModel(
            weights_path=args.weights,
            d_model=128, num_heads=8, ff_dim=256,
            L_enc=3, L_dec=2, dropout=0.0,   # no dropout at inference
            **common,
        )

    else:
        raise ValueError(f"Unknown model: {args.model!r}")


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, dataset, obs_len: int, pred_len: int, max_samples: int):
    """Run model on dataset and collect per-scene metrics."""
    results = []

    indices = list(range(len(dataset)))
    if 0 < max_samples < len(indices):
        import random; random.shuffle(indices)
        indices = indices[:max_samples]

    for idx in tqdm(indices, desc="Evaluating"):
        scene_id, scene, goal = dataset[idx]

        # Skip too-short scenes
        if scene.shape[0] < obs_len + pred_len:
            continue

        pred = model.predict(scene, goal)      # [T_pred, 1, 2]
        gt   = scene[obs_len: obs_len + pred_len]   # [T_pred, N, 2]

        # Expand pred to match gt shape for collision check
        pred_full = pred                       # [T_pred, 1, 2]
        if gt.shape[1] > 1:
            # collision checks pred ego vs gt neighbours
            pass

        ade_v  = calc_ade(pred_full, gt[:, :pred_full.shape[1]])
        fde_v  = calc_fde(pred_full, gt[:, :pred_full.shape[1]])
        # For collision: need ego pred [T,1,2] vs all gt [T,N,2]
        col_v  = calc_collision(pred_full, gt, threshold=0.2)

        results.append({
            "scene_id": scene_id,
            "ade":      ade_v,
            "fde":      fde_v,
            "col":      col_v,
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trajectory prediction model.")

    # Model
    parser.add_argument("--model", required=True,
                        choices=["d_pool", "social_lstm", "autobot",
                                 "eq_motion", "transformer"])
    parser.add_argument("--weights", default=None,
                        help="Path to model weights (.pth / .state file)")

    # Data
    parser.add_argument("--data_root",   default="DATA_BLOCK/trajdata")
    parser.add_argument("--data_part",   default="test",
                        choices=["train", "test", "secret"])
    parser.add_argument("--max_samples", type=int, default=1000,
                        help="Max scenes to evaluate (-1 = all)")

    # Sequence lengths
    parser.add_argument("--obs_length",  type=int, default=9)
    parser.add_argument("--pred_length", type=int, default=12)

    # Misc
    parser.add_argument("--device",       default=None)
    parser.add_argument("--output_file",  default=None,
                        help="Optional path to save results as TSV")

    args = parser.parse_args()

    # Auto device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device     : {args.device}")
    print(f"Model      : {args.model}")
    print(f"Weights    : {args.weights}")
    print(f"Data part  : {args.data_part}")
    print(f"Max samples: {args.max_samples if args.max_samples > 0 else 'all'}")

    # Load dataset
    print("\nLoading dataset …")
    dataset = TrajectoryDataset(
        data_root=args.data_root,
        data_part=args.data_part,
    )
    print(f"  Scenes: {len(dataset)}")

    # Load model
    print("Loading model …")
    model = load_model(args)

    # Run evaluation
    print("\nRunning evaluation …")
    results = evaluate(
        model, dataset,
        obs_len=args.obs_length,
        pred_len=args.pred_length,
        max_samples=args.max_samples,
    )

    if not results:
        print("No valid scenes evaluated.")
        return

    # Aggregate
    import numpy as np
    ades = [r["ade"] for r in results]
    fdes = [r["fde"] for r in results]
    cols = [r["col"] for r in results]

    print("\n" + "=" * 52)
    print(f"  Model         : {args.model}")
    print(f"  Scenes eval'd : {len(results)}")
    print(f"  ADE   (↓)     : {np.mean(ades):.4f} m")
    print(f"  FDE   (↓)     : {np.mean(fdes):.4f} m")
    print(f"  Col.  (↓)     : {np.mean(cols)*100:.2f} %")
    print("=" * 52)

    # Optional: save TSV
    if args.output_file:
        header = "scene_id\tade\tfde\tcol\n"
        rows = "\n".join(
            f"{r['scene_id']}\t{r['ade']:.6f}\t{r['fde']:.6f}\t{r['col']}"
            for r in results
        )
        Path(args.output_file).write_text(header + rows + "\n")
        print(f"\nResults saved → {args.output_file}")


if __name__ == "__main__":
    main()
