"""
dataset.py
----------
Dataloader for TrajNet++ trajectory prediction.

Scene format  : TrajNet++ .ndjson files read via trajnetplusplustools
Dataset output: (scene_id, scene_tensor [T x N x 2], goal_tensor [N x 2])
  - T  = obs_length + pred_length  (default 9 + 12 = 21 frames, 0.4 s apart)
  - N  = number of agents kept after 6-metre proximity filtering
  - axis 2 = (x, y) in world coordinates (metres)

Neighbour filtering
  - drop_distant : keeps every agent whose minimum distance to the primary
                   pedestrian (index 0) is < r metres at any observed frame.
  - Optional stationary-agent removal (any agent that never moves > 2 m).
"""

import os
import math
import pickle
import random
from operator import itemgetter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

# TrajNet++ reader  – install with: pip install trajnetplusplustools
from trajnetplusplustools.trajnetplusplustools import Reader


# ---------------------------------------------------------------------------
# Scene-level helpers
# ---------------------------------------------------------------------------

def drop_distant(xy: np.ndarray, r: float = 6.0):
    """Keep only agents within *r* metres of the primary pedestrian (index 0).

    Parameters
    ----------
    xy : np.ndarray  shape (T, N, 2)
    r  : float  proximity radius in metres

    Returns
    -------
    xy_filtered : np.ndarray  shape (T, N_kept, 2)
    mask        : np.ndarray  bool, shape (N,)
    """
    # Squared distance of every agent from agent-0 at each frame
    dist_sq = np.sum(np.square(xy - xy[:, 0:1]), axis=2)   # (T, N)
    # An agent is kept if it is ever within r metres
    mask = np.nanmin(dist_sq, axis=0) < r ** 2             # (N,)
    return xy[:, mask], mask


def is_stationary(xs, ys, threshold_sq: float = 4.0) -> bool:
    """Return True if the agent never moves more than 2 m from any position."""
    xs = [x for x in xs if not math.isnan(x)]
    ys = [y for y in ys if not math.isnan(y)]
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if (xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2 > threshold_sq:
                return False
    return True


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(data_root: str, subset: str = "/train/",
                 sample: float = 1.0, goals: bool = False):
    """Read all .ndjson scene files in *data_root/subset*.

    Parameters
    ----------
    data_root : str   path to the TrajNet++ DATA_BLOCK/trajdata folder
    subset    : str   one of '/train/', '/test/', '/test_private/'
    sample    : float fraction of scenes to keep (1.0 = all)
    goals     : bool  if True, loads pre-computed destination pkl files

    Returns
    -------
    all_scenes : list of (filename, scene_id, paths)
    all_goals  : dict | None
    """
    all_goals: dict = {}
    all_scenes: list = []

    if sample <= 0 or sample > 1.0:
        raise ValueError("sample must be in the interval (0, 1].")

    folder = Path(data_root) / subset.strip("/")
    if not folder.exists():
        raise FileNotFoundError(f"Could not find dataset split directory: {folder}")

    ndjson_files = sorted(folder.rglob("*.ndjson"))
    if not ndjson_files:
        raise FileNotFoundError(f"No .ndjson files found under {folder}")

    for scene_file in ndjson_files:
        fname = str(scene_file.relative_to(folder).with_suffix(""))
        reader = Reader(str(scene_file), scene_type="paths")
        scenes = list(reader.scenes())

        # The vendored trajnet reader uses random.sample() on dict_keys,
        # which crashes on Python 3.12. Sampling here keeps the caller API
        # unchanged and works for both full and fractional loads.
        if sample < 1.0 and scenes:
            keep_count = min(len(scenes), max(1, int(len(scenes) * sample)))
            scenes = random.sample(scenes, keep_count)

        scenes = [(fname, s_id, s) for s_id, s in scenes]

        if goals:
            goal_path = Path("dest_new") / subset.strip("/") / f"{fname}.pkl"
            with goal_path.open("rb") as handle:
                goal_dict = pickle.load(handle)
            all_goals[fname] = {
                s_id: [goal_dict[path[0].pedestrian] for path in s]
                for _, s_id, s in scenes
            }

        all_scenes += scenes

    return all_scenes, (all_goals if goals else None)


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class TrajectoryDataset(TorchDataset):
    """PyTorch Dataset wrapping TrajNet++ scenes.

    Each item is a tuple:
        (scene_id : int,
         scene    : torch.FloatTensor  [T, N, 2],
         goal     : torch.FloatTensor  [N, 2])

    Attributes
    ----------
    obs_length  : int   number of observed frames   (default 9)
    pred_length : int   number of frames to predict (default 12)
    proximity_r : float neighbour-filtering radius in metres (default 6.0)
    """

    def __init__(
        self,
        data_root: str,
        data_part: str = "train",       # 'train' | 'test' | 'secret'
        sample: float = 1.0,
        goals: bool = False,
        remove_static: bool = False,
        proximity_r: float = 6.0,
        device: str = "cpu",
    ):
        self.proximity_r = proximity_r
        self.remove_static = remove_static
        self.device = torch.device(device)

        subset_map = {
            "train": "/train/",
            "test": "/test/",
            "secret": "/test_private/",
        }
        if data_part not in subset_map:
            raise ValueError(f"data_part must be one of {list(subset_map)}")

        scenes, goal_dict = prepare_data(
            data_root,
            subset=subset_map[data_part],
            sample=sample,
            goals=goals,
        )

        self.all_data = self._preprocess(scenes, goal_dict)

    # ------------------------------------------------------------------
    def _preprocess(self, scenes, goal_dict):
        all_data = []

        for filename, scene_id, paths in scenes:
            # (T, N, 2)  – raw world coordinates
            scene_np = Reader.paths_to_xy(paths)

            # Goal tensor
            if goal_dict is not None:
                scene_goal_np = np.array(goal_dict[filename][scene_id])
            else:
                scene_goal_np = np.zeros((scene_np.shape[1], 2))

            # Drop agents more than proximity_r metres away from primary
            scene_np, mask = drop_distant(scene_np, r=self.proximity_r)
            scene_goal_np = scene_goal_np[mask]

            # Optional: skip scenes containing a stationary agent
            if self.remove_static:
                skip = False
                for agent_path in paths:
                    xs = [row[0] for row in agent_path]
                    ys = [row[1] for row in agent_path]
                    if is_stationary(xs, ys):
                        skip = True
                        break
                if skip:
                    continue

            scene_t = torch.tensor(scene_np, dtype=torch.float32).to(self.device)
            goal_t = torch.tensor(scene_goal_np, dtype=torch.float32).to(self.device)

            all_data.append((scene_id, scene_t, goal_t))

        # Deterministic ordering by scene id
        all_data.sort(key=itemgetter(0))
        return all_data

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.all_data)

    def __getitem__(self, idx):
        """Return (scene_id, scene [T,N,2], goal [N,2])."""
        return self.all_data[idx]


# ---------------------------------------------------------------------------
# Collate function for DataLoader batching
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """Collate a list of (scene_id, scene, goal) into padded batch tensors.

    Scenes in a batch may have different numbers of agents (N varies).
    We pad the N dimension with NaN so downstream models can mask them out.

    Returns
    -------
    scene_ids : list[int]
    scenes    : list[Tensor]   – kept as a list; each is [T, N_i, 2]
    goals     : list[Tensor]   – kept as a list; each is [N_i, 2]
    """
    scene_ids = [item[0] for item in batch]
    scenes    = [item[1] for item in batch]
    goals     = [item[2] for item in batch]
    return scene_ids, scenes, goals
