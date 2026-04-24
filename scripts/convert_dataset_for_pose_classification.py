"""
convert_dataset_for_pose_classification.py
----------------------------------------
Converts DeepStream bodypose-3D JSON skeleton files into the NumPy / pickle
format required by NVIDIA TAO PoseClassificationNet (ST-GCN, nvidia layout).

Expected dataset layout
-----------------------
dataset/skeletons_json/
    backhand_1h/
        p1_backhand_s1.json
        p2_backhand_s1.json
        ...
    backhand_2h/
        ...
    (12 action folders total)

Output layout
-------------
dataset/pose_classification/
    train_data.npy        shape (N_train, 3, 300, 34, 1)
    train_label.pkl       [ [name, ...], [class_id, ...] ]
    val_data.npy          shape (N_val,   3, 300, 34, 1)
    val_label.pkl         [ [name, ...], [class_id, ...] ]
    test_data.npy         shape (N_test,  3, 300, 34, 1)
    test_label.pkl        [ [name, ...], [class_id, ...] ]

Normalisation (per NVIDIA docs)
--------------------------------
  normalised_joint = (joint_xyz - root_xyz) / focal_length
  root joint = joint index 0 (pelvis)
  focal_length = 1200.0  (default for 1080p; override with --focal_length)

Train / val / test split
------------------------
The split is always by actor index so that whole subjects end up
exclusively in one split (prevents data leakage).
  - train: actors p(1)            … p(val_actor-1)    (default p1–p39,  ~71 %)
  - val:   actors p(val_actor)  … p(test_actor-1)   (default p40–p49, ~18 %)
  - test:  actors p(test_actor) … p(num_actors)      (default p50–p55, ~11 %)

Usage
-----
python convert_dataset_for_pose_classification.py \
    --input_dir  dataset/skeletons_json \
    --output_dir dataset/pose_classification \
    [--focal_length 1200.0] \
    [--max_seq_len 300] \
    [--min_seq_len 10] \
    [--val_actor 40] \
    [--test_actor 50] \
    [--num_actors 55]
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from collections import Counter

import numpy as np

# ── Label map ────────────────────────────────────────────────────────────────
# Directory-name  →  (display name, class index)
ACTION_MAP = {
    "backhand_1h":        ("Backhand with one hand",  0),
    "backhand_2h":        ("Backhand with two hands", 0),
    "backhand_slice":     ("Backhand slice",          0),
    "backhand_volley":    ("Backhand volley",         0),
    "forehand_flat":      ("Forehand flat",           1),
    "forehand_open":      ("Forehand open stance",    1),
    "forehand_slice":     ("Forehand slice",          1),
    "forehand_volley":    ("Forehand volley",         1),
    "overhead":           ("Overhead",                2),
    "serve_flat":         ("Flat service",            2),
    "serve_kick":         ("Kick service",           2),
    "serve_slice":        ("Slice service",          2),
}

NUM_JOINTS   = 34   # NVIDIA bodypose format
NUM_CHANNELS = 3    # x, y, z


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_dominant_object_id(data):
    ids = []
    for entry in data:
        batches = entry.get("batches", [])
        if batches:
            for obj in batches[0].get("objects", []):
                ids.append(obj.get("object_id"))
    most_common = Counter(ids).most_common(1)
    return most_common[0][0] if most_common else None


def parse_actor_index(filename: str) -> int:
    """Extract numeric actor index from filename, e.g. 'p12_backhand_s1' → 12."""
    name = Path(filename).stem          # strip extension
    actor_part = name.split("_")[0]     # 'p12'
    return int(actor_part.lstrip("p"))


def load_pose3d_sequence(json_path: str) -> np.ndarray | None:
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  [WARN] Cannot read {json_path}: {exc}", file=sys.stderr)
        return None

    # Find the dominant object_id (the main athlete)
    id_counts = Counter()
    for entry in data:
        batches = entry.get("batches", [])
        if batches:
            for obj in batches[0].get("objects", []):
                id_counts[obj.get("object_id")] += 1
    if not id_counts:
        return None
    dominant_id = id_counts.most_common(1)[0][0]

    # Sort frames by frame_num to ensure correct temporal order
    frames_xyz = []
    for entry in sorted(data, key=lambda e: e["batches"][0]["frame_num"]
                        if e.get("batches") else 0):
        batches = entry.get("batches", [])
        if not batches:
            continue
        objects = batches[0].get("objects", [])

        # Find the dominant athlete in this frame
        obj = next((o for o in objects if o.get("object_id") == dominant_id), None)
        if obj is None:
            continue

        raw = obj.get("pose3d")
        if raw is None or len(raw) < NUM_JOINTS * 4:
            continue

        raw = np.array(raw, dtype=np.float32).reshape(NUM_JOINTS, 4)
        frames_xyz.append(raw[:, :3])

    if not frames_xyz:
        return None

    return np.stack(frames_xyz, axis=0)  # (T, 34, 3)


def normalise(seq: np.ndarray, focal_length: float) -> np.ndarray:
    """
    Normalise pose sequence.
      1. Make joints relative to root (joint 0 = pelvis).
      2. Divide by focal_length.
    Input:  (T, V, 3)
    Output: (T, V, 3)
    """
    root = seq[:, 0:1, :]          # (T, 1, 3)
    seq  = (seq - root) / focal_length
    return seq


def to_nctvm(seq: np.ndarray, max_len: int) -> np.ndarray:
    """
    Convert (T, V, C) → (C, max_len, V, 1) with zero-padding / truncation.
    """
    T = seq.shape[0]
    out = np.zeros((NUM_CHANNELS, max_len, NUM_JOINTS, 1), dtype=np.float32)
    T_use = min(T, max_len)
    # seq: (T, V, C) → transpose to (C, T, V)
    seq_t = seq[:T_use].transpose(2, 0, 1)   # (C, T_use, V)
    out[:, :T_use, :, 0] = seq_t
    return out


def assign_split(actor_idx: int, val_actor: int, test_actor: int) -> str:
    """Return 'train', 'val', or 'test' for a given actor index."""
    if actor_idx < val_actor:
        return "train"
    elif actor_idx < test_actor:
        return "val"
    else:
        return "test"


# ── Main ──────────────────────────────────────────────────────────────────────

def build_dataset(
    input_dir:    str,
    output_dir:   str,
    focal_length: float = 1200.0,
    max_seq_len:  int   = 300,
    min_seq_len:  int   = 10,
    val_actor:    int   = 40,
    test_actor:   int   = 50,
    num_actors:   int   = 55,
):
    """
    Actor split boundaries (inclusive):
      train : p1            … p(val_actor-1)
      val   : p(val_actor)  … p(test_actor-1)
      test  : p(test_actor) … p(num_actors)
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Split boundaries:")
    print(f"  train : p1  – p{val_actor - 1}")
    print(f"  val   : p{val_actor} – p{test_actor - 1}")
    print(f"  test  : p{test_actor} – p{num_actors}")
    print()

    buckets = {
        "train": ([], [], []),
        "val":   ([], [], []),
        "test":  ([], [], []),
    }

    missing_actions = []

    for action_dir_name, (action_display, class_id) in ACTION_MAP.items():
        action_dir = input_dir / action_dir_name
        if not action_dir.is_dir():
            missing_actions.append(action_dir_name)
            print(f"[WARN] Directory not found, skipping: {action_dir}")
            continue

        json_files = sorted(action_dir.glob("*.json"))
        print(f"[{action_dir_name}]  class={class_id}  files={len(json_files)}")

        for jf in json_files:
            seq_raw = load_pose3d_sequence(str(jf))
            if seq_raw is None:
                print(f"  [SKIP] {jf.name}  (no valid frames)")
                continue

            if seq_raw.shape[0] < min_seq_len:
                print(f"  [SKIP] {jf.name}  (only {seq_raw.shape[0]} frames < min {min_seq_len})")
                continue

            # Normalise
            seq_norm = normalise(seq_raw, focal_length)

            # Convert to (C, T, V, M) ready for stacking
            sample = to_nctvm(seq_norm, max_seq_len)   # (3, 300, 34, 1)

            # Determine split by actor index
            try:
                actor_idx = parse_actor_index(jf.name)
            except ValueError:
                print(f"  [WARN] Cannot parse actor from '{jf.name}', assigning to train.")
                actor_idx = 1

            split = assign_split(actor_idx, val_actor, test_actor)
            sample_name = jf.stem   # e.g. "p1_backhand_s1"

            data_list, names_list, labels_list = buckets[split]
            data_list.append(sample)
            names_list.append(sample_name)
            labels_list.append(class_id)

    # ── Save ──────────────────────────────────────────────────────────────────
    def save_split(split_name):
        data_list, names, labels = buckets[split_name]
        if not data_list:
            print(f"[WARN] No samples for '{split_name}' split — skipping.")
            return

        arr = np.stack(data_list, axis=0)       # (N, 3, 300, 34, 1)
        npy_path = output_dir / f"{split_name}_data.npy"
        pkl_path = output_dir / f"{split_name}_label.pkl"

        np.save(str(npy_path), arr)
        with open(str(pkl_path), "wb") as f:
            pickle.dump([names, labels], f)

        print(f"\n✓  {split_name}: {arr.shape[0]} sequences  →  {npy_path.name}, {pkl_path.name}")
        print(f"   array shape : {arr.shape}  dtype={arr.dtype}")

        # Class distribution
        from collections import Counter
        dist = Counter(labels)
        label_names = {v: k for k, (_, v) in ACTION_MAP.items()}
        for cid in sorted(dist):
            print(f"   class {cid:2d} ({label_names.get(cid, '?'):>12s}): {dist[cid]:4d} samples")

    print("\n" + "="*60)
    for split_name in ("train", "val", "test"):
        save_split(split_name)

    # ── Write TAO spec template ────────────────────────────────────────────────
    spec_path = output_dir / "pose_classification_spec.yaml"
    spec = f"""model:
  model_type: ST-GCN
  pretrained_model_path: "/path/to/pretrained_model.pth"   # optional
  input_channels: {NUM_CHANNELS}
  dropout: 0.5
  graph_layout: "nvidia"
  graph_strategy: "spatial"
  edge_importance_weighting: True

dataset:
  train_dataset:
    data_path: "{output_dir.resolve()}/train_data.npy"
    label_path: "{output_dir.resolve()}/train_label.pkl"
  val_dataset:
    data_path: "{output_dir.resolve()}/val_data.npy"
    label_path: "{output_dir.resolve()}/val_label.pkl"
  num_classes: {len(ACTION_MAP)}
  label_map:
{chr(10).join(f"    {name}: {idx}" for name, (_, idx) in ACTION_MAP.items())}
  batch_size: 16
  num_workers: 4
  random_choose: False
  random_move: False
  window_size: -1

train:
  optim:
    lr: 0.1
    momentum: 0.9
    nesterov: True
    weight_decay: 0.0001
    lr_scheduler: "MultiStep"
    lr_steps:
      - 30
      - 60
    lr_decay: 0.1
  num_epochs: 70
  checkpoint_interval: 5
"""
    with open(str(spec_path), "w") as f:
        f.write(spec)
    print(f"\n✓  TAO spec template written to: {spec_path.name}")

    if missing_actions:
        print(f"\n[WARN] The following action folders were missing: {missing_actions}")

    test_data_path  = output_dir.resolve() / "test_data.npy"
    test_label_path = output_dir.resolve() / "test_label.pkl"
    print("\nDone.")
    print("\nTo train:")
    print(f"  tao model pose_classification train \\")
    print(f"      -e {spec_path} \\")
    print(f"      -r ./results \\")
    print(f"      -k $KEY")
    print("\nTo evaluate on the held-out test set:")
    print(f"  tao model pose_classification evaluate \\")
    print(f"      -e {spec_path} \\")
    print(f"      -r ./results \\")
    print(f"      -k $KEY \\")
    print(f"      evaluate.checkpoint=<trained_model.tlt> \\")
    print(f"      evaluate.test_dataset.data_path={test_data_path} \\")
    print(f"      evaluate.test_dataset.label_path={test_label_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert DeepStream skeleton JSONs to PoseClassificationNet format."
    )
    parser.add_argument(
        "--input_dir", required=True,
        help="Root directory containing 12 action sub-folders of JSON files "
             "(e.g. dataset/skeletons_json)"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Where to write train/val/test .npy and .pkl files "
             "(e.g. dataset/pose_classification)"
    )
    parser.add_argument(
        "--focal_length", type=float, default=1200.0,
        help="Camera focal length for 3D pose normalisation (default: 1200.0 for 1080p)"
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=300,
        help="Maximum sequence length T (default: 300 = 10 s @ 30 FPS)"
    )
    parser.add_argument(
        "--min_seq_len", type=int, default=10,
        help="Minimum number of frames required to keep a sequence (default: 10)"
    )
    parser.add_argument(
        "--val_actor", type=int, default=40,
        help="First actor index assigned to val split; p1..p(val_actor-1) → train "
             "(default: 40)"
    )
    parser.add_argument(
        "--test_actor", type=int, default=50,
        help="First actor index assigned to test split; p(val_actor)..p(test_actor-1) → val "
             "(default: 50)"
    )
    parser.add_argument(
        "--num_actors", type=int, default=55,
        help="Total number of actors (default: 55)"
    )
    args = parser.parse_args()

    if not (1 < args.val_actor < args.test_actor <= args.num_actors):
        parser.error(
            f"Must satisfy: 1 < val_actor ({args.val_actor}) < "
            f"test_actor ({args.test_actor}) <= num_actors ({args.num_actors})"
        )

    build_dataset(
        input_dir    = args.input_dir,
        output_dir   = args.output_dir,
        focal_length = args.focal_length,
        max_seq_len  = args.max_seq_len,
        min_seq_len  = args.min_seq_len,
        val_actor    = args.val_actor,
        test_actor   = args.test_actor,
        num_actors   = args.num_actors,
    )


if __name__ == "__main__":
    main()