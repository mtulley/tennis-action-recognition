"""
visualize_skeletons.py
----------------------
Overlays DeepStream bodypose-3D JSON skeletons onto a video using the 2.5D
(pose25d) coordinates, which are already in pixel space.

Usage:
    python visualize_skeletons.py \
        --video  p1_backhand_s1.mp4 \
        --json   p1_backhand_s1.json \
        --output p1_backhand_s1_overlay.mp4
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

import cv2
import numpy as np

# ── NVIDIA 34-keypoint skeleton connections ───────────────────────────────────
# Each tuple is (joint_a, joint_b) drawn as a bone
BONES = [
    # Spine / torso
    (0, 1), (0, 2), (1, 3), (2, 4),        # pelvis → hips
    (3, 5), (4, 6),                          # hips → knees
    (5, 7), (6, 8),                          # knees → ankles
    (7, 9), (8, 10),                         # ankles → feet
    (0, 11),                                 # pelvis → spine
    (11, 12), (12, 13),                      # spine → neck → head
    # Arms
    (13, 14), (13, 15),                      # neck → shoulders
    (14, 16), (15, 17),                      # shoulders → elbows
    (16, 18), (17, 19),                      # elbows → wrists
    # Hands (low confidence — shown dimly)
    (18, 20), (18, 21), (18, 22),
    (19, 23), (19, 24), (19, 25),
]

JOINT_COLOR  = (0,   255, 100)   # green
BONE_COLOR   = (255, 180,   0)   # amber
HAND_COLOR   = (100, 100, 255)   # blue-ish for uncertain hand joints
HAND_JOINTS  = set(range(20, 34))
CONF_THRESH  = 0.5


def load_json(json_path):
    with open(json_path) as f:
        return json.load(f)


def get_dominant_id(data):
    counts = Counter()
    for entry in data:
        for obj in entry.get("batches", [{}])[0].get("objects", []):
            counts[obj.get("object_id")] += 1
    return counts.most_common(1)[0][0] if counts else None


def index_by_frame(data, dominant_id):
    """Returns dict: frame_num → list of (x, y, conf) per joint (pose25d)."""
    frames = {}
    for entry in data:
        batch = entry.get("batches", [{}])[0]
        fnum  = batch.get("frame_num")
        for obj in batch.get("objects", []):
            if obj.get("object_id") != dominant_id:
                continue
            raw = obj.get("pose25d")
            if not raw or len(raw) < 34 * 4:
                continue
            arr = np.array(raw, dtype=np.float32).reshape(34, 4)
            # pose25d columns: x_pixel, y_pixel, z_relative, confidence
            frames[fnum] = arr
    return frames


def draw_skeleton(frame_img, joints, src_w=1280, src_h=720):
    """joints: (34, 4) array — x, y, z, conf in pixel space."""
    h, w = frame_img.shape[:2]
    scale_x = w / src_w
    scale_y = h / src_h

    for i, j in BONES:
        if i >= len(joints) or j >= len(joints):
            continue
        ci, cj = joints[i, 3], joints[j, 3]
        if ci < CONF_THRESH or cj < CONF_THRESH:
            continue
        is_hand = (i in HAND_JOINTS or j in HAND_JOINTS)
        color = HAND_COLOR if is_hand else BONE_COLOR
        p1 = (int(joints[i, 0] * scale_x), int(joints[i, 1] * scale_y))
        p2 = (int(joints[j, 0] * scale_x), int(joints[j, 1] * scale_y))
        cv2.line(frame_img, p1, p2, color, 2, cv2.LINE_AA)

    for idx, (x, y, _, conf) in enumerate(joints):
        if conf < CONF_THRESH:
            continue
        color = HAND_COLOR if idx in HAND_JOINTS else JOINT_COLOR
        radius = 3 if idx in HAND_JOINTS else 5
        cv2.circle(frame_img, (int(x * scale_x), int(y * scale_y)), radius, color, -1, cv2.LINE_AA)


def overlay(video_path, json_path, output_path):
    data        = load_json(json_path)
    dominant_id = get_dominant_id(data)
    skeleton_by_frame = index_by_frame(data, dominant_id)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video {video_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Video resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    matched   = 0

    while True:
        ret, img = cap.read()
        if not ret:
            break

        if frame_idx in skeleton_by_frame:
            draw_skeleton(img, skeleton_by_frame[frame_idx])
            matched += 1

        # Frame counter overlay
        cv2.putText(img, f"frame {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(img)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Done. {matched}/{frame_idx} frames had skeleton data → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",  required=True)
    parser.add_argument("--json",   required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    overlay(args.video, args.json, args.output)


if __name__ == "__main__":
    main()