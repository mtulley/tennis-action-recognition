import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ----------------------------
# CONFIG
# ----------------------------
PRED_FILE = "./results/pose_classification/v2.1/inference/predictions.txt"
PKL_FILE  = "./dataset/pose_classification/v1.5_correct-focal-length/test_label.pkl"
OUT_FILE  = "./results/pose_classification/v2.1/inference/confusion_matrix.png"

# Class names in index order (must match ACTION_MAP in convert_dataset script)
CLASS_NAMES = [
    "backhand_1h",       # 0
    "backhand_2h",       # 1
    "backhand_slice",    # 2
    "backhand_volley",   # 3
    "forehand_flat",     # 4
    "forehand_open",     # 5
    "forehand_slice",    # 6
    "forehand_volley",   # 7
    "overhead",          # 8
    "serve_flat",        # 9
    "serve_kick",        # 10
    "serve_slice",       # 11
]

# =========================
# LOAD PREDICTIONS
# =========================
# TAO inference writes integer class indices, one per line e.g. "0", "1", ...
with open(PRED_FILE, "r") as f:
    raw_preds = [line.strip() for line in f if line.strip()]

y_pred = raw_preds

# =========================
# LOAD GROUND TRUTH
# =========================
with open(PKL_FILE, "rb") as f:
    data = pickle.load(f)

# TAO label pkl format: [names_list, labels_list]
y_true_int = data[1]

# Convert integer labels → class name strings
y_true = [CLASS_NAMES[i] for i in y_true_int]

# =========================
# SANITY CHECK
# =========================
assert len(y_true) == len(y_pred), (
    f"Mismatch: {len(y_true)} ground-truth vs {len(y_pred)} predictions. "
    f"Check that inference was run on the correct dataset."
)

print(f"Total samples : {len(y_true)}")
print(f"Correct       : {sum(t == p for t, p in zip(y_true, y_pred))}")
print(f"Accuracy      : {sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true):.4f}")

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)

# =========================
# PLOT
# =========================
fig, ax = plt.subplots(figsize=(13, 11))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(
    ax=ax,
    xticks_rotation=45,
    cmap="Blues",
    values_format="d",
    colorbar=False,
)

ax.set_title("Pose Classification — Confusion Matrix", fontsize=14, pad=15)
ax.set_xlabel("Predicted label", fontsize=11)
ax.set_ylabel("True label", fontsize=11)

plt.tight_layout()

import os
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
plt.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
print(f"\n✓ Confusion matrix saved to: {OUT_FILE}")