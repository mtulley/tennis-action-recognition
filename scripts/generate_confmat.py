import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ----------------------------
# CONFIG
# ----------------------------
PRED_FILE = "./results/inference/output.txt"
PKL_FILE = "./dataset/pose_classification/test_label.pkl"

# EDIT THIS to match your dataset label order exactly
class_names = [
    "backhand_1h"
    "backhand_2h",
    "backhand_slice",
    "forehand_flat", 
    "forehand_openstance",
    "forehand_slice",
    "serve_flat",
    "serve_kick",
    "serve_slice",
    "overhead",
    "forehand_volley",
    "backhand_volley"
]

# =========================
# LOAD PREDICTIONS
# =========================
with open(PRED_FILE, "r") as f:
    y_pred = [line.strip() for line in f]

# =========================
# LOAD GROUND TRUTH
# =========================
with open(PKL_FILE, "rb") as f:
    data = pickle.load(f)

y_true = data[1]  # integer labels

print(y_true)

# =========================
# CONVERT TRUE LABELS → STRINGS
# =========================
y_true = [class_names[i] for i in y_true]

# =========================
# SANITY CHECK
# =========================
assert len(y_true) == len(y_pred), (
    f"Mismatch: {len(y_true)} true vs {len(y_pred)} pred"
)

# =========================
# LABEL SET
# =========================
labels = class_names  # keep fixed order for clean matrix

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_true, y_pred, labels=labels)

# =========================
# PLOT
# =========================
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

plt.figure(figsize=(12, 10))
disp.plot(
    xticks_rotation=45,
    cmap="Blues",
    values_format="d"
)

plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()