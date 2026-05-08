"""
Data-driven threshold baseline.

Instead of hardcoded angle thresholds, this baseline:
  1. Splits the data the same way as the ML models (70/15/15)
  2. On the training set, computes the mean elbow and back angle
     for correct clips and incorrect clips separately
  3. Sets the decision threshold = midpoint between the two means
  4. Classifies test clips using those learned thresholds
  5. Compares against the LSTM checkpoint if available

Run:
    python threshold_baseline.py
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

import config

# ── Angle computation ─────────────────────────────────────────────────────────

def compute_angle(a, b, c):
    ba, bc = a - b, c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))


def clip_features(seq):
    """
    seq: (150, 33, 2)
    Returns mean elbow angle and mean back alignment during bottom phase (frames 0–74).
    """
    bottom = seq[:seq.shape[0] // 2]
    elbows = [compute_angle(f[12], f[14], f[16]) for f in bottom]
    backs  = [compute_angle(f[12], f[24], f[28]) for f in bottom]
    return np.mean(elbows), np.mean(backs)


# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading dataset …")
correct   = np.load(config.CORRECT_NPY).reshape(50, 150, 33, 2)
incorrect = np.load(config.INCORRECT_NPY).reshape(50, 150, 33, 2)

features = []
labels   = []

for seq in correct:
    features.append(clip_features(seq))
    labels.append(1)
for seq in incorrect:
    features.append(clip_features(seq))
    labels.append(0)

features = np.array(features)   # (100, 2) — [elbow_mean, back_mean]
labels   = np.array(labels)     # (100,)

# ── Same train/test split as ML models ───────────────────────────────────────

X_train, X_temp, y_train, y_temp = train_test_split(
    features, labels,
    test_size=(config.VAL_SPLIT + config.TEST_SPLIT),
    random_state=config.RANDOM_SEED,
    stratify=labels,
)
val_ratio = config.VAL_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT)
_, X_test, _, y_test = train_test_split(
    X_temp, y_temp,
    test_size=(1 - val_ratio),
    random_state=config.RANDOM_SEED,
    stratify=y_temp,
)

# ── Learn thresholds from training set ───────────────────────────────────────

correct_train   = X_train[y_train == 1]
incorrect_train = X_train[y_train == 0]

elbow_correct_mean   = correct_train[:, 0].mean()
elbow_incorrect_mean = incorrect_train[:, 0].mean()
elbow_threshold      = (elbow_correct_mean + elbow_incorrect_mean) / 2

back_correct_mean   = correct_train[:, 1].mean()
back_incorrect_mean = incorrect_train[:, 1].mean()
back_threshold      = (back_correct_mean + back_incorrect_mean) / 2

print(f"\nLearned thresholds (from training set):")
print(f"  Elbow — correct mean: {elbow_correct_mean:.1f}°  incorrect mean: {elbow_incorrect_mean:.1f}°  → threshold: {elbow_threshold:.1f}°")
print(f"  Back  — correct mean: {back_correct_mean:.1f}°  incorrect mean: {back_incorrect_mean:.1f}°  → threshold: {back_threshold:.1f}°")

# ── Classify test set ─────────────────────────────────────────────────────────
#
# A clip is "correct" if it looks more like a correct clip on either joint.
# We try three voting strategies and pick the best.

def predict(X, elbow_thresh, back_thresh, strategy="either"):
    preds = []
    for elbow, back in X:
        # Closer to correct = lower elbow angle (more bent) and higher back angle (straighter)
        elbow_vote = 1 if elbow <= elbow_thresh else 0
        back_vote  = 1 if back  >= back_thresh  else 0

        if strategy == "either":
            preds.append(1 if (elbow_vote + back_vote) >= 1 else 0)
        elif strategy == "both":
            preds.append(1 if (elbow_vote + back_vote) == 2 else 0)
        elif strategy == "elbow_only":
            preds.append(elbow_vote)
        elif strategy == "back_only":
            preds.append(back_vote)
    return np.array(preds)


print(f"\n{'='*55}")
print(f"{'Strategy':<20} {'Accuracy':>10} {'F1':>10}")
print(f"{'-'*55}")

best_acc, best_f1, best_strategy = 0, 0, ""
for strategy in ["either", "both", "elbow_only", "back_only"]:
    preds = predict(X_test, elbow_threshold, back_threshold, strategy)
    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds, average="binary", zero_division=0)
    print(f"  {strategy:<18} {acc:>10.4f} {f1:>10.4f}")
    if acc > best_acc:
        best_acc, best_f1, best_strategy = acc, f1, strategy

print(f"{'='*55}")
print(f"\nBest strategy: {best_strategy}  (acc={best_acc:.4f}  f1={best_f1:.4f})")

best_preds = predict(X_test, elbow_threshold, back_threshold, best_strategy)
print("\nClassification Report:")
print(classification_report(y_test, best_preds, target_names=["Poor Form", "Good Form"]))

# ── Compare against LSTM if checkpoint exists ─────────────────────────────────

from pathlib import Path
lstm_results_path = Path(config.RESULTS_DIR) / "lstm_results.npy"
if lstm_results_path.exists():
    lstm = np.load(lstm_results_path, allow_pickle=True).item()
    print(f"{'='*55}")
    print(f"{'Model':<25} {'Accuracy':>10} {'F1':>10}")
    print(f"{'-'*55}")
    print(f"  {'Threshold baseline':<23} {best_acc:>10.4f} {best_f1:>10.4f}")
    print(f"  {'LSTM':<23} {lstm['accuracy']:>10.4f} {lstm['f1']:>10.4f}")
    lstm_gain = lstm['accuracy'] - best_acc
    print(f"{'='*55}")
    print(f"\nLSTM improvement over threshold baseline: {lstm_gain:+.4f} accuracy")
else:
    print("\n(Run `python -m src.evaluate --model lstm` first to compare against LSTM)")
