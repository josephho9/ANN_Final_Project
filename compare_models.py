"""
compare_models.py

Train and evaluate all four models plus the threshold baseline,
then print a comparison table and save a bar chart.

Usage:
    python compare_models.py
    python compare_models.py --quick    # 5 epochs each, for testing
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import config
from src.train import train
from src.evaluate import evaluate_model


MODELS = ["mlp", "lstm", "transformer", "cnn_lstm"]


def compute_angle(a, b, c):
    """Angle at joint B. Returns degrees."""
    ba = a - b
    bc = c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))


def run_threshold_baseline():
    """Threshold baseline: back alignment >= 160 -> good form.
    No training, just geometry.
    """
    correct   = np.load(config.CORRECT_NPY).reshape(50, 150, 33, 2)
    incorrect = np.load(config.INCORRECT_NPY).reshape(50, 150, 33, 2)

    features, labels = [], []
    for seq in correct:
        bottom = seq[:75]
        elbows = [compute_angle(f[12], f[14], f[16]) for f in bottom]
        backs  = [compute_angle(f[12], f[24], f[28]) for f in bottom]
        features.append([np.mean(elbows), np.mean(backs)])
        labels.append(1)
    for seq in incorrect:
        bottom = seq[:75]
        elbows = [compute_angle(f[12], f[14], f[16]) for f in bottom]
        backs  = [compute_angle(f[12], f[24], f[28]) for f in bottom]
        features.append([np.mean(elbows), np.mean(backs)])
        labels.append(0)

    features = np.array(features)
    labels   = np.array(labels)

    # use same test split as the ML models
    _, X_temp, _, y_temp = train_test_split(
        features, labels,
        test_size=(config.VAL_SPLIT + config.TEST_SPLIT),
        random_state=config.RANDOM_SEED, stratify=labels,
    )
    val_ratio = config.VAL_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT)
    _, X_test, _, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_ratio),
        random_state=config.RANDOM_SEED, stratify=y_temp,
    )

    # single threshold rule — back alignment column is index 1
    preds = (X_test[:, 1] >= config.BACK_ALIGNMENT_MIN).astype(int)

    return {
        "accuracy":  accuracy_score(y_test, preds),
        "f1":        f1_score(y_test, preds, average="binary", zero_division=0),
        "precision": precision_score(y_test, preds, average="binary", zero_division=0),
        "recall":    recall_score(y_test, preds, average="binary", zero_division=0),
    }


def run_all(epochs=config.NUM_EPOCHS, quick=False):
    results = {}

    # threshold baseline first (no training needed)
    print("\n[compare] running threshold baseline...")
    results["threshold"] = run_threshold_baseline()

    # train each ML model
    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"  training: {model_name.upper()}")
        print(f"{'='*60}")
        try:
            train(model_name=model_name, epochs=epochs if not quick else 5)
            res = evaluate_model(model_name)
            results[model_name] = res
        except Exception as e:
            print(f"[WARN] {model_name} failed: {e}")

    # print summary table
    print("\n" + "=" * 60)
    print(f"{'Model':<15} {'Accuracy':>10} {'F1':>8} {'Precision':>12} {'Recall':>10}")
    print("-" * 60)
    for name, res in results.items():
        print(f"{name:<15} {res['accuracy']:>10.4f} {res['f1']:>8.4f} "
              f"{res['precision']:>12.4f} {res['recall']:>10.4f}")

    # save bar chart
    # this code was AI generated
    if results:
        metrics = ["accuracy", "f1", "precision", "recall"]
        x = np.arange(len(metrics))
        n = len(results)
        width = 0.8 / n

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor("#1a1a1a")
        ax.set_facecolor("#2a2a2a")

        colors = ["#95a5a6", "#f39c12", "#e74c3c", "#3498db", "#2ecc71"]
        for i, (name, res) in enumerate(results.items()):
            vals = [res[m] for m in metrics]
            ax.bar(x + i * width, vals, width, label=name, color=colors[i % len(colors)])

        ax.set_xticks(x + width * (n - 1) / 2)
        ax.set_xticklabels([m.capitalize() for m in metrics], color="white")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score", color="white")
        ax.set_title("Model Comparison", color="white", fontsize=14)
        ax.tick_params(colors="gray")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        ax.legend(facecolor="#333", labelcolor="white")

        results_dir = Path(config.RESULTS_DIR)
        results_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(results_dir / "model_comparison.png", dpi=150,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"\n[compare] saved chart -> {results_dir}/model_comparison.png")
        plt.close()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--quick", action="store_true", help="run 5 epochs per model for testing")
    args = parser.parse_args()
    run_all(args.epochs, args.quick)
