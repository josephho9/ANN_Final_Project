"""
Ablation study — train and evaluate all models, then print a comparison table.

Usage:
    python compare_models.py
    python compare_models.py --epochs 30 --quick
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

import config
from src.train import train
from src.evaluate import evaluate_model


MODELS = ["mlp", "lstm", "transformer", "cnn_lstm"]


def run_all(epochs: int, quick: bool = False):
    results = {}
    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"  Training: {model_name.upper()}")
        print(f"{'='*60}")
        try:
            train(model_name=model_name, epochs=epochs if not quick else 5)
            res = evaluate_model(model_name)
            results[model_name] = res
        except Exception as e:
            print(f"[WARN] {model_name} failed: {e}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'Model':<15} {'Accuracy':>10} {'F1':>8} {'Precision':>12} {'Recall':>10}")
    print("-" * 60)
    for name, res in results.items():
        print(
            f"{name:<15} {res['accuracy']:>10.4f} {res['f1']:>8.4f} "
            f"{res['precision']:>12.4f} {res['recall']:>10.4f}"
        )

    # ── Bar chart ─────────────────────────────────────────────────────────────
    if results:
        metrics = ["accuracy", "f1", "precision", "recall"]
        x = np.arange(len(metrics))
        width = 0.2
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#1a1a1a")
        ax.set_facecolor("#2a2a2a")

        colors = ["#f39c12", "#e74c3c", "#3498db", "#2ecc71"]
        for i, (name, res) in enumerate(results.items()):
            vals = [res[m] for m in metrics]
            bars = ax.bar(x + i * width, vals, width, label=name, color=colors[i % len(colors)])

        ax.set_xticks(x + width * (len(results) - 1) / 2)
        ax.set_xticklabels([m.capitalize() for m in metrics], color="white")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score", color="white")
        ax.set_title("Model Comparison", color="white", fontsize=14)
        ax.tick_params(colors="gray")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        legend = ax.legend(facecolor="#333", labelcolor="white")

        results_dir = Path(config.RESULTS_DIR)
        results_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(results_dir / "model_comparison.png", dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"\n[compare] Saved chart → {results_dir}/model_comparison.png")
        plt.close()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--quick", action="store_true", help="5 epochs per model for fast testing")
    args = parser.parse_args()
    run_all(args.epochs, args.quick)
