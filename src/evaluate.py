"""
evaluate.py

Load a trained checkpoint and run it on the test set.
Prints accuracy, F1, precision, recall, and confusion matrix.

Usage:
    python -m src.evaluate --model lstm
    python -m src.evaluate --model cnn_lstm
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config
from src.models import build_model
from src.preprocessing import build_dataset
from src.dataset import make_dataloaders


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds  = []
    all_labels = []
    all_probs  = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        probs  = torch.softmax(logits, dim=-1)
        preds  = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def evaluate_model(model_name, checkpoint_path=None, device=None):
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if checkpoint_path is None:
        checkpoint_path = str(Path(config.CHECKPOINTS_DIR) / f"{model_name}_best.pt")

    print(f"[evaluate] {model_name}  checkpoint={checkpoint_path}  device={device}")

    X, y = build_dataset()
    _, _, test_loader, (X_test, y_test) = make_dataloaders(X, y)

    model = build_model(model_name).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"  loaded epoch {ckpt['epoch']}  (val_loss={ckpt['val_loss']:.4f})")

    y_true, y_pred, y_prob = get_predictions(model, test_loader, device)

    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="binary")
    prec = precision_score(y_true, y_pred, average="binary")
    rec  = recall_score(y_true, y_pred, average="binary")
    cm   = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 50)
    print(f"model: {model_name}")
    print("=" * 50)
    print(f"  accuracy:  {acc:.4f}")
    print(f"  f1:        {f1:.4f}")
    print(f"  precision: {prec:.4f}")
    print(f"  recall:    {rec:.4f}")
    print(f"\nconfusion matrix:")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
    print()
    print(classification_report(y_true, y_pred, target_names=["Poor Form", "Good Form"]))

    results = {
        "model":            model_name,
        "accuracy":         acc,
        "f1":               f1,
        "precision":        prec,
        "recall":           rec,
        "confusion_matrix": cm,
        "y_true":           y_true,
        "y_pred":           y_pred,
        "y_prob":           y_prob,
    }

    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    np.save(results_dir / f"{model_name}_results.npy", results)
    print(f"[evaluate] saved to {results_dir}/{model_name}_results.npy")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="lstm", choices=["lstm", "transformer", "cnn_lstm", "mlp"])
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    evaluate_model(args.model, args.checkpoint)
