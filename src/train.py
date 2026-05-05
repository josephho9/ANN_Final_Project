"""
Training loop with early stopping and checkpoint saving.

Usage:
    python -m src.train --model lstm --epochs 50
    python -m src.train --model transformer --lr 5e-4
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config
from src.models import build_model
from src.preprocessing import build_dataset
from src.dataset import make_dataloaders


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(dim=-1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(dim=-1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


def train(
    model_name: str = "lstm",
    epochs: int = config.NUM_EPOCHS,
    lr: float = config.LEARNING_RATE,
    batch_size: int = config.BATCH_SIZE,
    patience: int = 10,
    device: str = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[train] model={model_name}  device={device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("[train] Loading dataset …")
    X, y = build_dataset()
    train_loader, val_loader, test_loader, _ = make_dataloaders(X, y, batch_size=batch_size)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(model_name).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Parameters: {num_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Training loop ─────────────────────────────────────────────────────────
    ckpt_dir = Path(config.CHECKPOINTS_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"({elapsed:.1f}s)"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            ckpt_path = ckpt_dir / f"{model_name}_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, ckpt_path)
            print(f"  ✓ Saved checkpoint → {ckpt_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[train] Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a pushup form classifier")
    parser.add_argument("--model", default="lstm", choices=["lstm", "transformer", "cnn_lstm", "mlp"])
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    train(
        model_name=args.model,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
    )
