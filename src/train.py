"""
train.py

Training loop. Runs for up to NUM_EPOCHS with early stopping.
Saves the best checkpoint by val loss.

Usage:
    python -m src.train --model lstm
    python -m src.train --model cnn_lstm --epochs 30
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
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()

        # gradient clipping to avoid exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(dim=-1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_loop(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(dim=-1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)

    return total_loss / total, correct / total


def train(model_name="lstm", epochs=config.NUM_EPOCHS, lr=config.LEARNING_RATE,
          batch_size=config.BATCH_SIZE, patience=10, device=None):

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"[train] model={model_name}  device={device}")

    # load data
    print("[train] loading dataset...")
    X, y = build_dataset()
    train_loader, val_loader, test_loader, _ = make_dataloaders(X, y, batch_size=batch_size)

    # build model
    model = build_model(model_name).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] {n_params:,} trainable parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt_dir = Path(config.CHECKPOINTS_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_loop(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"epoch {epoch:03d}/{epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"({elapsed:.1f}s)"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            ckpt_path = ckpt_dir / f"{model_name}_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, ckpt_path)
            print(f"  -> saved checkpoint: {ckpt_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[train] early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="lstm", choices=["lstm", "transformer", "cnn_lstm", "mlp"])
    parser.add_argument("--epochs",     type=int,   default=config.NUM_EPOCHS)
    parser.add_argument("--lr",         type=float, default=config.LEARNING_RATE)
    parser.add_argument("--batch_size", type=int,   default=config.BATCH_SIZE)
    parser.add_argument("--patience",   type=int,   default=10)
    args = parser.parse_args()

    train(
        model_name=args.model,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
    )
