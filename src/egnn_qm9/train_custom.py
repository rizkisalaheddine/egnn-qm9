# src/egnn_qm9/train.py

import argparse
import os
import json
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


from .config import TrainConfig
from .data import load_qm9_splits, PROPERTY_TO_IDX
from .custom_model import EGNNQM9Model
from .utils import get_device




def run_epoch(model, loader, device, y_mean, y_mad, target_idx, train: bool, optimizer=None):
    if train:
        model.train()
    else:
        model.eval()

    total_loss, total_mae, total_graphs = 0.0, 0.0, 0

    for batch in tqdm(loader, disable=not train):
        batch = batch.to(device)
        y = batch.y[:, target_idx]               # (B,)
        y_norm = (y - y_mean) / y_mad            # MAD normalization

        if train:
            optimizer.zero_grad()

        # NEW: model just takes the PyG batch
        pred_norm = model(batch)                 # (B,)

        loss = F.l1_loss(pred_norm, y_norm)

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # de-normalize for MAE in original units
        pred = pred_norm * y_mad + y_mean
        mae = (pred - y).abs().sum().item()

        Bcur = batch.num_graphs
        total_loss += loss.item() * Bcur
        total_mae += mae
        total_graphs += Bcur

    return total_loss / total_graphs, total_mae / total_graphs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--property", type=str, default="mu", choices=list(PROPERTY_TO_IDX.keys()))
    args = parser.parse_args()

    cfg = TrainConfig(property_name=args.property)
    device = get_device()
    print("Using device:", device)
    print("Config:", cfg)

    (train_loader, val_loader, test_loader), (y_mean, y_mad), dataset, target_idx = load_qm9_splits(
        root=cfg.data_root,
        batch_size=cfg.batch_size,
        property_name=cfg.property_name,
    )

    print(f"{cfg.property_name} mean={y_mean:.6f}, MAD={y_mad:.6f}")

    in_dim = dataset.num_features
    edge_attr_dim = 0 if dataset[0].edge_attr is None else dataset[0].edge_attr.size(-1)

    model = EGNNQM9Model(
        in_dim=in_dim,
        edge_attr_dim=edge_attr_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.depth
    ).to(device)

    # Choose LR based on property (paper: 1e-3 for homo/lumo/gap)
    if cfg.property_name in ["homo", "lumo", "gap"]:
        lr = cfg.lr_homo_lumo_gap
    else:
        lr = cfg.base_lr

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)

    # Cosine LR schedule over epochs (optional)
    if cfg.use_cosine_lr:
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs, eta_min=0.0)
    else:
        scheduler = None
    
    # ---------- OUTPUT / MODEL DIRS ----------
    os.makedirs("outputs", exist_ok=True)
    models_root = "models"
    property_dir = os.path.join(models_root, cfg.property_name)
    os.makedirs(property_dir, exist_ok=True)

    # Save config once at the beginning (as JSON)
    config_path = os.path.join(property_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    print(f"Saved config to {config_path}")

    best_model_path = os.path.join(property_dir, "best_model_state.pt")

    # --- NEW: lists to track the history ---
    train_losses, train_maes = [], []
    val_losses, val_maes = [], []

    best_val_loss = float("inf")
    best_val_mae = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    patience = cfg.patience
    min_delta = cfg.min_delta

    for epoch in range(1, cfg.num_epochs + 1):
        train_loss, train_mae = run_epoch(
            model, train_loader, device, y_mean, y_mad, target_idx, train=True, optimizer=optimizer
        )
        val_loss, val_mae = run_epoch(
            model, val_loader, device, y_mean, y_mad, target_idx, train=False
        )

        # --- NEW: store them ---
        train_losses.append(train_loss)
        train_maes.append(train_mae)
        val_losses.append(val_loss)
        val_maes.append(val_mae)

        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch {epoch:03d} | "
            f"Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f}"
            f"Val loss: {val_loss:.6f} | Best val loss: {best_val_loss:.6f}"
        )
        # ---------- EARLY STOPPING + SAVE BEST MODEL ----------
        if val_loss + min_delta < best_val_loss:
            best_val_mae = val_mae
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0

            # save full model (not just state_dict)
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best val MAE. Saved model to {best_model_path}")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement for {epochs_no_improve} epochs")

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}, best val MAE {best_val_mae:.4f})")
            break
    # ---------- LOAD BEST MODEL ----------
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    
    # --- evaluate on test set ---
    test_loss, test_mae = run_epoch(
        model, test_loader, device, y_mean, y_mad, target_idx, train=False
    )
    print(f"Test MAE for {cfg.property_name}: {test_mae:.4f}")

    # --- NEW: save metrics to disk ---
    metrics = {
        "train_loss": train_losses,
        "train_mae": train_maes,
        "val_loss": val_losses,
        "val_mae": val_maes,
        "test_loss": test_loss,
        "test_mae": test_mae,
        "property": cfg.property_name,
    }

    
    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", f"metrics_{cfg.property_name}.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics to {out_path}")

if __name__ == "__main__":
    main()
