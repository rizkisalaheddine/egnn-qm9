# src/egnn_qm9/train.py

import argparse
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from .config import TrainConfig
from .data import load_qm9_splits, PROPERTY_TO_IDX
from .model import EGNNQM9Model
from .utils import get_device


def run_epoch(model, loader, device, y_mean, y_mad, target_idx, train: bool, optimizer=None):
    if train:
        model.train()
    else:
        model.eval()

    total_loss, total_mae, total_graphs = 0.0, 0.0, 0

    for batch in tqdm(loader, disable=not train):
        batch = batch.to(device)
        y = batch.y[:, target_idx]  # (B,)
        y_norm = (y - y_mean) / y_mad

        if train:
            optimizer.zero_grad()

        pred_norm = model(batch, y_mean, y_mad)
        loss = F.l1_loss(pred_norm, y_norm)

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

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

    print(f"{cfg.property_name} mean={y_mean:.6f}, std={y_mad:.6f}")

    num_atom_types = dataset.num_features
    max_nodes = max(data.num_nodes for data in dataset)

    model = EGNNQM9Model(
        num_atom_types=num_atom_types,
        max_nodes=max_nodes,
        hidden_dim=cfg.hidden_dim,
        depth=cfg.depth,
        num_nearest_neighbors=cfg.num_nearest_neighbors,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=cfg.lr)

    for epoch in range(1, cfg.num_epochs + 1):
        train_loss, train_mae = run_epoch(
            model, train_loader, device, y_mean, y_mad, target_idx, train=True, optimizer=optimizer
        )
        val_loss, val_mae = run_epoch(
            model, val_loader, device, y_mean, y_mad, target_idx, train=False
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f}"
        )

    test_loss, test_mae = run_epoch(
        model, test_loader, device, y_mean, y_mad, target_idx, train=False
    )
    print(f"Test MAE for {cfg.property_name}: {test_mae:.4f}")


if __name__ == "__main__":
    main()
