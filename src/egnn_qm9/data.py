# src/egnn_qm9/data.py

import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

PROPERTY_TO_IDX = {
    "mu": 0,
    "alpha": 1,
    "homo": 2,
    "lumo": 3,
    "gap": 4,
    "r2": 5,
    "zpve": 6,
    "U0": 7,
    "U": 8,
    "H": 9,
    "G": 10,
    "Cv": 11,
}


def load_qm9_splits(root: str, batch_size: int, property_name: str):
    target_idx = PROPERTY_TO_IDX[property_name]

    dataset = QM9(root=root)
    torch.manual_seed(0)
    perm = torch.randperm(len(dataset))
    dataset = dataset[perm]

    N = len(dataset)
    n_train = 100_000
    n_val   = 18_000
    n_test  = 13_000
    train_dataset = dataset[:n_train]
    val_dataset = dataset[n_train:n_train + n_val]
    test_dataset = dataset[n_train + n_val:n_train + n_val + n_test]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # normalization of target
    with torch.no_grad():
        ys = torch.stack([d.y[target_idx] for d in train_dataset], dim=0)
        y_mean = ys.mean().item()
        y_mad = (ys - y_mean).abs().mean().item()

    return (
        (train_loader, val_loader, test_loader),
        (y_mean, y_mad),
        dataset,  # for num_features, etc.
        target_idx,
    )
