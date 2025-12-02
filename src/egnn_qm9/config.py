# src/egnn_qm9/config.py

from dataclasses import dataclass

@dataclass
class TrainConfig:
    data_root: str = "data/QM9"
    batch_size: int = 1024
    num_epochs: int = 30
    lr: float = 5e-4
    hidden_dim: int = 128
    depth: int = 7
    num_nearest_neighbors: int = 8
    property_name: str = "mu"  # can be overridden by CLI
