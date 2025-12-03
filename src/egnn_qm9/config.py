from dataclasses import dataclass

@dataclass
class TrainConfig:
    data_root: str = "data/QM9"

    # Paper-ish hyperparameters
    batch_size: int = 96          # paper uses 96
    num_epochs: int = 1000        # paper trains 1000 epochs

    # Learning rates: smaller for most, higher for homo/lumo/gap
    base_lr: float = 5e-4         # for most properties
    lr_homo_lumo_gap: float = 1e-3

    weight_decay: float = 1e-12   # tiny weight decay as in many EGNN impls

    # Model architecture (paper: 7 layers, 128 channels)
    hidden_dim: int = 128
    depth: int = 7

    # Whether to use cosine schedule
    use_cosine_lr: bool = True

    # Whether to update coordinates x or just use distances (paper often freezes x for QM9)
    coord_updates: bool = False   # set False to “no x update”

    # Which property we are training on
    property_name: str = "mu"
