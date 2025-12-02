# src/egnn_qm9/utils.py

import torch


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
