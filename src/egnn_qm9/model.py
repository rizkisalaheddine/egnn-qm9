# src/egnn_qm9/model.py

import torch
from torch import nn
from egnn_pytorch import EGNN_Network


def batch_to_egnn_inputs(batch):
    """
    Convert a PyG Batch to (token_ids, coords, mask) for EGNN_Network.
    Assumes batch.x is one-hot atom type encoding.
    """
    B = batch.num_graphs
    x_all = batch.x
    pos_all = batch.pos
    batch_idx = batch.batch

    token_ids_all = x_all.argmax(dim=-1)  # (total_nodes,)

    num_nodes_per_graph = torch.bincount(batch_idx, minlength=B)
    max_n = int(num_nodes_per_graph.max().item())

    tokens = torch.full((B, max_n), fill_value=0, device=batch.x.device, dtype=torch.long)
    coords = torch.zeros(B, max_n, 3, device=batch.x.device)
    mask = torch.zeros(B, max_n, dtype=torch.bool, device=batch.x.device)

    node_offset = 0
    for g in range(B):
        n = int(num_nodes_per_graph[g])
        idx = torch.arange(node_offset, node_offset + n, device=batch.x.device)

        tokens[g, :n] = token_ids_all[idx]
        coords[g, :n] = pos_all[idx]
        mask[g, :n] = True

        node_offset += n

    return tokens, coords, mask


class EGNNQM9Model(nn.Module):
    def __init__(self, num_atom_types: int, max_nodes: int,
                 hidden_dim: int = 64, depth: int = 4,
                 num_nearest_neighbors: int = 8):
        super().__init__()

        self.egnn = EGNN_Network(
            num_tokens = num_atom_types,
            num_positions = max_nodes,
            dim = hidden_dim,
            depth = depth,
            num_nearest_neighbors = num_nearest_neighbors,
            norm_coors = True,
            coor_weights_clamp_value = 2.0,
        )

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch, y_mean: float, y_mad: float):
        tokens, coords, mask = batch_to_egnn_inputs(batch)
        feats, _ = self.egnn(tokens, coords, mask=mask)  # (B, N, dim)

        # mean pooling over valid nodes
        mask_f = mask.float().unsqueeze(-1)  # (B, N, 1)
        feats_masked = feats * mask_f
        mol_repr = feats_masked.sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

        pred_norm = self.readout(mol_repr).squeeze(-1)  # (B,)
        return pred_norm
