# src/egnn_qm9/model.py

import torch
from torch import nn

def global_mean_pool_pure(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    num_graphs = int(batch.max()) + 1
    out = torch.zeros(num_graphs, x.size(-1), device=x.device)
    out.index_add_(0, batch, x)
    count = torch.bincount(batch, minlength=num_graphs).unsqueeze(-1).clamp(min=1)
    return out / count


class EGNNLayer(nn.Module):
    """
    EGNN layer with edge inference (gating), as in Section 3.3 of the paper:
    - compute messages m_ij
    - compute gate a_ij = sigmoid(Linear(m_ij))
    - use m_ij * a_ij for node and coord updates
    """

    def __init__(self, hidden_dim: int, edge_attr_dim: int = 0, m_dim: int = 128,update_coords: bool = False):
        super().__init__()

        self.update_coords = update_coords

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1 + edge_attr_dim, m_dim),
            nn.SiLU(),
            nn.Linear(m_dim, m_dim),
            nn.SiLU(),
        )

        # edge gate: infer edges
        self.edge_gate = nn.Sequential(
            nn.Linear(m_dim, 1),
            nn.Sigmoid()
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + m_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.coor_mlp = nn.Sequential(
            nn.Linear(m_dim, m_dim),
            nn.SiLU(),
            nn.Linear(m_dim, 1),
        )

    def forward(self, h, x, edge_index, edge_attr=None):
        row, col = edge_index  # row: source j, col: target i

        x_j = x[row]
        x_i = x[col]
        coord_diff = x_i - x_j
        dist = coord_diff.norm(dim=-1, keepdim=True)

        h_j = h[row]
        h_i = h[col]

        msg_inputs = [h_i, h_j, dist]
        if edge_attr is not None:
            msg_inputs.append(edge_attr)
        msg_inputs = torch.cat(msg_inputs, dim=-1)

        m_ij = self.edge_mlp(msg_inputs)              # (E, m_dim)

        # ---- edge inference gate (Section 3.3) ----
        gate = self.edge_gate(m_ij)                  # (E, 1) in (0,1)
        m_ij = m_ij * gate                           # gated messages

        # ---- node update ----
        N = h.size(0)
        m_aggr = torch.zeros(N, m_ij.size(-1), device=h.device)
        m_aggr.index_add_(0, col, m_ij)

        node_input = torch.cat([h, m_aggr], dim=-1)
        delta_h = self.node_mlp(node_input)
        h = h + delta_h

        # ---- coordinate update (still equivariant) ----
        if self.update_coords:
            coor_weights = self.coor_mlp(m_ij).squeeze(-1)      # (E,)
            coord_updates = coord_diff * coor_weights.unsqueeze(-1)

            coord_aggr = torch.zeros_like(x)
            coord_aggr.index_add_(0, col, coord_updates)

            x = x + coord_aggr

        return h, x


class EGNNQM9Model(nn.Module):
    def __init__(self, in_dim: int, edge_attr_dim: int,
                 hidden_dim: int = 128, num_layers: int = 7,update_coords: bool = False):
        super().__init__()

        self.embedding = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList(
            [EGNNLayer(hidden_dim, edge_attr_dim=edge_attr_dim, m_dim=hidden_dim, update_coords=update_coords)
             for _ in range(num_layers)]
        )

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        h = self.embedding(data.x)
        x = data.pos
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr)

        mol_repr = global_mean_pool_pure(h, data.batch)
        out = self.readout(mol_repr).squeeze(-1)
        return out
