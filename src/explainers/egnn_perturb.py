import torch
from torch import nn
import torch.nn.functional as F

# import your model pieces
from src.egnn_qm9.custom_model import EGNNQM9Model, global_mean_pool_pure


def build_undirected_edge_ids(edge_index: torch.Tensor):
    """
    Given directed edge_index [2, E], build:
      - undirected_pairs: [E, 2] with sorted (u,v)
      - unique_pairs:     [M, 2] unique undirected edges
      - inv:              [E] mapping each directed edge -> undirected id in [0..M-1]
    This ties mask parameters between (u->v) and (v->u).
    """
    row, col = edge_index
    u = torch.minimum(row, col)
    v = torch.maximum(row, col)
    pairs = torch.stack([u, v], dim=1)  # [E,2]

    unique_pairs, inv = torch.unique(pairs, dim=0, return_inverse=True)
    return pairs, unique_pairs, inv


class EGNNLayerPerturb(nn.Module):
    """
    Same as your EGNNLayer, but multiplies an external edge mask into the gated messages.
    """
    def __init__(self, base_layer: nn.Module):
        super().__init__()
        # reuse the same submodules (weights shared via load_state_dict)
        self.update_coords = base_layer.update_coords
        self.edge_mlp = base_layer.edge_mlp
        self.edge_gate = base_layer.edge_gate
        self.node_mlp = base_layer.node_mlp
        self.coor_mlp = base_layer.coor_mlp

    def forward(self, h, x, edge_index, edge_mask, edge_attr=None):
        row, col = edge_index  # j -> i

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

        m_ij = self.edge_mlp(msg_inputs)      # (E, m_dim)

        gate = self.edge_gate(m_ij)           # (E, 1) in (0,1)

        # ---- counterfactual edge mask (E,1) ----
        # edge_mask is expected in (0,1) already (sigmoid of parameters)
        m_ij = m_ij * gate * edge_mask

        # ---- node update ----
        N = h.size(0)
        m_aggr = torch.zeros(N, m_ij.size(-1), device=h.device)
        m_aggr.index_add_(0, col, m_ij)

        node_input = torch.cat([h, m_aggr], dim=-1)
        delta_h = self.node_mlp(node_input)
        h = h + delta_h

        # ---- coordinate update ----
        if self.update_coords:
            coor_weights = self.coor_mlp(m_ij).squeeze(-1)  # (E,)
            coord_updates = coord_diff * coor_weights.unsqueeze(-1)

            coord_aggr = torch.zeros_like(x)
            coord_aggr.index_add_(0, col, coord_updates)

            x = x + coord_aggr

        return h, x


class EGNNQM9PerturbRegressionTarget(nn.Module):
    """
    Counterfactual wrapper for your EGNNQM9Model:
      - Keeps same predictor weights (frozen)
      - Learns edge mask parameters only
      - Regression-target CF loss: gated MSE to (1+alpha)*y_orig + beta * edge deletions count
    """
    def __init__(self, base_model: EGNNQM9Model, data):
        super().__init__()

        # copy architecture modules
        self.embedding = base_model.embedding
        self.readout = base_model.readout

        # wrap each EGNN layer to accept edge masks
        self.layers = nn.ModuleList([EGNNLayerPerturb(l) for l in base_model.layers])

        # build undirected edge parameterization
        edge_index = data.edge_index
        _, unique_pairs, inv = build_undirected_edge_ids(edge_index)

        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_inv", inv)               # [E] -> undirected id
        self.register_buffer("unique_pairs", unique_pairs) # [M,2]
        self.num_undirected = int(unique_pairs.size(0))

        # mask parameters (one per undirected edge)
        # init near "keep edges" (sigmoid large-ish) -> start at +4 ~ 0.982
        self.p = nn.Parameter(torch.full((self.num_undirected,), 4.0, device=edge_index.device))

        self.beta = None  # set by explainer

    def edge_mask_soft(self, threshold=0.5):
        """
        Straight-Through (STE) mask:
        - forward uses hard 0/1 mask
        - backward uses sigmoid gradients
        Returns per-directed-edge mask of shape (E,1).
        """
        p_soft_undir = torch.sigmoid(self.p)  # (M,)
        p_hard_undir = (p_soft_undir >= threshold).float()  # (M,)

        # STE trick: forward = hard, gradient = soft
        p_ste_undir = p_hard_undir + p_soft_undir - p_soft_undir.detach()

        p_dir = p_ste_undir[self.edge_inv]  # (E,)
        return p_dir.unsqueeze(-1)          # (E,1)


    @torch.no_grad()
    def edge_mask_hard(self, threshold=0.5):
        """
        Returns per-undirected hard mask (M,) and per-directed hard mask (E,1)
        """
        hard_undir = (torch.sigmoid(self.p) >= threshold).float()   # (M,)
        hard_dir = hard_undir[self.edge_inv].unsqueeze(-1)          # (E,1)
        return hard_undir, hard_dir

    def forward(self, data):
        h = self.embedding(data.x)
        x = data.pos
        edge_attr = data.edge_attr

        edge_mask = self.edge_mask_soft(threshold=0.5)  # (E,1)

        for layer in self.layers:
            h, x = layer(h, x, self.edge_index, edge_mask, edge_attr)

        mol_repr = global_mean_pool_pure(h, data.batch)
        out = self.readout(mol_repr).squeeze(-1)  # (B,)
        return out

    @torch.no_grad()
    def forward_prediction(self, data, threshold=0.5):
        """
        Same forward but uses hard masks for interpretability and success checking.
        """
        hard_undir, hard_dir = self.edge_mask_hard(threshold=threshold)

        h = self.embedding(data.x)
        x = data.pos
        edge_attr = data.edge_attr

        for layer in self.layers:
            h, x = layer(h, x, self.edge_index, hard_dir, edge_attr)

        mol_repr = global_mean_pool_pure(h, data.batch)
        out = self.readout(mol_repr).squeeze(-1)

        return out, hard_undir

    def loss_regression_target(self, y_cf_soft, y_target, success):
        y_cf_soft = y_cf_soft.squeeze()
        y_target = y_target.squeeze()

        loss_pred = (y_cf_soft - y_target).pow(2)

        p_soft = torch.sigmoid(self.p)
        loss_graph = torch.sum(1.0 - p_soft)

        loss_total = (1.0 - success) * loss_pred + self.beta * loss_graph
        return loss_total, loss_pred, loss_graph


