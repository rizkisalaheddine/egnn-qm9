import torch
import numpy as np
import torch.optim as optim
from torch.nn.utils import clip_grad_norm

from .egnn_perturb import EGNNQM9PerturbRegressionTarget
from tqdm import tqdm

class CFExplainerEGNNRegressionTarget:
    def __init__(self, model, beta, device):
        self.model = model
        self.model.eval()
        self.beta = beta
        self.device = device

    def explain(self, data, alpha=0.1, tau=0.05, lr=1e-2, num_epochs=200, grad_clip=2.0):
        data = data.to(self.device)

        # original prediction (graph-level)
        with torch.no_grad():
            y_orig = self.model(data).squeeze()  # scalar (since one graph)
        y_target = (1.0 + float(alpha)) * y_orig

        # CF model
        cf_model = EGNNQM9PerturbRegressionTarget(self.model, data).to(self.device)
        cf_model.load_state_dict(self.model.state_dict(), strict=False)

        # freeze all weights except edge mask params
        for name, p in cf_model.named_parameters():
            p.requires_grad = (name == "p")

        cf_model.beta = self.beta

        opt = optim.Adam([cf_model.p], lr=lr)

        best = None
        best_loss = float("inf")

        for epoch in tqdm(range(num_epochs),desc="CF Explainer Training"):
            cf_model.train()
            opt.zero_grad()

            y_cf_soft = cf_model(data).squeeze()

            cf_model.eval()
            with torch.no_grad():
                y_cf_hard, hard_edges = cf_model.forward_prediction(data, threshold=0.5)
                y_cf_hard = y_cf_hard.squeeze()
                err_hard = torch.abs(y_cf_hard - y_target)
                success = (err_hard <= tau).float()
                removed = int(torch.sum(1.0 - hard_edges).item())

            cf_model.train()

            # loss uses HARD success gating
            loss_total, loss_pred, loss_graph = cf_model.loss_regression_target(
                y_cf_soft=y_cf_soft,
                y_target=y_target,
                success=success
            )

            loss_total.backward()
            clip_grad_norm([cf_model.p], grad_clip)
            opt.step()

            with torch.no_grad():
                p_soft = torch.sigmoid(cf_model.p)
                print("p_soft min/max:", p_soft.min().item(), p_soft.max().item())

            err_soft = torch.abs(y_cf_soft.detach() - y_target).item()

            print(
                f"Epoch {epoch+1:04d} | "
                f"loss={loss_total.item():.4f} pred={loss_pred.item():.4f} graph={loss_graph.item():.4f} | "
                f"y_orig={y_orig.item():.6f} y_target={(y_target).item():.6f} y_cf={y_cf_hard.item():.6f} | "
                f"err_hard={err_hard.item():.6f} err_soft={err_soft:.6f}  removed={removed} success={int(success.item())} "
            )

            if success.item() == 1.0 and loss_total.item() < best_loss:
                best_loss = loss_total.item()
                best = {
                    "y_orig": y_orig.detach().cpu().item(),
                    "y_target": y_target.detach().cpu().item(),
                    "y_cf": y_cf_hard.detach().cpu().item(),
                    "removed_edges": removed,
                    "hard_edge_mask_undirected": hard_edges.detach().cpu().numpy(),
                    "unique_pairs": cf_model.unique_pairs.detach().cpu().numpy(),
                    "loss_total": loss_total.item(),
                    "loss_pred": loss_pred.item(),
                    "loss_graph": loss_graph.item(),
                }

        return best
