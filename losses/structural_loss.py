import torch
import numpy as np
from typing import List


class GromovWassersteinLoss:

    def __init__(self, max_iter: int = 50, tol: float = 1e-5, reg: float = 0.1):
        self.max_iter = max_iter
        self.tol = tol
        self.reg = reg

    def compute_distance_matrix(self, adj: torch.Tensor) -> torch.Tensor:
        n = adj.shape[0]
        dist = 1.0 - adj
        dist = dist.fill_diagonal_(0)
        return dist

    def gw_distance_simple(self, C1: torch.Tensor, C2: torch.Tensor) -> float:
        mean1 = C1.mean()
        mean2 = C2.mean()

        std1 = C1.std()
        std2 = C2.std()

        max1 = C1.max()
        max2 = C2.max()

        dist = (mean1 - mean2) ** 2 + (std1 - std2) ** 2 + 0.1 * (max1 - max2) ** 2
        return dist.item()

    def gw_distance_sinkhorn(self, C1: torch.Tensor, C2: torch.Tensor) -> float:
        n1, n2 = C1.shape[0], C2.shape[0]
        device = C1.device

        p = torch.ones(n1, device=device) / n1
        q = torch.ones(n2, device=device) / n2

        T = torch.outer(p, q)

        for _ in range(self.max_iter):
            loss_matrix = torch.zeros(n1, n2, device=device)

            for i in range(n1):
                for j in range(n2):
                    diff = C1[i, :].unsqueeze(1) - C2[j, :].unsqueeze(0)
                    loss_matrix[i, j] = (diff ** 2 * T).sum()

            K = torch.exp(-loss_matrix / self.reg)

            u = p / (K @ q + 1e-10)
            v = q / (K.t() @ u + 1e-10)

            T_new = u.unsqueeze(1) * K * v.unsqueeze(0)

            if torch.norm(T - T_new) < self.tol:
                break

            T = T_new

        final_loss = (loss_matrix * T).sum()
        return final_loss.item()

    def graph_structure_distance_differentiable(self, A_syn: torch.Tensor, A_tgt: torch.Tensor) -> torch.Tensor:
        degree_syn = A_syn.sum(dim=1)
        degree_tgt = A_tgt.sum(dim=1)

        mean_degree_syn = degree_syn.mean()
        mean_degree_tgt = degree_tgt.mean()
        std_degree_syn = degree_syn.std() + 1e-6
        std_degree_tgt = degree_tgt.std() + 1e-6

        density_syn = A_syn.sum() / (A_syn.shape[0] * (A_syn.shape[0] - 1) + 1e-6)
        density_tgt = A_tgt.sum() / (A_tgt.shape[0] * (A_tgt.shape[0] - 1) + 1e-6)

        A_syn_binary = (A_syn > 0.5).float()
        A_tgt_binary = (A_tgt > 0.5).float()
        triangles_syn = torch.trace(torch.matrix_power(A_syn_binary, 3)) / 6.0
        triangles_tgt = torch.trace(torch.matrix_power(A_tgt_binary, 3)) / 6.0
        triangle_ratio_syn = triangles_syn / (A_syn.shape[0] + 1e-6)
        triangle_ratio_tgt = triangles_tgt / (A_tgt.shape[0] + 1e-6)

        dist = (
            (mean_degree_syn - mean_degree_tgt) ** 2 +
            (std_degree_syn - std_degree_tgt) ** 2 +
            (density_syn - density_tgt) ** 2 * 10.0 +
            (triangle_ratio_syn - triangle_ratio_tgt) ** 2 * 5.0
        )

        return dist

    def __call__(self, syn_adjacency: List[torch.Tensor], target_batch) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=syn_adjacency[0].device, requires_grad=True)
        count = 0

        target_graphs = target_batch.to_data_list()

        for A_syn in syn_adjacency:
            target_graph = target_graphs[np.random.randint(len(target_graphs))]

            n_target = target_graph.num_nodes
            A_tgt = torch.zeros(n_target, n_target, device=A_syn.device)

            if hasattr(target_graph, 'edge_index'):
                edge_index = target_graph.edge_index
                A_tgt[edge_index[0], edge_index[1]] = 1.0

            A_tgt = (A_tgt + A_tgt.t()) / 2
            A_tgt = A_tgt.detach()

            dist = self.graph_structure_distance_differentiable(A_syn, A_tgt)

            total_loss = total_loss + dist
            count += 1

        return total_loss / max(count, 1)