import torch
from torch_geometric.utils import get_laplacian, to_dense_adj
from typing import List


class SpectralEnergyLoss:

    def __init__(self):
        pass

    def compute_dirichlet_energy(self, x: torch.Tensor, edge_index: torch.Tensor,
                                  num_nodes: int) -> torch.Tensor:
        try:
            edge_index_laplacian, edge_weight = get_laplacian(
                edge_index,
                normalization='sym',
                num_nodes=num_nodes
            )

            L = to_dense_adj(
                edge_index_laplacian,
                edge_attr=edge_weight,
                max_num_nodes=num_nodes
            ).squeeze(0)

            energy = torch.trace(x.t() @ L @ x)

        except Exception:
            energy = torch.tensor(0.0, device=x.device)

        return energy

    def __call__(self, synthetic_manager, target_batch) -> torch.Tensor:
        syn_batch = synthetic_manager.get_pyg_batch()
        syn_energies = []

        for data in syn_batch.to_data_list():
            energy = self.compute_dirichlet_energy(
                data.x,
                data.edge_index,
                data.num_nodes
            )
            syn_energies.append(energy)

        if len(syn_energies) > 0:
            syn_mean_energy = torch.stack(syn_energies).mean()
        else:
            syn_mean_energy = torch.tensor(0.0, device=syn_batch.x.device)

        tgt_energies = []

        for data in target_batch.to_data_list():
            energy = self.compute_dirichlet_energy(
                data.x,
                data.edge_index,
                data.num_nodes
            )
            tgt_energies.append(energy)

        if len(tgt_energies) > 0:
            tgt_mean_energy = torch.stack(tgt_energies).mean()
        else:
            tgt_mean_energy = torch.tensor(0.0, device=target_batch.x.device)

        loss = (syn_mean_energy - tgt_mean_energy) ** 2

        return loss