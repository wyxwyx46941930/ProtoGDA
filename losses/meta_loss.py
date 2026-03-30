import torch
from .semantic_loss import SemanticFidelityLoss
from .structural_loss import GromovWassersteinLoss
from .spectral_loss import SpectralEnergyLoss


class MetaLoss:

    def __init__(self,
                 lambda_sem: float = 10.0,
                 lambda_struct: float = 1.0,
                 lambda_spec: float = 0.5,
                 gw_max_iter: int = 50,
                 gw_reg: float = 0.1):
        self.lambda_sem = lambda_sem
        self.lambda_struct = lambda_struct
        self.lambda_spec = lambda_spec

        self.semantic_loss_fn = SemanticFidelityLoss()
        self.structural_loss_fn = GromovWassersteinLoss(
            max_iter=gw_max_iter,
            reg=gw_reg
        )
        self.spectral_loss_fn = SpectralEnergyLoss()

    def __call__(self,
                 student_model,
                 synthetic_manager,
                 source_batch,
                 target_batch_struct,
                 target_batch_node):

        L_sem = self.semantic_loss_fn(student_model, source_batch)

        syn_adjacency = synthetic_manager.get_adjacency_matrices()
        L_struct = self.structural_loss_fn(syn_adjacency, target_batch_struct)

        L_spec = self.spectral_loss_fn(synthetic_manager, target_batch_node)

        total_loss = (
            self.lambda_sem * L_sem +
            self.lambda_struct * L_struct +
            self.lambda_spec * L_spec
        )

        return {
            'total': total_loss,
            'semantic': L_sem.item(),
            'structural': L_struct.item(),
            'spectral': L_spec.item()
        }