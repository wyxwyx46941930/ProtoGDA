import torch
import torch.nn.functional as F


class SemanticFidelityLoss:

    def __init__(self):
        pass

    def __call__(self, model, source_batch):
        model.eval()

        logits = model(
            source_batch.x,
            source_batch.edge_index,
            source_batch.batch,
            edge_weight=None
        )

        loss = F.cross_entropy(logits, source_batch.y)

        return loss