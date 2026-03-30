import torch
import torch.nn.functional as F
from torch.optim import Adam
import copy


class InnerLoopTrainer:

    def __init__(self, model_template, lr: float = 0.01, steps: int = 50,
                 early_stop_patience: int = 10, early_stop_threshold: float = 1e-4):
        self.model_template = model_template
        self.lr = lr
        self.steps = steps
        self.early_stop_patience = early_stop_patience
        self.early_stop_threshold = early_stop_threshold

    def train(self, synthetic_batch, device='cuda'):
        model = copy.deepcopy(self.model_template)
        model.to(device)
        model.train()

        optimizer = Adam(model.parameters(), lr=self.lr)

        y_detached = synthetic_batch.y.detach()
        best_loss = float('inf')
        patience_counter = 0

        for step in range(self.steps):
            optimizer.zero_grad()

            if hasattr(synthetic_batch, 'edge_attr') and synthetic_batch.edge_attr is not None:
                logits = model(
                    synthetic_batch.x,
                    synthetic_batch.edge_index,
                    synthetic_batch.batch,
                    synthetic_batch.edge_attr
                )
            else:
                logits = model(
                    synthetic_batch.x,
                    synthetic_batch.edge_index,
                    synthetic_batch.batch
                )

            loss = F.cross_entropy(logits, y_detached)
            current_loss = loss.item()

            if current_loss < best_loss - self.early_stop_threshold:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.early_stop_patience:
                
                loss.backward(retain_graph=False)
                optimizer.step()
                break

            retain = (step < self.steps - 1)
            loss.backward(retain_graph=retain)
            optimizer.step()

        return model
