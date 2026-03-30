"""
双层优化主控器：协调内外层循环
"""
import torch
from torch_geometric.loader import DataLoader
import copy
from typing import Iterator


class MetaTrainer:

    def __init__(self,
                 config,
                 synthetic_manager,
                 student_model,
                 meta_loss_fn,
                 inner_trainer,
                 outer_updater,
                 source_loader,
                 target_loader_struct,
                 target_loader_node):
        self.config = config
        self.synthetic_manager = synthetic_manager
        self.student_model = student_model
        self.meta_loss_fn = meta_loss_fn
        self.inner_trainer = inner_trainer
        self.outer_updater = outer_updater

        self.source_loader = source_loader
        self.target_loader_struct = target_loader_struct
        self.target_loader_node = target_loader_node

        self.source_iter = self._infinite_iterator(source_loader)
        self.target_struct_iter = self._infinite_iterator(target_loader_struct)
        self.target_node_iter = self._infinite_iterator(target_loader_node)

    def _infinite_iterator(self, loader):
        while True:
            for batch in loader:
                yield batch

    def train_epoch(self, epoch):
        device = self.config.device

        synthetic_batch = self.synthetic_manager.get_pyg_batch(
            threshold=self.config.loss.adjacency_threshold
        )
        synthetic_batch = synthetic_batch.to(device)

        student_trained = self.inner_trainer.train(synthetic_batch, device)

        source_batch = next(self.source_iter).to(device)
        target_batch_struct = next(self.target_struct_iter).to(device)
        target_batch_node = next(self.target_node_iter).to(device)

        meta_loss_dict = self.meta_loss_fn(
            student_trained,
            self.synthetic_manager,
            source_batch,
            target_batch_struct,
            target_batch_node
        )
        self.outer_updater.update(meta_loss_dict)

        return meta_loss_dict

    def train(self):
        best_loss = float('inf')
        patience_counter = 0
        num_epochs = self.config.training.num_meta_epochs
        patience = self.config.training.patience
        log_interval = self.config.training.log_interval

        print(f"\n{'='*60}")
        print(f"Starting Meta-Training for {num_epochs} epochs")
        print(f"{'='*60}\n")

        for epoch in range(num_epochs):
            loss_dict = self.train_epoch(epoch)
            if epoch % log_interval == 0:
                print(f"Epoch {epoch:4d}/{num_epochs} | "
                      f"Total: {loss_dict['total']:.4f} | "
                      f"Sem: {loss_dict['semantic']:.4f} | "
                      f"Struct: {loss_dict['structural']:.4f} | "
                      f"Spec: {loss_dict['spectral']:.4f}")

            current_loss = loss_dict['total'].item() if torch.is_tensor(loss_dict['total']) else loss_dict['total']

            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1

                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    print(f"Best loss: {best_loss:.4f}")
                    break

        print(f"\n{'='*60}")
        print(f"Meta-Training Completed!")
        print(f"Best loss: {best_loss:.4f}")
        print(f"{'='*60}\n")

        stats = self.synthetic_manager.get_statistics()
        print("Synthetic Graph Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:.4f}")

        return self.synthetic_manager
