"""
模型验证器
"""
import torch
from torch_geometric.loader import DataLoader
from .metrics import compute_metrics


class ModelValidator:
    """模型验证器"""

    def __init__(self, device='cuda'):
        """
        Args:
            device: 计算设备
        """
        self.device = device

    def evaluate(self, model, loader, return_predictions=False):
        """
        评估模型
        Args:
            model: GNN模型
            loader: 数据加载器
            return_predictions: 是否返回预测结果
        Returns:
            评估指标字典 (和可选的预测结果)
        """
        model.eval()
        model.to(self.device)

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)

                # 前向传播（不使用edge_attr，避免维度不匹配）
                logits = model(
                    batch.x,
                    batch.edge_index,
                    batch.batch,
                    edge_weight=None
                )

                # 预测
                preds = logits.argmax(dim=1)

                all_preds.append(preds.cpu())
                all_labels.append(batch.y.cpu())

        # 合并结果
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        # 计算指标
        metrics = compute_metrics(all_preds, all_labels)

        if return_predictions:
            return metrics, all_preds, all_labels
        else:
            return metrics

    def evaluate_single_batch(self, model, batch):
        """
        评估单个batch
        Args:
            model: GNN模型
            batch: PyG batch
        Returns:
            准确率
        """
        model.eval()
        model.to(self.device)
        batch = batch.to(self.device)

        with torch.no_grad():
            if hasattr(batch, 'edge_attr'):
                logits = model(
                    batch.x,
                    batch.edge_index,
                    batch.batch,
                    batch.edge_attr
                )
            else:
                logits = model(
                    batch.x,
                    batch.edge_index,
                    batch.batch
                )

            preds = logits.argmax(dim=1)
            correct = (preds == batch.y).sum().item()
            total = len(batch.y)

        return correct / total
