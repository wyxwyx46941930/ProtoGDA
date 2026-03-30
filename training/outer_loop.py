import torch
from torch.optim import Adam


class OuterLoopUpdater:

    def __init__(self, synthetic_manager, meta_lr: float = 0.0001, grad_clip: float = 1.0):
        self.synthetic_manager = synthetic_manager
        self.grad_clip = grad_clip

        self.optimizer = Adam(
            synthetic_manager.parameters(),
            lr=meta_lr
        )

    def update(self, meta_loss_dict):
        self.optimizer.zero_grad()

        total_loss = meta_loss_dict['total']

        # 反向传播
        total_loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.synthetic_manager.parameters(),
            max_norm=self.grad_clip
        )

        # 更新参数
        self.optimizer.step()

        return meta_loss_dict

    def get_optimizer(self):
        return self.optimizer
