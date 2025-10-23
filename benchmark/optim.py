import torch

from torch.optim import Adam

class GCAdam(Adam):
    #适用于其他优化器
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.dim() > 1:  # Centerize over all dims except last
                    grad_mean = grad.mean(dim=tuple(range(grad.dim()-1)), keepdim=True)
                    grad = grad - grad_mean
                    p.grad.data = grad
        # 调用原始RMSprop的step
        return super(GCAdam, self).step(closure)

class ALRS:
    """
    refer to
    Bootstrap Generalization Ability from Loss Landscape Perspective
    """

    def __init__(self, optimizer, loss_threshold=0.01, loss_ratio_threshold=0.01, decay_rate=0.97):
        self.optimizer = optimizer
        self.loss_threshold = loss_threshold
        self.decay_rate = decay_rate
        self.loss_ratio_threshold = loss_ratio_threshold

        self.last_loss = 999

    def step(self, loss):
        delta = self.last_loss - loss
        if delta < self.loss_threshold and delta / self.last_loss < self.loss_ratio_threshold:
            for group in self.optimizer.param_groups:
                group["lr"] *= self.decay_rate
                now_lr = group["lr"]
                print(f"now lr = {now_lr}")

        self.last_loss = loss