"""Loss functions for the DR experiments."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional per-class alpha weights.

        FL = alpha_t * (1 - p_t)**gamma * (-log p_t)

    `weight` (per-class alpha) is applied via the CE term, so this combines
    inverse-frequency class weighting with focal down-weighting of easy examples
    -- useful for lifting recall on rare grades (R2/R3). gamma=0 reduces to
    weighted cross-entropy. Compatible with RETFound's train_one_epoch (hard
    targets; mixup is off).
    """
    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        logits = logits.float()                       # stable under AMP
        logp = F.log_softmax(logits, dim=1)
        p_t = logp.gather(1, target.unsqueeze(1)).squeeze(1).exp()   # prob of true class
        focal = (1.0 - p_t).clamp(min=1e-6) ** self.gamma
        ce = F.nll_loss(logp, target, weight=self.weight, reduction="none")  # includes alpha_t
        loss = focal * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
