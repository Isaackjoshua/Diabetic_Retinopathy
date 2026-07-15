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


class LogitAdjustedLoss(nn.Module):
    """Logit-adjusted softmax cross-entropy (Menon et al., ICLR 2021).

        loss = CE( logits + tau * log(prior),  target )

    Adding tau*log(prior_y) (a large *negative* offset for rare classes) INSIDE the
    softmax forces the network to produce a larger margin for rare classes to reach the
    same loss -- so the effect is learned into the features, not bolted on post-hoc. At
    inference you use the RAW logits (argmax), no adjustment. This directly targets
    *balanced error / macro-recall*, which is exactly the macro-sensitivity objective, and
    unlike per-class threshold tuning the gain transfers to unseen data.

    `class_counts` is the per-class train frequency (any positive scale; normalised here).
    `tau` controls strength: 0 == plain CE, 1.0 is the paper default; push to 1.5-2.0 to
    favour rare classes more (at some cost to head-class recall / precision). Do NOT also
    pass inverse-frequency `weight` -- that double-corrects the imbalance; leave weight=None.
    """
    def __init__(self, class_counts, tau=1.0, weight=None, reduction="mean", eps=1e-12):
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        prior = counts / counts.sum()
        self.register_buffer("adjust", tau * torch.log(prior + eps))   # shape [C]
        self.register_buffer("weight", weight if weight is not None else None)
        self.tau = tau
        self.reduction = reduction

    def forward(self, logits, target):
        logits = logits.float()                                        # stable under AMP
        return F.cross_entropy(logits + self.adjust.to(logits.device), target,
                               weight=self.weight, reduction=self.reduction)
