import torch
import torch.nn as nn
import torch.nn.functional as F


def log_softmax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically accurate log-softmax that avoids Softmax Collapse.

    Reference:
        "A Fix for Softmax and Cross-Entropy", Jonathan T. Barron et al.
        https://jonbarron.info/softmaxfix/
    """
    i_max = torch.argmax(z, dim=dim, keepdim=True)
    z_max = torch.take_along_dim(z, i_max, dim=dim)                          # Max logit.
    is_max = torch.zeros_like(z, dtype=torch.bool).scatter_(dim, i_max, True)  # Max mask.
    shifted = torch.where(is_max, 0.0, z - z_max)                            # Exact 0 at argmax.
    others = torch.where(is_max, -torch.inf, shifted)                        # Exclude argmax from sum.
    m = -torch.logsumexp(others, dim=dim, keepdim=True)                      # Logit margin.
    return shifted - F.softplus(-m)                                          # log_softmax(z).


_SOFTMAX_HOOKS = []


def softmax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically accurate softmax that avoids Softmax Collapse.

    Reference:
        "A Fix for Softmax and Cross-Entropy", Jonathan T. Barron et al.
        https://jonbarron.info/softmaxfix/
    """
    out = torch.exp(log_softmax(z, dim=dim))
    if _SOFTMAX_HOOKS:
        for hook in _SOFTMAX_HOOKS:
            hook(out, dim)
    return out


def gumbel_softmax(
    logits: torch.Tensor,
    tau: float = 1.0,
    hard: bool = False,
    eps: float = 1e-10,
    dim: int = -1,
) -> torch.Tensor:
    """Numerically accurate Gumbel-Softmax that avoids Softmax Collapse in its backward pass.

    Drop-in replacement for ``torch.nn.functional.gumbel_softmax`` using the fixed ``softmax``.
    """
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0, 1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits, tau)
    y_soft = softmax(gumbels, dim=dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


def cross_entropy(z: torch.Tensor, t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically accurate hard-target cross-entropy that avoids Softmax Collapse.

    Reference:
        "A Fix for Softmax and Cross-Entropy", Jonathan T. Barron et al.
        https://jonbarron.info/softmaxfix/
    """
    t_idx = (t.unsqueeze(dim) if t.ndim == z.ndim - 1 else t).long()
    z_t = torch.take_along_dim(z, t_idx, dim=dim)                            # Target logit.
    is_t = torch.zeros_like(z, dtype=torch.bool).scatter_(dim, t_idx, True)  # Target mask.
    others = torch.where(is_t, -torch.inf, z - z_t)                          # Exclude target from sum.
    m = -torch.logsumexp(others, dim=dim)                                    # Logit margin.
    return F.softplus(-m)                                                    # -log_softmax(z)_t.


def soft_target_cross_entropy(z_S: torch.Tensor, z_T: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically accurate soft-target cross-entropy that avoids Softmax Collapse and Softmax Cancellation.

    Reference:
        "A Fix for Softmax and Cross-Entropy", Jonathan T. Barron et al.
        https://jonbarron.info/softmaxfix/
    """
    log_q = log_softmax(z_S, dim=dim)
    q, p = torch.exp(log_q), softmax(z_T, dim=dim)
    delta = z_S - z_T                                                        # Logit difference.
    d = delta - torch.sum(p * delta, dim=dim, keepdim=True)                  # Center under p.
    s = d - torch.log1p(torch.sum(p * torch.expm1(d), dim=dim, keepdim=True))  # log(q / p).
    u = torch.where(torch.abs(s) < 1, p * torch.expm1(s), q - p)             # Difference q - p.
    g = u - q * torch.sum(u, dim=dim, keepdim=True)                          # Ensure sum(g) = 0.
    loss = -torch.sum(p * log_q.detach(), dim=dim)                           # Forward loss.
    return loss + torch.sum(g.detach() * (z_S - z_S.detach()), dim=dim)      # Custom gradient g.


class LogSoftmax(nn.Module):
    """Drop-in ``nn.Module`` replacement for ``nn.LogSoftmax`` that avoids Softmax Collapse."""

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return log_softmax(z, dim=self.dim)


class Softmax(nn.Module):
    """Drop-in ``nn.Module`` replacement for ``nn.Softmax`` that avoids Softmax Collapse."""

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return softmax(z, dim=self.dim)


class GumbelSoftmax(nn.Module):
    """``nn.Module`` wrapper for ``gumbel_softmax`` that avoids Softmax Collapse."""

    def __init__(self, tau: float = 1.0, hard: bool = False, eps: float = 1e-10, dim: int = -1) -> None:
        super().__init__()
        self.tau = tau
        self.hard = hard
        self.eps = eps
        self.dim = dim

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return gumbel_softmax(logits, tau=self.tau, hard=self.hard, eps=self.eps, dim=self.dim)

