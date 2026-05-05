import torch
import torch.nn as nn
import torch.nn.functional as F


class HGNLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 lam=0.7, alpha=1.2, learn_params=True, activation="relu"):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.activation = activation
        if learn_params:
            self.lam   = nn.Parameter(torch.tensor(float(lam)))
            self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        else:
            self.register_buffer("lam",   torch.tensor(float(lam)))
            self.register_buffer("alpha", torch.tensor(float(alpha)))
        self.register_buffer("h", None)

    def reset_fatigue(self):
        stats = {}
        if self.h is not None:
            stats = {"mean": self.h.mean().item(),
                     "max":  self.h.max().item(),
                     "dead_pct": (self.h > 0.99).float().mean().item()}
        self.h = None
        return stats

    def fold_fatigue(self):
        """Fold steady-state fatigue into bias for zero-overhead inference."""
        if self.h is not None and self.linear.bias is not None:
            alpha = self.alpha.clamp(min=0.0)
            with torch.no_grad():
                self.linear.bias.data -= alpha * self.h.squeeze()
        self.h = None
        self.forward = self._forward_folded

    def _forward_folded(self, x):
        return F.relu(self.linear(x))

    def forward(self, x):
        z     = self.linear(x)
        lam   = self.lam.clamp(0.01, 0.99)
        alpha = self.alpha.clamp(min=0.0)
        if self.h is None or self.h.shape != z.shape:
            self.h = torch.zeros_like(z)
        with torch.no_grad():
            self.h = lam * self.h + (1 - lam) * torch.sigmoid(z)
        z_tilde = z - alpha * self.h
        if self.activation == "relu": return F.relu(z_tilde)
        if self.activation == "gelu": return F.gelu(z_tilde)
        return F.relu(z_tilde)


class HGNConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True,
                 lam=0.7, alpha=1.2, learn_params=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias)
        if learn_params:
            self.lam   = nn.Parameter(torch.tensor(float(lam)))
            self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        else:
            self.register_buffer("lam",   torch.tensor(float(lam)))
            self.register_buffer("alpha", torch.tensor(float(alpha)))
        self.register_buffer("h", None)

    def reset_fatigue(self):
        stats = {}
        if self.h is not None:
            stats = {"mean": self.h.mean().item(),
                     "max":  self.h.max().item(),
                     "dead_pct": (self.h > 0.99).float().mean().item()}
        self.h = None
        return stats

    def fold_fatigue(self):
        """Fold steady-state fatigue into bias for zero-overhead inference."""
        if self.h is not None and self.conv.bias is not None:
            alpha = self.alpha.clamp(min=0.0)
            with torch.no_grad():
                self.conv.bias.data -= alpha * self.h.squeeze()
        self.h = None
        self.forward = self._forward_folded

    def _forward_folded(self, x):
        return F.relu(self.conv(x))

    def forward(self, x):
        z     = self.conv(x)
        lam   = self.lam.clamp(0.01, 0.99)
        alpha = self.alpha.clamp(min=0.0)
        if self.h is None or self.h.shape[1] != z.shape[1]:
            self.h = torch.zeros(1, z.shape[1], 1, 1,
                                 device=z.device, dtype=z.dtype)
        with torch.no_grad():
            self.h = lam * self.h + (1 - lam) * torch.sigmoid(z).mean(dim=(0,2,3), keepdim=True)
        return F.relu(z - alpha * self.h)
