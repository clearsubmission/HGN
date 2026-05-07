import torch
import torch.nn.functional as F
import numpy as np


class DERppBuffer:
    """
    Dark Experience Replay++ buffer.
    Stores (x, y, logits) tuples from past tasks.
    Reservoir sampling to maintain fixed buffer size.
    """
    def __init__(self, buffer_size=200, device='cuda'):
        self.buffer_size = buffer_size
        self.device = device
        self.x = []
        self.y = []
        self.logits = []
        self.n_seen = 0

    def add(self, x, y, logits):
        """Add examples using reservoir sampling."""
        for i in range(len(x)):
            if len(self.x) < self.buffer_size:
                self.x.append(x[i].detach().cpu())
                self.y.append(y[i].detach().cpu())
                self.logits.append(logits[i].detach().cpu())
            else:
                idx = np.random.randint(0, self.n_seen + i + 1)
                if idx < self.buffer_size:
                    self.x[idx] = x[i].detach().cpu()
                    self.y[idx] = y[i].detach().cpu()
                    self.logits[idx] = logits[i].detach().cpu()
        self.n_seen += len(x)

    def sample(self, n):
        """Sample n examples from buffer."""
        n = min(n, len(self.x))
        idx = np.random.choice(len(self.x), n, replace=False)
        x = torch.stack([self.x[i] for i in idx]).to(self.device)
        y = torch.stack([self.y[i] for i in idx]).to(self.device)
        logits = torch.stack([self.logits[i] for i in idx]).to(self.device)
        return x, y, logits

    def __len__(self):
        return len(self.x)


def derpp_loss(model, x_buf, y_buf, logits_buf, alpha=0.2, beta=1.0):
    """
    DER++ loss on buffer samples.
    alpha: weight for MSE distillation loss (knowledge distillation)
    beta:  weight for CE replay loss (label replay)
    """
    if len(x_buf) == 0:
        return torch.tensor(0.0, device=x_buf.device if hasattr(x_buf, 'device') else 'cuda')

    new_logits = model(x_buf)

    # MSE between current logits and stored logits (dark experience)
    loss_mse = F.mse_loss(new_logits, logits_buf)

    # Cross entropy on stored labels
    loss_ce = F.cross_entropy(new_logits, y_buf)

    return alpha * loss_mse + beta * loss_ce
