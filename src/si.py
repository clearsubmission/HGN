import torch

class SI:
    def __init__(self, model, lam=0.01, eps=1e-8):
        self.model = model
        self.lam = lam
        self.eps = eps
        self.prev_params = {}
        self.omega = {}
        self.w = {}

        for name, p in model.named_parameters():
            if p.requires_grad:
                self.prev_params[name] = p.detach().clone()
                self.omega[name] = torch.zeros_like(p)
                self.w[name] = torch.zeros_like(p)

    def accumulate(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                delta = p.detach() - self.prev_params[name]
                self.w[name] += -p.grad.detach() * delta

    def update_omega(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                delta = p.detach() - self.prev_params[name]
                self.omega[name] += self.w[name] / (delta.pow(2) + self.eps)
                self.prev_params[name] = p.detach().clone()
                self.w[name].zero_()

    def penalty(self):
        loss = 0.0
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                loss = loss + (self.omega[name] * (p - self.prev_params[name]).pow(2)).sum()
        return self.lam * loss
