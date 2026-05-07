import copy
import torch
import torch.nn.functional as F

class LwF:
    def __init__(self, model, temperature=2.0, alpha=0.5, device="cuda"):
        self.model = model
        self.old_model = None
        self.temperature = temperature
        self.alpha = alpha
        self.device = device

    def update_old_model(self):
        self.old_model = copy.deepcopy(self.model).to(self.device)
        self.old_model.eval()
        for p in self.old_model.parameters():
            p.requires_grad = False

    def loss(self, x):
        if self.old_model is None:
            return torch.tensor(0.0, device=x.device)

        T = self.temperature

        with torch.no_grad():
            old_logits = self.old_model(x)
        new_logits = self.model(x)

        soft_old = F.softmax(old_logits / T, dim=1)
        log_soft_new = F.log_softmax(new_logits / T, dim=1)

        return self.alpha * (T * T) * F.kl_div(
            log_soft_new,
            soft_old,
            reduction="batchmean"
        )
