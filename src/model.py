import torch
import torch.nn as nn
import torchvision.models as tv
from hgn import HGNConv2d


def _replace_convs(module, lam, alpha, learn_params):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d) and child.kernel_size == (3, 3):
            new = HGNConv2d(
                child.in_channels, child.out_channels,
                kernel_size=3, stride=child.stride[0],
                padding=child.padding[0],
                bias=(child.bias is not None),
                lam=lam, alpha=alpha, learn_params=learn_params,
            )
            new.conv.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                new.conv.bias.data.copy_(child.bias.data)
            setattr(module, name, new)
        else:
            _replace_convs(child, lam, alpha, learn_params)


def resnet18_hgn(num_classes=100, lam=0.7, alpha=1.2, learn_params=True, pretrained=False):
    if pretrained:
        model = tv.resnet18(weights=tv.ResNet18_Weights.DEFAULT)
        model.fc = __import__("torch.nn", fromlist=["Linear"]).Linear(512, num_classes)
    else:
        model = tv.resnet18(weights=None, num_classes=num_classes)
    _replace_convs(model, lam=lam, alpha=alpha, learn_params=learn_params)
    return model


def resnet18_baseline(num_classes=100, pretrained=False):
    if pretrained:
        model = tv.resnet18(weights=tv.ResNet18_Weights.DEFAULT)
        model.fc = __import__("torch.nn", fromlist=["Linear"]).Linear(512, num_classes)
        return model
    return tv.resnet18(weights=None, num_classes=num_classes)


def fold_all_fatigue(model):
    """Replace HGNConv2d/HGNLinear with plain nn modules after folding fatigue into bias."""
    import torch.nn as nn
    from hgn import HGNConv2d, HGNLinear

    def _replace(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, HGNConv2d):
                child.fold_fatigue()
                device = child.conv.weight.device
                plain = nn.Conv2d(
                    child.conv.in_channels, child.conv.out_channels,
                    kernel_size=child.conv.kernel_size,
                    stride=child.conv.stride,
                    padding=child.conv.padding,
                    bias=(child.conv.bias is not None)
                ).to(device)
                plain.weight.data.copy_(child.conv.weight.data)
                if child.conv.bias is not None:
                    plain.bias.data.copy_(child.conv.bias.data)
                setattr(parent, name, plain)
            elif isinstance(child, HGNLinear):
                child.fold_fatigue()
                device = child.linear.weight.device
                plain = nn.Linear(
                    child.linear.in_features, child.linear.out_features,
                    bias=(child.linear.bias is not None)
                ).to(device)
                plain.weight.data.copy_(child.linear.weight.data)
                if child.linear.bias is not None:
                    plain.bias.data.copy_(child.linear.bias.data)
                setattr(parent, name, plain)
            else:
                _replace(child)
    _replace(model)

def reset_all_fatigue(model):
    stats = {}
    for name, m in model.named_modules():
        if isinstance(m, HGNConv2d):
            stats[name] = m.reset_fatigue()
    return stats


def count_dead_neurons(model, dataloader, device, threshold=0.01, is_mnist=False):
    """
    Count dead neurons using the INNER conv/linear layer for HGNConv2d,
    so we measure the raw pre-fatigue output, not the suppressed output.
    """
    model.eval()
    act_sum, act_cnt, hooks = {}, {}, []

    def make_hook(name):
        def hook(mod, inp, out):
            val = out.detach().abs()
            if val.dim() == 4:
                val = val.mean(dim=(0, 2, 3))
            elif val.dim() == 2:
                val = val.mean(dim=0)
            act_sum[name] = act_sum.get(name, 0) + val.cpu()
            act_cnt[name] = act_cnt.get(name, 0) + 1
        return hook

    for name, m in model.named_modules():
        if isinstance(m, HGNConv2d):
            # Hook the INNER conv — measures pre-fatigue output
            hooks.append(m.conv.register_forward_hook(make_hook(name + ".conv")))
        elif isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(make_hook(name)))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            if is_mnist:
                x = x.view(x.size(0), -1)
            model(x)

    for h in hooks:
        h.remove()

    total = dead = 0
    for name in act_sum:
        avg    = act_sum[name] / act_cnt[name]
        total += avg.numel()
        dead  += (avg < threshold).sum().item()

    return dead / total if total > 0 else 0.0