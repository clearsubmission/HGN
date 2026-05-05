import argparse, json, os, sys, time
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import (get_split_cifar100, get_split_cifar10,
                      get_split_mnist, get_permuted_mnist, get_split_stl10,
                      get_split_fashionmnist, get_split_tinyimagenet)
from model import resnet18_hgn, resnet18_baseline, reset_all_fatigue, count_dead_neurons
from hgn import HGNLinear
from si import SI
from lwf import LwF


class EWC:
    def __init__(self, model, dataloader, device, lam=1.0, is_mnist=False):
        self.lam    = lam
        self.params = {n: p.clone().detach()
                       for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher(model, dataloader, device, is_mnist=is_mnist)

    def _compute_fisher(self, model, dataloader, device, is_mnist=False):
        fisher = {n: torch.zeros_like(p)
                  for n, p in model.named_parameters() if p.requires_grad}
        model.eval()
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            if is_mnist:
                x = x.view(x.size(0), -1)
            model.zero_grad()
            nn.CrossEntropyLoss()(model(x), y).backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
        return {n: f / len(dataloader) for n, f in fisher.items()}

    def penalty(self, model):
        loss = 0.0
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return self.lam / 2 * loss


@torch.no_grad()
def evaluate(model, loader, device, is_mnist=False, task_classes=None):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if is_mnist:
            x = x.view(x.size(0), -1)
        logits = model(x)
        if task_classes is not None:
            # Restrict to task classes only (task-aware evaluation)
            mask = torch.tensor(task_classes, device=device)
            logits = logits[:, mask]
            y_local = torch.zeros_like(y)
            for i, c in enumerate(task_classes):
                y_local[y == c] = i
            preds = logits.argmax(1)
            correct += (preds == y_local).sum().item()
        else:
            correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return correct / total if total > 0 else 0.0


def get_mlp(use_hgn, lam, alpha):
    if use_hgn:
        return nn.Sequential(
            HGNLinear(784, 400, lam=lam, alpha=alpha),
            HGNLinear(400, 400, lam=lam, alpha=alpha),
            nn.Linear(400, 10),
        )
    return nn.Sequential(
        nn.Linear(784, 400), nn.ReLU(),
        nn.Linear(400, 400), nn.ReLU(),
        nn.Linear(400, 10),
    )

def train_continual(model, task_loaders, device, args, is_mnist=False, num_classes=10):
    optimizer   = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion   = nn.CrossEntropyLoss()
    n_tasks     = len(task_loaders)
    use_ewc     = args.model in ("ewc", "hgn_ewc")
    use_hgn     = args.model in ("hgn", "hgn_ewc", "hgn_si", "hgn_lwf")
    use_si      = args.model in ("si","hgn_si")
    use_lwf     = args.model in ("lwf","hgn_lwf")
    ewc_tasks   = []
    si = SI(model) if use_si else None
    lwf = LwF(model, device=device) if use_lwf else None
    acc_matrix  = torch.zeros(n_tasks, n_tasks)
    dead_pcts   = []
    h_stats_log = []

    for task_id, (train_loader, test_loader) in enumerate(task_loaders):
        print(f"\n[Task {task_id+1}/{n_tasks}]", flush=True)

        if task_id > 0:
            if use_hgn:
                h_stats_log.append(reset_all_fatigue(model))
            if use_lwf:
                lwf.update_old_model()
            if use_ewc:
                ewc_lam = 1.0 / max(num_classes, 1)
                ewc_tasks.append(EWC(model, task_loaders[task_id-1][0], device, lam=ewc_lam, is_mnist=is_mnist))

        model.train()
        for epoch in range(args.epochs_per_task):
            total_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                if is_mnist:
                    x = x.view(x.size(0), -1)
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                if use_lwf:
                    loss = loss + lwf.loss(x)
                if use_si:
                    loss = loss + si.penalty()
                for ewc in ewc_tasks:
                    loss = loss + ewc.penalty(model)
                loss.backward()
                if use_si:
                    si.accumulate()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 5 == 0 or args.epochs_per_task <= 2:
                print(f"  Epoch {epoch+1:3d}  loss={total_loss/len(train_loader):.4f}",
                      flush=True)

        if use_si:
            si.update_omega()

        model.eval()
        classes_per_task = num_classes // len(task_loaders)
        for j in range(task_id + 1):
            task_cls = list(range(j * classes_per_task, (j+1) * classes_per_task))
            use_task_mask = (not is_mnist) and (not args.shared_head)
            acc = evaluate(model, task_loaders[j][1], device, is_mnist,
                          task_classes=task_cls if use_task_mask else None)
            acc_matrix[task_id, j] = acc
            print(f"  Task {j+1} acc: {acc*100:.2f}%", flush=True)

        dead = count_dead_neurons(model, test_loader, device, is_mnist=is_mnist)
        dead_pcts.append(dead)
        print(f"  Dead neurons: {dead*100:.1f}%", flush=True)

    T       = n_tasks - 1
    avg_acc = acc_matrix[T, :].mean().item()
    bwt     = float(sum((acc_matrix[T,j]-acc_matrix[j,j]).item()
                        for j in range(T)) / T) if T > 0 else 0.0
    fwt     = float(sum(acc_matrix[j-1,j].item()
                        for j in range(1, n_tasks)) / (n_tasks-1)) if n_tasks > 1 else 0.0
    return {"avg_acc": avg_acc, "bwt": bwt, "fwt": fwt,
            "dead_pcts": dead_pcts, "acc_matrix": acc_matrix.tolist(),
            "h_stats_log": h_stats_log}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp",             default="e1")
    p.add_argument("--model",           default="hgn",
                   choices=["hgn","baseline","ewc","hgn_ewc","si","lwf","hgn_si","hgn_lwf"])
    p.add_argument("--dataset",         default="split_cifar100",
                   choices=["split_cifar100","split_cifar10",
                            "split_mnist","permuted_mnist","split_stl10","split_fashionmnist","split_tinyimagenet"])
    p.add_argument("--lam",             type=float, default=0.7)
    p.add_argument("--alpha",           type=float, default=1.2)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--epochs_per_task", type=int,   default=10)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--data_dir",        default="data")
    p.add_argument("--output_dir",      default="results/run")
    p.add_argument("--pretrained",       action="store_true", default=False)
    p.add_argument("--shared_head",        action="store_true", default=False)
    p.add_argument("--layer_mode",      default="all",
                   choices=["all","early","middle","late","none"])
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_mnist = args.dataset in ("split_mnist", "permuted_mnist")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Device: {device} | Model: {args.model} | "
          f"lam={args.lam} alpha={args.alpha} seed={args.seed}", flush=True)

    if args.dataset == "split_tinyimagenet":
        task_loaders = get_split_tinyimagenet(args.data_dir)
        num_classes  = 200
    elif args.dataset == "split_stl10":
        task_loaders = get_split_stl10(args.data_dir)
        num_classes  = 10
    elif args.dataset == "split_cifar100":
        task_loaders = get_split_cifar100(args.data_dir)
        num_classes  = 100
    elif args.dataset == "split_cifar10":
        task_loaders = get_split_cifar10(args.data_dir)
        num_classes  = 10
    elif args.dataset == "split_mnist":
        task_loaders = get_split_mnist(args.data_dir)
        num_classes  = 10
    else:
        task_loaders = get_permuted_mnist(args.data_dir, seed=args.seed)
        num_classes  = 10

    if is_mnist:
        model = get_mlp(use_hgn=(args.model != "baseline"),
                        lam=args.lam, alpha=args.alpha)
    elif args.model in ("hgn", "hgn_ewc", "hgn_si", "hgn_lwf"):
        model = resnet18_hgn(num_classes=num_classes,
                             lam=args.lam, alpha=args.alpha,
                             pretrained=args.pretrained)
    else:
        model = resnet18_baseline(num_classes=num_classes,
                                  pretrained=args.pretrained)

    model   = model.to(device)
    t0      = time.time()
    results = train_continual(model, task_loaders, device, args, is_mnist, num_classes=num_classes)
    results["elapsed_sec"] = time.time() - t0
    results["args"]        = vars(args)

    out = os.path.join(args.output_dir, "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Avg Accuracy : {results['avg_acc']*100:.2f}%")
    print(f"BWT          : {results['bwt']*100:.2f}%")
    print(f"Dead neurons : {results['dead_pcts'][-1]*100:.1f}%")
    print(f"Time         : {results['elapsed_sec']/60:.1f} min")
    print(f"Saved -> {out}")

if __name__ == "__main__":
    main()