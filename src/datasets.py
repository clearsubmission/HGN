import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)
CIFAR10_MEAN  = (0.4914, 0.4822, 0.4465)
CIFAR10_STD   = (0.2023, 0.1994, 0.2010)


def get_split_cifar100(data_dir="data", n_tasks=20, batch_size=256, num_workers=0):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    train_full = datasets.CIFAR100(data_dir, train=True,  download=False, transform=train_tf)
    test_full  = datasets.CIFAR100(data_dir, train=False, download=False, transform=test_tf)
    classes_per_task = 100 // n_tasks
    import torch as _torch
    train_targets = _torch.tensor(train_full.targets)
    test_targets  = _torch.tensor(test_full.targets)
    loaders = []
    for t in range(n_tasks):
        task_classes = list(range(t * classes_per_task, (t+1) * classes_per_task))
        train_idx = [i for i,y in enumerate(train_targets.tolist()) if y in task_classes]
        test_idx  = [i for i,y in enumerate(test_targets.tolist())  if y in task_classes]
        loaders.append((
            DataLoader(Subset(train_full, train_idx), batch_size=batch_size,
                       shuffle=True,  num_workers=num_workers, pin_memory=True),
            DataLoader(Subset(test_full,  test_idx),  batch_size=512,
                       shuffle=False, num_workers=num_workers, pin_memory=True),
        ))
    return loaders


def get_split_cifar10(data_dir="data", n_tasks=5, batch_size=256, num_workers=0):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_full = datasets.CIFAR10(data_dir, train=True,  download=False, transform=train_tf)
    test_full  = datasets.CIFAR10(data_dir, train=False, download=False, transform=test_tf)
    classes_per_task = 10 // n_tasks
    import torch as _torch
    train_targets = _torch.tensor(train_full.targets)
    test_targets  = _torch.tensor(test_full.targets)
    loaders = []
    for t in range(n_tasks):
        task_classes = list(range(t * classes_per_task, (t+1) * classes_per_task))
        train_idx = [i for i,y in enumerate(train_targets.tolist()) if y in task_classes]
        test_idx  = [i for i,y in enumerate(test_targets.tolist())  if y in task_classes]
        loaders.append((
            DataLoader(Subset(train_full, train_idx), batch_size=batch_size,
                       shuffle=True,  num_workers=num_workers, pin_memory=True),
            DataLoader(Subset(test_full,  test_idx),  batch_size=512,
                       shuffle=False, num_workers=num_workers, pin_memory=True),
        ))
    return loaders


def get_split_mnist(data_dir="data", n_tasks=5, batch_size=256, num_workers=0):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_full = datasets.MNIST(data_dir, train=True,  download=False, transform=tf)
    test_full  = datasets.MNIST(data_dir, train=False, download=False, transform=tf)
    classes_per_task = 10 // n_tasks
    loaders = []
    for t in range(n_tasks):
        task_classes = list(range(t * classes_per_task, (t+1) * classes_per_task))
        train_idx = [i for i,(_, y) in enumerate(train_full) if y in task_classes]
        test_idx  = [i for i,(_, y) in enumerate(test_full)  if y in task_classes]
        loaders.append((
            DataLoader(Subset(train_full, train_idx), batch_size=batch_size,
                       shuffle=True,  num_workers=num_workers),
            DataLoader(Subset(test_full,  test_idx),  batch_size=512,
                       shuffle=False, num_workers=num_workers),
        ))
    return loaders


def get_permuted_mnist(data_dir="data", n_tasks=10, batch_size=256, num_workers=0, seed=42):
    rng = np.random.RandomState(seed)
    loaders = []
    for t in range(n_tasks):
        perm = torch.from_numpy(rng.permutation(784))
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x, p=perm: x.view(-1)[p].view(1, 28, 28)),
        ])
        train_d = datasets.MNIST(data_dir, train=True,  download=False, transform=tf)
        test_d  = datasets.MNIST(data_dir, train=False, download=False, transform=tf)
        loaders.append((
            DataLoader(train_d, batch_size=batch_size, shuffle=True,  num_workers=num_workers),
            DataLoader(test_d,  batch_size=512,        shuffle=False, num_workers=num_workers),
        ))
    return loaders

def get_split_stl10(data_dir="data", n_tasks=5, batch_size=128, num_workers=0):
    """Split-STL10: 5 tasks x 2 classes. 96x96 RGB — harder than MNIST, no Toronto needed."""
    import torchvision.transforms as T
    train_tf = T.Compose([
        T.Resize(32),
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4467,0.4398,0.4066),(0.2603,0.2566,0.2713)),
    ])
    test_tf = T.Compose([
        T.Resize(32),
        T.ToTensor(),
        T.Normalize((0.4467,0.4398,0.4066),(0.2603,0.2566,0.2713)),
    ])
    from torchvision import datasets
    from torch.utils.data import DataLoader, Subset
    train_full = datasets.STL10(data_dir, split="train", download=False, transform=train_tf)
    test_full  = datasets.STL10(data_dir, split="test",  download=False, transform=test_tf)
    classes_per_task = 10 // n_tasks
    loaders = []
    for t in range(n_tasks):
        task_classes = list(range(t * classes_per_task, (t+1) * classes_per_task))
        train_idx = [i for i,(_, y) in enumerate(train_full) if y in task_classes]
        test_idx  = [i for i,(_, y) in enumerate(test_full)  if y in task_classes]
        loaders.append((
            DataLoader(Subset(train_full, train_idx), batch_size=batch_size,
                       shuffle=True,  num_workers=num_workers, pin_memory=True),
            DataLoader(Subset(test_full,  test_idx),  batch_size=256,
                       shuffle=False, num_workers=num_workers, pin_memory=True),
        ))
    return loaders


def get_split_tinyimagenet(data_dir="data", n_tasks=20, batch_size=128, num_workers=0):
    """Split Tiny-ImageNet: 200 classes -> 20 tasks x 10 classes. 64x64 RGB."""
    import torchvision.transforms as T
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader, Subset
    import os

    train_tf = T.Compose([
        T.Resize(72),
        T.RandomCrop(64, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4802,0.4481,0.3975),(0.2770,0.2691,0.2821)),
    ])
    test_tf = T.Compose([
        T.Resize(72),
        T.CenterCrop(64),
        T.ToTensor(),
        T.Normalize((0.4802,0.4481,0.3975),(0.2770,0.2691,0.2821)),
    ])

    train_dir = os.path.join(data_dir, "tiny-imagenet-200", "train")
    val_dir   = os.path.join(data_dir, "tiny-imagenet-200", "val")

    train_full = ImageFolder(train_dir, transform=train_tf)
    test_full  = ImageFolder(val_dir,   transform=test_tf)

    import torch as _torch
    classes_per_task = len(train_full.classes) // n_tasks
    # Use .targets for fast indexing (no image loading)
    train_targets = _torch.tensor(train_full.targets)
    test_targets  = _torch.tensor(test_full.targets)
    loaders = []
    for t in range(n_tasks):
        task_classes = list(range(t * classes_per_task, (t+1) * classes_per_task))
        task_t = _torch.tensor(task_classes)
        train_idx = [i for i, y in enumerate(train_targets.tolist()) if y in task_classes]
        test_idx  = [i for i, y in enumerate(test_targets.tolist())  if y in task_classes]
        loaders.append((
            DataLoader(Subset(train_full, train_idx), batch_size=batch_size,
                       shuffle=True,  num_workers=num_workers, pin_memory=True),
            DataLoader(Subset(test_full,  test_idx),  batch_size=256,
                       shuffle=False, num_workers=num_workers, pin_memory=True),
        ))
    return loaders


def get_split_fashionmnist(data_dir="data", n_tasks=5, batch_size=256, num_workers=0):
    import torchvision.transforms as T
    from torchvision import datasets
    from torch.utils.data import DataLoader, Subset
    tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.2860,), (0.3530,)),
    ])
    train_full = datasets.FashionMNIST(data_dir, train=True,  download=False, transform=tf)
    test_full  = datasets.FashionMNIST(data_dir, train=False, download=False, transform=tf)
    classes_per_task = 10 // n_tasks
    loaders = []
    for t in range(n_tasks):
        task_classes = list(range(t * classes_per_task, (t+1) * classes_per_task))
        train_idx = [i for i,(_, y) in enumerate(train_full) if y in task_classes]
        test_idx  = [i for i,(_, y) in enumerate(test_full)  if y in task_classes]
        loaders.append((
            DataLoader(Subset(train_full, train_idx), batch_size=batch_size,
                       shuffle=True,  num_workers=num_workers),
            DataLoader(Subset(test_full,  test_idx),  batch_size=512,
                       shuffle=False, num_workers=num_workers),
        ))
    return loaders
