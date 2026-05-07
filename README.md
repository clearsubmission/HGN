# Hysteresis-Gated Networks (HGN)

Official implementation of "Hysteresis-Gated Networks: Mitigating Catastrophic Forgetting Through Adaptive Neuron Fatigue".

# Overview

HGN is a biologically-inspired continual learning mechanism that prevents catastrophic forgetting through adaptive neuron fatigue. Each neuron maintains a scalar fatigue state that rises with repeated activation and decays at task boundaries, suppressing overactive neurons and recruiting underutilized ones. HGN introduces fewer than 0.001% additional parameters and incurs zero inference overhead.


## Requirements
```bash
torch>=2.0
torchvision>=0.15
numpy
```
## Datasets
- Split-CIFAR-100: downloaded automatically via torchvision
- Split-TinyImageNet: download from http://cs231n.stanford.edu/tiny-imagenet-200.zip and place in `data/`

## Running experiments
### Main result
```bash
# HGN + EWC on Split-CIFAR-100 with pretrained ResNet-18
python src/train.py --model hgn_ewc --dataset split_cifar100 --pretrained --seed 42

# HGN + SI on Split-CIFAR-100
python src/train.py --model hgn_si --dataset split_cifar100 --pretrained --seed 42

# TinyImageNet
python src/train.py --model hgn_ewc --dataset split_tinyimagenet --pretrained --seed 42
```

### Ablation studies
```bash
for ALPHA in 0.0 0.5 1.0 1.2 1.5 2.0 3.0; do
    python src/train.py --model hgn --dataset split_cifar100 --alpha $ALPHA --seed 42
done
```
## Lambda ablation
```bash
for LAM in 0.1 0.3 0.5 0.7 0.9 0.95 0.99; do
    python src/train.py --model hgn --dataset split_cifar100 --lam $LAM --seed 42
done
```

## Models
- `baseline` — sequential fine-tuning, no forgetting mitigation
- `ewc` — Elastic Weight Consolidation (Kirkpatrick et al. 2017)
- `si` — Synaptic Intelligence (Zenke et al. 2017)
- `lwf` — Learning without Forgetting (Li & Hoiem 2017)
- `hgn` — Hysteresis-Gated Network (ours)
- `hgn_ewc` — HGN + EWC (ours)
- `hgn_si` — HGN + SI (ours)
- `hgn_lwf` — HGN + LwF (ours)

## Results
See `results/final_results_all.csv` for all experimental results.


## Key Results

### Split-CIFAR-100 (20 tasks, pretrained ResNet-18, 3 seeds)

| Method       | Avg Acc ↑ | ±std | BWT ↑   | ±std |
|--------------|-----------|------|---------|------|
| Fine-tuning  | 20.23%    | 0.17 | -25.03% | 0.92 |
| EWC          | 19.51%    | 0.39 | -29.74% | 4.21 |
| SI           | 19.87%    | 0.31 | -5.96%  | 0.50 |
| LwF          | 20.41%    | 0.52 | -29.52% | 3.22 |
| HGN (ours)   | 19.76%    | 0.21 | -1.07%  | 0.41 |
| HGN+EWC (ours)| 20.14%   | 0.21 | -0.07%  | 0.30 |
| HGN+SI (ours)| 19.99%    | 0.21 | -0.13%  | 0.39 |
| HGN+LwF (ours)| 20.08%   | 0.43 | -1.60%  | 1.14 |

### Split-TinyImageNet (20 tasks, pretrained ResNet-18, 3 seeds)

| Method        | Avg Acc ↑ | ±std | BWT ↑   | ±std |
|---------------|-----------|------|---------|------|
| Fine-tuning   | 11.51%    | 0.82 | -32.83% | 2.11 |
| EWC           | 11.95%    | 0.43 | -34.73% | 0.80 |
| SI            | 10.02%    | 0.05 | -3.33%  | 0.26 |
| LwF           | 12.00%    | 0.74 | -33.97% | 2.46 |
| HGN (ours)    | 12.46%    | 0.49 | -32.92% | 2.99 |
| HGN+EWC (ours)| 10.43%    | 0.38 | -0.19%  | 0.40 |
| HGN+SI (ours) | 10.01%    | 0.15 | -0.35%  | 0.12 |
| HGN+LwF (ours)| 13.02%    | 0.31 | -34.65% | 1.83 |

**BWT = backward transfer. Higher (closer to 0) = less forgetting.**


## Method

HGN adds a fatigue gate to each layer:

h(t) = λ · h(t−1) + (1−λ) · σ(z(t−1))    # fatigue update
z̃(t) = z(t) − α · h(t)                     # suppressed pre-activation
a(t) = ReLU(z̃(t))                           # output

- **λ** — learnable decay parameter (default 0.7)
- **α** — learnable suppression strength (default 1.2)
- At test time: fold h* into bias → b* = b − α·h* → zero inference overhead

## Requirements

```bash
pip install torch>=2.0 torchvision>=0.15 numpy
```

Tested on Python 3.12, CUDA 12.2, NVIDIA H100.

## Datasets

**Split-CIFAR-100** — downloaded automatically via torchvision.

**Split-TinyImageNet** — download manually:
```bash
mkdir -p data
cd data
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
```

Then reorganize the validation set:
```python
import os, shutil
val_dir = "data/tiny-imagenet-200/val"
with open(f"{val_dir}/val_annotations.txt") as f:
    for line in f:
        fname, cls = line.strip().split("\t")[:2]
        os.makedirs(f"{val_dir}/{cls}", exist_ok=True)
        src = f"{val_dir}/images/{fname}"
        if os.path.exists(src):
            shutil.move(src, f"{val_dir}/{cls}/{fname}")
```

## Usage

### Basic training

```bash
# Fine-tuning baseline
python src/train.py --model baseline --dataset split_cifar100 --pretrained --seed 42

# HGN alone
python src/train.py --model hgn --dataset split_cifar100 --pretrained --seed 42

# HGN + EWC (best result)
python src/train.py --model hgn_ewc --dataset split_cifar100 --pretrained --seed 42

# HGN + SI (best BWT)
python src/train.py --model hgn_si --dataset split_cifar100 --pretrained --seed 42
```

### All models

```bash
for MODEL in baseline ewc si lwf hgn hgn_ewc hgn_si hgn_lwf; do
    python src/train.py \
        --model $MODEL \
        --dataset split_cifar100 \
        --pretrained \
        --seed 42 \
        --output_dir results/${MODEL}_s42
done
```

### TinyImageNet

```bash
python src/train.py \
    --model hgn_ewc \
    --dataset split_tinyimagenet \
    --pretrained \
    --seed 42 \
    --output_dir results/tinyimagenet_hgn_ewc_s42
```

### Ablation studies

```bash
# Alpha ablation (suppression strength)
for ALPHA in 0.0 0.5 1.0 1.2 1.5 2.0 3.0; do
    python src/train.py --model hgn --dataset split_cifar100 \
        --alpha $ALPHA --lam 0.7 --seed 42 \
        --output_dir results/ablation/alpha${ALPHA}_s42
done

# Lambda ablation (decay parameter)
for LAM in 0.1 0.3 0.5 0.7 0.9 0.95 0.99; do
    python src/train.py --model hgn --dataset split_cifar100 \
        --lam $LAM --alpha 1.2 --seed 42 \
        --output_dir results/ablation/lam${LAM}_s42
done
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `hgn` | Model: baseline, ewc, si, lwf, hgn, hgn_ewc, hgn_si, hgn_lwf |
| `--dataset` | `split_cifar100` | Dataset: split_cifar100, split_tinyimagenet, split_mnist, permuted_mnist |
| `--lam` | `0.7` | HGN fatigue decay λ |
| `--alpha` | `1.2` | HGN suppression strength α |
| `--lr` | `1e-3` | Learning rate |
| `--epochs_per_task` | `10` | Epochs per task |
| `--seed` | `42` | Random seed |
| `--pretrained` | `False` | Use ImageNet pretrained ResNet-18 |
| `--shared_head` | `False` | Use shared head evaluation |
| `--output_dir` | `results/run` | Output directory |

## Output

Each run saves a `results.json` file containing:
- `avg_acc` — average accuracy across all tasks
- `bwt` — backward transfer
- `fwt` — forward transfer
- `dead_pcts` — dead neuron percentage per task
- `acc_matrix` — full N×N accuracy matrix
- `args` — all hyperparameters

## Repository Structure

```bash
HGN
├── src/
│   ├── hgn.py          # HGNLinear and HGNConv2d modules
│   ├── model.py        # ResNet-18 with HGN, fold_all_fatigue
│   ├── train.py        # Training loop, EWC, evaluation
│   ├── datasets.py     # Split-CIFAR-100, TinyImageNet, MNIST loaders
│   ├── si.py           # Synaptic Intelligence
│   └── lwf.py          # Learning without Forgetting
├── results/
│   ├── final_results_all.csv       # All experimental results
│   └── final_results_summary.json  # Aggregated results with mean±std
└── README.md

```

## Reproducing Paper Results

To reproduce the main CIFAR-100 pretrained results (Table 1):

```bash
for MODEL in baseline ewc si lwf hgn hgn_ewc hgn_si hgn_lwf; do
for SEED in 42 123 456; do
    python src/train.py \
        --model $MODEL \
        --dataset split_cifar100 \
        --pretrained \
        --lam 0.7 --alpha 1.2 \
        --epochs_per_task 10 \
        --seed $SEED \
        --output_dir results/main/${MODEL}_s${SEED}
done
done
```

## Citation



## License



# Push to GitHub
```bash
git add README.md
git commit -m "Add complete README"
git push
```
