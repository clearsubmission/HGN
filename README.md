# Hysteresis-Gated Networks (HGN)

Official implementation of "Hysteresis-Gated Networks: Mitigating Catastrophic Forgetting Through Adaptive Neuron Fatigue" (NeurIPS 2026 submission).

## Requirements
torch>=2.0

torchvision>=0.15

numpy
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
for ALPHA in 0.0 0.5 1.0 1.2 1.5 2.0 3.0; do
    python src/train.py --model hgn --dataset split_cifar100 --alpha $ALPHA --seed 42
done

# Lambda ablation
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

## Key results (Split-CIFAR-100, pretrained ResNet-18, 3 seeds)
| Method | Avg Acc | BWT |
|--------|---------|-----|
| Fine-tuning | 20.23% | -25.03% |
| HGN | 19.76% | -1.07% |
| HGN+EWC | 20.14% | -0.07% |
| HGN+SI | 19.99% | -0.13% |
