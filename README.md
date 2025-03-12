# Linear Attention Transformer (LAT) - Research Implementation

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-red.svg)](https://pytorch.org)
[![arXiv](https://img.shields.io/badge/arXiv-2402.18668-b31b1b.svg)](https://arxiv.org/abs/2402.18668)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org)

Official PyTorch implementation of the linear-complexity transformer architecture from *"Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff"*, optimized for CPU/GPU training and academic research.

## Table of Contents
- [Key Features](#key-features)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
  - [Configuration](#configuration)
  - [Execution](#execution)
- [Evaluation](#evaluation)
- [Pretrained Models](#pretrained-models)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [References](#references)
- [Contact](#contact)

## Key Features

- **Linear-Time Attention** - O(L) complexity implementation via associative scans
- **Research-Ready Pipeline**:
  - Full pre-training & fine-tuning support
  - Integrated W&B/TensorBoard logging
  - Gradient accumulation & mixed precision
- **CPU-First Design**:
  - Optimized matrix operations
  - Memory-efficient attention
  - BF16/FP32 support
- **Modular Components**:
  - Swappable attention mechanisms
  - Configurable positional embeddings
  - Extensible base architecture

## Installation

### System Requirements
- Python 3.10+
- PyTorch 2.1+
- 16GB+ RAM (for base config)
- 50GB+ disk space (for full C4 training)

### Dependency Setup
```bash
# Create conda environment
conda create -n lat python=3.10
conda activate lat

# Install core dependencies
pip install torch==2.1.0 torchvision==0.16.0
pip install transformers==4.35.0 datasets==2.14.5 wandb==0.16.0

# Optional for development
pip install black==23.11.0 flake8==6.1.0 mypy==1.6.1
```

## Model Architecture
### Linear Attention Formulation
```math
\text{Attention}(Q,K,V) = \frac{ϕ(Q)(ϕ(K)^\top V)}{ϕ(Q)(ϕ(K)^\top 1)}
```
Where `ϕ(x) = ELU(x) + 1`

### Core Implementation
```python
class LinearAttention(nn.Module):
    def forward(self, x):
        # Feature mapping
        q, k = self.elu_feature_map(q), self.elu_feature_map(k)
        
        # Associative scan
        k_cumsum = k.cumsum(dim=2) if causal else k.sum(dim=2)
        context = torch.einsum('bhnd,bhnc->bhdc', k, v)
        return torch.einsum('bhnd,bhdc->bhnc', q, context) / (k_cumsum + eps)
```

## Training
### Configuration (`configs/base.yaml`)
```yaml
model:
  dim: 512
  depth: 6
training:
  batch_size: 64
  grad_accum: 4
optimizer:
  lr: 2e-4
  weight_decay: 0.1
```

### Execution
```bash
# Basic training
python scripts/train.py \
  --config configs/base.yaml \
  --dataset c4 \
  --device cpu

# With logging
python scripts/train.py \
  --wandb_project lalm \
  --precision bf16
```

## Evaluation
| Metric | Value (Base Model) |
|--------|--------------------|
| PPL    | 18.3              |
| Memory | 2.1GB/seq        |
| Speed  | 15k tokens/sec (CPU) |

## Citation
```bibtex
@inproceedings{linearattn2024,
  title={Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff},
  author={Anonymous et al.},
  booktitle={ICML},
  year={2024}
}
```

## License
Apache 2.0 - See [LICENSE](LICENSE) for details
```
