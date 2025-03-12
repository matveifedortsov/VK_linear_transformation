# Linear Attention Transformer (LAT) - Research Implementation

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-red.svg)](https://pytorch.org)
[![arXiv](https://img.shields.io/badge/arXiv-2402.18668-b31b1b.svg)](https://arxiv.org/abs/2402.18668)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org)

Official PyTorch implementation of the linear-complexity transformer architecture from *"Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff"*, optimized for CPU/GPU training and academic research.

## Table of Contents
- [Key Features](#key-features)
- [Installation](#installation)
- [Project Structure](#project-structure)
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

Model Architecture
Core Formulation
The linear attention mechanism implements the associative reformulation:
\text{Attention}(Q,K,V) = \frac{ϕ(Q)(ϕ(K)^\top V)}{ϕ(Q)(ϕ(K)^\top 1)}

