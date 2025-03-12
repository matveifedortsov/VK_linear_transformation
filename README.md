# Linear Transformer Research

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![CI Status](https://github.com/yourusername/linear-transformer-research/actions/workflows/tests.yml/badge.svg)](https://github.com/yourusername/linear-transformer-research/actions)

Official implementation of "Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff" (arXiv:2402.18668)

## Features

- 🚀 **Linear Complexity Attention** - O(n) memory/time implementation
- 📊 **Research-Friendly** - Full experiment tracking integration
- 🧩 **Modular Design** - Swap attention/FFN components easily

## Quick Start

```bash
# Install with PyPI
pip install linear-transformer

# Train base model
python -m scripts.train --config configs/base.yaml
