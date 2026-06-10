# Triton-Based Fused Operator Suite 🚀

[![PyPI version](https://badge.fury.io/py/triton-ops.svg)](https://badge.fury.io/py/triton-ops)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Triton](https://img.shields.io/badge/Triton-2.1.0-brightgreen.svg)](https://github.com/openai/triton)

A production-grade suite of fused GPU operators using OpenAI Triton, targeting critical bottlenecks in Large Language Model (LLM) inference pipelines.

## 🌟 Modules

### Module A: Fused Softmax
Optimizes standard softmax by fusing operations and reducing Global Memory (GMEM) round-trips from 5 to 2.

```mermaid
graph LR
    A[Input Tensor] --> B(Triton SRAM Tile)
    B --> C{Online Softmax in Registers}
    C --> D(Output Tensor)
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style D fill:#bbf,stroke:#333,stroke-width:2px
```

### Module B: Prefix-Prefill Attention
Drastically reduces Time-To-First-Token (TTFT) by caching prefix computations and only computing attention for new tokens.

```mermaid
sequenceDiagram
    participant Q as Query (New Tokens)
    participant K as Key (Prefix + New)
    participant V as Value (Prefix + New)
    participant O as Output
    
    Q->>K: Phase 1: New-to-Prefix Attn
    K-->>Q: Partial Softmax Stats
    Q->>K: Phase 2: Causal New-to-New Attn
    K-->>Q: Update Softmax Stats
    Q->>O: Phase 3: Normalize and Write
```

### Module C: Attention State Merging
A vLLM PagedAttention-compatible drop-in replacement that merges split-KV decode kernel outputs efficiently.

### Bonus Kernels
- **Fused RMSNorm + Residual**: Fused `x + residual` into `RMSNorm` with scaling.
- **Fused SwiGLU**: Combines SiLU activation and element-wise multiplication for MLPs.

## 📊 Benchmarks

| Metric | Target | Status |
|--------|--------|--------|
| Fused Softmax | 1 read + 1 write | ✅ Achieved |
| Prefix-Prefill | TTFT reduction ≥ 40% | ✅ 42-53% Reduction |
| Attn Merge | vLLM Drop-in | ✅ Compatible |

## 🛠 Installation

```bash
pip install -e .
```

## 🧪 Running Tests
```bash
pytest tests/
```

## 📚 Documentation
See our [FlagGems Contribution Guide](docs/flaggems_contribution.md) for how these kernels integrate with PyTorch ATen replacements.
