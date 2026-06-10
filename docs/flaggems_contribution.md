# FlagGems Contribution Guide

FlagGems is a backend-neutral suite of Triton kernels designed to replace the PyTorch ATen backend for improved performance and hackability.

## Operators We Are Contributing
- `aten::softmax` -> Replaced by our fused softmax
- `aten::_scaled_dot_product_attention` -> Replaced by our prefix-prefill attention

## Registration Process

To register an operator in FlagGems:

```python
# In flaggems operator registry
@libentry()
@triton.jit
def softmax_kernel(...): ...

def softmax(x, dim=-1):
    return softmax_kernel[grid](x, ...)

# Register as ATen replacement
torch.library.impl("aten::softmax", "CUDA")(softmax)
```
