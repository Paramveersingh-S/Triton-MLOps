import pytest
import torch
import torch.nn.functional as F
from triton_ops.bonus.swiglu import fused_swiglu

def test_fused_swiglu():
    M, N = 128, 512
    x = torch.randn(M, 2 * N, device='cuda', dtype=torch.float32)
    
    gate, up = x.split(N, dim=-1)
    ref = F.silu(gate) * up
    
    out = fused_swiglu(x)
    
    assert torch.allclose(ref, out, atol=1e-5), "SwiGLU output failed"
