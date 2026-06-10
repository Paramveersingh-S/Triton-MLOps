import pytest
import torch
from triton_ops.bonus.rmsnorm import fused_rmsnorm_residual

def test_rmsnorm_residual():
    M, N = 128, 512
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    residual = torch.randn(M, N, device='cuda', dtype=torch.float32)
    weight = torch.randn(N, device='cuda', dtype=torch.float32)
    eps = 1e-6
    
    residual_ref = residual.clone()
    
    residual_ref = residual_ref + x
    variance = residual_ref.pow(2).mean(-1, keepdim=True)
    ref_out = residual_ref * torch.rsqrt(variance + eps) * weight
    
    out = fused_rmsnorm_residual(x, residual, weight, eps)
    
    assert torch.allclose(residual_ref, residual, atol=1e-5), "Residual in-place add failed"
    assert torch.allclose(ref_out, out, atol=1e-5), "RMSNorm output failed"
