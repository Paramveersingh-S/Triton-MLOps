import pytest
import torch
import torch.nn.functional as F

from triton_ops.softmax.baseline import naive_softmax
from triton_ops.softmax.fused_softmax import fused_softmax

@pytest.mark.parametrize("M,N", [
    (1, 128),
    (128, 128),
    (1024, 512),
    (4096, 1024),
    (128, 65537),  # non-power-of-2
])
def test_softmax_implementations(M, N):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    ref = F.softmax(x, dim=-1)
    
    out_naive = naive_softmax(x, dim=-1)
    assert torch.allclose(ref, out_naive, atol=1e-5), "Naive softmax failed"
    
    out_fused = fused_softmax(x)
    assert torch.allclose(ref, out_fused, atol=1e-5), "Fused softmax failed"

@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fused_softmax_dtypes(dtype):
    M, N = 128, 128
    x = torch.randn(M, N, device='cuda', dtype=dtype)
    # Using float32 for reference
    ref = F.softmax(x.to(torch.float32), dim=-1).to(dtype)
    out_fused = fused_softmax(x)
    assert torch.allclose(ref, out_fused, atol=1e-3 if dtype == torch.float16 else 1e-2), f"Fused softmax failed for {dtype}"

def test_fused_scale_mask_softmax():
    M, N = 128, 128
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    mask = torch.randn(M, N, device='cuda', dtype=torch.float32)
    scale = 0.5
    
    ref = F.softmax(x * scale + mask, dim=-1)
    # Import locally to avoid modifying the top level import for now
    from triton_ops.softmax.fused_softmax import fused_scale_mask_softmax
    assert torch.allclose(ref, out, atol=1e-5), "Fused scale+mask+softmax failed"

def test_numerical_stability():
    M, N = 128, 128
    # Test with x containing large values (> 80 in magnitude)
    x = torch.randn(M, N, device='cuda', dtype=torch.float32) * 100.0
    
    ref = F.softmax(x, dim=-1)
    from triton_ops.softmax.fused_softmax import fused_softmax
    out = fused_softmax(x)
    
    assert torch.allclose(ref, out, atol=1e-5), "Numerical stability test failed for large values"

def test_fused_softmax_backward():
    M, N = 128, 128
    x = torch.randn(M, N, device='cuda', dtype=torch.float32, requires_grad=True)
    
    # PyTorch reference backward
    y_ref = F.softmax(x, dim=-1)
    dy = torch.randn_like(y_ref)
    y_ref.backward(dy)
    dx_ref = x.grad.clone()
    
    # Triton backward
    from triton_ops.softmax.fused_softmax import fused_softmax_backward, fused_softmax
    with torch.no_grad():
        y_triton = fused_softmax(x)
        dx_triton = fused_softmax_backward(y_triton, dy)
        
    assert torch.allclose(dx_ref, dx_triton, atol=1e-5), "Fused softmax backward failed"
