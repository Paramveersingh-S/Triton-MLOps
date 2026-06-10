import pytest
import torch
import torch.nn.functional as F

from triton_ops.softmax.baseline import naive_softmax

@pytest.mark.parametrize("M,N", [
    (1, 128),
    (128, 128),
    (1024, 512),
    (4096, 1024),
    (128, 65537),  # non-power-of-2
])
def test_naive_softmax(M, N):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    ref = F.softmax(x, dim=-1)
    out = naive_softmax(x, dim=-1)
    assert torch.allclose(ref, out, atol=1e-5)
