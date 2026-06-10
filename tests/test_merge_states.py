import pytest
import torch
from triton_ops.attention.merge_states import _merge_attn_states_triton

def test_merge_attn_states():
    B, H, P, D = 2, 4, 2, 128
    
    o1 = torch.randn(B, H, D, device='cuda')
    m1 = torch.randn(B, H, device='cuda')
    l1 = torch.rand(B, H, device='cuda') + 0.1
    
    o2 = torch.randn(B, H, D, device='cuda')
    m2 = torch.randn(B, H, device='cuda')
    l2 = torch.rand(B, H, device='cuda') + 0.1
    
    partial_o = torch.stack([o1, o2], dim=2)
    partial_m = torch.stack([m1, m2], dim=2)
    partial_l = torch.stack([l1, l2], dim=2)
    
    out = _merge_attn_states_triton(partial_o, partial_m, partial_l)
    
    # Compute ref
    m_global = torch.maximum(m1, m2)
    w1 = torch.exp(m1 - m_global)
    w2 = torch.exp(m2 - m_global)
    
    l_global = w1 * l1 + w2 * l2
    
    ref = (w1.unsqueeze(-1) * o1 + w2.unsqueeze(-1) * o2) / l_global.unsqueeze(-1)
    
    assert torch.allclose(ref, out, atol=1e-5)
