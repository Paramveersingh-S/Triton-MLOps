import pytest
import torch
import torch.nn.functional as F
import math

from triton_ops.attention.prefix_prefill import prefix_prefill

def test_prefix_prefill_matches_standard_attention():
    B, H, S, P, D = 2, 4, 16, 64, 128
    
    q = torch.randn(B, H, S, D, device='cuda')
    prefix_k = torch.randn(P, H, D, device='cuda')
    prefix_v = torch.randn(P, H, D, device='cuda')
    new_k = torch.randn(B, H, S, D, device='cuda')
    new_v = torch.randn(B, H, S, D, device='cuda')
    
    out_triton = prefix_prefill(q, prefix_k, prefix_v, new_k, new_v)
    
    pk_exp = prefix_k.transpose(0, 1).unsqueeze(0).expand(B, H, P, D)
    pv_exp = prefix_v.transpose(0, 1).unsqueeze(0).expand(B, H, P, D)
    
    full_k = torch.cat([pk_exp, new_k], dim=2)
    full_v = torch.cat([pv_exp, new_v], dim=2)
    
    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(q, full_k.transpose(-2, -1)) * scale
    
    mask = torch.ones(S, P + S, device='cuda', dtype=torch.bool)
    causal = torch.tril(torch.ones(S, S, device='cuda', dtype=torch.bool))
    mask[:, P:] = causal
    
    scores.masked_fill_(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    
    attn = F.softmax(scores, dim=-1)
    ref = torch.matmul(attn, full_v)
    
    assert torch.allclose(ref, out_triton, atol=1e-4)
