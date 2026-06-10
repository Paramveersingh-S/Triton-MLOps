import torch
import time
import math
import torch.nn.functional as F
from triton_ops.attention.prefix_prefill import prefix_prefill

def measure_standard_attention(B, H, total_seq, D):
    q = torch.randn(B, H, total_seq, D, device='cuda')
    k = torch.randn(B, H, total_seq, D, device='cuda')
    v = torch.randn(B, H, total_seq, D, device='cuda')
    
    # warmup
    for _ in range(5):
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D), dim=-1)
        out = torch.matmul(attn, v)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(20):
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D), dim=-1)
        out = torch.matmul(attn, v)
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / 20 * 1000  # ms

def measure_prefix_prefill(B, H, S, P, D):
    q = torch.randn(B, H, S, D, device='cuda')
    prefix_k = torch.randn(P, H, D, device='cuda')
    prefix_v = torch.randn(P, H, D, device='cuda')
    new_k = torch.randn(B, H, S, D, device='cuda')
    new_v = torch.randn(B, H, S, D, device='cuda')
    
    # warmup
    for _ in range(5):
        prefix_prefill(q, prefix_k, prefix_v, new_k, new_v)
        
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(20):
        prefix_prefill(q, prefix_k, prefix_v, new_k, new_v)
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / 20 * 1000  # ms

if __name__ == "__main__":
    scenarios = [
        # (prefix_len, new_tokens, heads, head_dim, batch_size)
        (1024,  128, 32, 128, 1),
        (2048,  256, 32, 128, 1),
        (4096,  64,  32, 128, 1), # Changed B=4 to B=1 to avoid OOM
        (6144,  32,  16, 128, 1), # Reduced P and H to avoid OOM
    ]

    print("Prefix P | New S | TTFT Standard | TTFT Prefix | Reduction")
    print("---------|-------|---------------|-------------|----------")
    for P, S, H, D, B in scenarios:
        # Clear cache before each standard benchmark to help with OOM
        torch.cuda.empty_cache()
        t_standard = measure_standard_attention(B, H, P+S, D)
        
        torch.cuda.empty_cache()
        t_prefix = measure_prefix_prefill(B, H, S, P, D)
        
        ttft_reduction = (t_standard - t_prefix) / t_standard * 100
        print(f"{P:<8} | {S:<5} | {t_standard:<11.2f}ms | {t_prefix:<9.2f}ms | {ttft_reduction:.1f}%")
