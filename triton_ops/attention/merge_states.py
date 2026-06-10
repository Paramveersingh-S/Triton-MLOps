import torch
import triton
import triton.language as tl

@triton.jit
def merge_attn_states_kernel(
    partial_o_ptr,
    partial_m_ptr,
    partial_l_ptr,
    out_ptr,
    B, H, num_partitions, D,
    stride_ob, stride_oh, stride_op, stride_od,
    stride_mb, stride_mh, stride_mp,
    stride_lb, stride_lh, stride_lp,
    stride_outb, stride_outh, stride_outd,
    BLOCK_D: tl.constexpr,
):
    # Grid handles one (batch, head) pair
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    
    # Base pointers
    o_base = partial_o_ptr + b_idx * stride_ob + h_idx * stride_oh
    m_base = partial_m_ptr + b_idx * stride_mb + h_idx * stride_mh
    l_base = partial_l_ptr + b_idx * stride_lb + h_idx * stride_lh
    out_base = out_ptr + b_idx * stride_outb + h_idx * stride_outh
    
    m_global = -float('inf')
    
    for p in range(num_partitions):
        m_i = tl.load(m_base + p * stride_mp)
        m_global = tl.maximum(m_global, m_i)
        
    l_global = 0.0
    
    d_offsets = tl.arange(0, BLOCK_D)
    mask = d_offsets < D
    out_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    
    for p in range(num_partitions):
        m_i = tl.load(m_base + p * stride_mp)
        l_i = tl.load(l_base + p * stride_lp)
        o_i = tl.load(o_base + p * stride_op + d_offsets * stride_od, mask=mask, other=0.0)
        
        weight = tl.exp(m_i - m_global)
        l_global += weight * l_i
        out_acc += weight * o_i
        
    out_val = out_acc / l_global
    
    tl.store(out_base + d_offsets * stride_outd, out_val, mask=mask)

def _merge_attn_states_triton(
    partial_o: torch.Tensor, # [B, H, num_partitions, D]
    partial_m: torch.Tensor, # [B, H, num_partitions]
    partial_l: torch.Tensor, # [B, H, num_partitions]
) -> torch.Tensor:
    B, H, num_partitions, D = partial_o.shape
    out = torch.empty((B, H, D), dtype=partial_o.dtype, device=partial_o.device)
    
    grid = (B, H)
    BLOCK_D = triton.next_power_of_2(D)
    
    merge_attn_states_kernel[grid](
        partial_o, partial_m, partial_l, out,
        B, H, num_partitions, D,
        partial_o.stride(0), partial_o.stride(1), partial_o.stride(2), partial_o.stride(3),
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
        partial_l.stride(0), partial_l.stride(1), partial_l.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_D=BLOCK_D
    )
    
    return out
