import torch
import triton
import triton.language as tl

@triton.jit
def prefix_prefill_kernel(
    q_ptr,          # [B, H, S, D]
    prefix_k_ptr,   # [P, H, D]
    prefix_v_ptr,   # [P, H, D]
    new_k_ptr,      # [B, H, S, D]
    new_v_ptr,      # [B, H, S, D]
    o_ptr,          # [B, H, S, D]
    B, H, S, P, D,
    softmax_scale,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_pkp, stride_pkh, stride_pkd,
    stride_pvp, stride_pvh, stride_pvd,
    stride_nkb, stride_nkh, stride_nks, stride_nkd,
    stride_nvb, stride_nvh, stride_nvs, stride_nvd,
    stride_ob, stride_oh, stride_os, stride_od,
    BLOCK_D: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    s_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    b_idx = tl.program_id(2)

    q_base = q_ptr + b_idx * stride_qb + h_idx * stride_qh + s_idx * stride_qs
    nk_base = new_k_ptr + b_idx * stride_nkb + h_idx * stride_nkh
    nv_base = new_v_ptr + b_idx * stride_nvb + h_idx * stride_nvh
    pk_base = prefix_k_ptr + h_idx * stride_pkh
    pv_base = prefix_v_ptr + h_idx * stride_pvh
    o_base = o_ptr + b_idx * stride_ob + h_idx * stride_oh + s_idx * stride_os

    d_offsets = tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D
    q = tl.load(q_base + d_offsets * stride_qd, mask=mask_d, other=0.0)

    m_i = -float('inf')
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for p_start in range(0, P, BLOCK_P):
        p_offsets = p_start + tl.arange(0, BLOCK_P)
        mask_p = p_offsets < P
        
        k_ptr = pk_base + p_offsets[:, None] * stride_pkp + d_offsets[None, :] * stride_pkd
        v_ptr = pv_base + p_offsets[:, None] * stride_pvp + d_offsets[None, :] * stride_pvd
        
        mask_2d = mask_p[:, None] & mask_d[None, :]
        k = tl.load(k_ptr, mask=mask_2d, other=0.0)
        v = tl.load(v_ptr, mask=mask_2d, other=0.0)
        
        qk = tl.sum(q[None, :] * k, axis=1) * softmax_scale
        qk = tl.where(mask_p, qk, -float('inf'))
        
        m_ij = tl.maximum(m_i, tl.max(qk, axis=0))
        alpha = tl.exp(m_i - m_ij)
        beta = tl.exp(qk - m_ij)
        
        l_i = alpha * l_i + tl.sum(beta, axis=0)
        acc = alpha * acc + tl.sum(beta[:, None] * v, axis=0)
        m_i = m_ij

    num_new_keys = s_idx + 1
    
    for s_start in range(0, num_new_keys, BLOCK_P):
        s_offsets = s_start + tl.arange(0, BLOCK_P)
        mask_s = s_offsets < num_new_keys
        
        k_ptr = nk_base + s_offsets[:, None] * stride_nks + d_offsets[None, :] * stride_nkd
        v_ptr = nv_base + s_offsets[:, None] * stride_nvs + d_offsets[None, :] * stride_nvd
        mask_2d = mask_s[:, None] & mask_d[None, :]
        
        k = tl.load(k_ptr, mask=mask_2d, other=0.0)
        v = tl.load(v_ptr, mask=mask_2d, other=0.0)
        
        qk = tl.sum(q[None, :] * k, axis=1) * softmax_scale
        qk = tl.where(mask_s, qk, -float('inf'))
        
        m_ij = tl.maximum(m_i, tl.max(qk, axis=0))
        alpha = tl.exp(m_i - m_ij)
        beta = tl.exp(qk - m_ij)
        
        l_i = alpha * l_i + tl.sum(beta, axis=0)
        acc = alpha * acc + tl.sum(beta[:, None] * v, axis=0)
        m_i = m_ij

    out = acc / l_i
    tl.store(o_base + d_offsets * stride_od, out, mask=mask_d)

def prefix_prefill(
    q: torch.Tensor,
    prefix_k: torch.Tensor,
    prefix_v: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
) -> torch.Tensor:
    B, H, S, D = q.shape
    P = prefix_k.shape[0]
    
    o = torch.empty_like(q)
    softmax_scale = 1.0 / (D ** 0.5)
    
    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_P = 128
    
    grid = (S, H, B)
    
    prefix_prefill_kernel[grid](
        q, prefix_k, prefix_v, new_k, new_v, o,
        B, H, S, P, D, softmax_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        prefix_k.stride(0), prefix_k.stride(1), prefix_k.stride(2),
        prefix_v.stride(0), prefix_v.stride(1), prefix_v.stride(2),
        new_k.stride(0), new_k.stride(1), new_k.stride(2), new_k.stride(3),
        new_v.stride(0), new_v.stride(1), new_v.stride(2), new_v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        BLOCK_D=BLOCK_D, BLOCK_P=BLOCK_P
    )
    return o
