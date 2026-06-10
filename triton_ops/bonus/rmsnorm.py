import torch
import triton
import triton.language as tl

@triton.jit
def fused_rmsnorm_residual_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    M, N,
    stride_xm,
    stride_rm,
    stride_outm,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_N)
    mask = col_offsets < N

    x = tl.load(x_ptr + row_idx * stride_xm + col_offsets, mask=mask, other=0.0)
    res = tl.load(residual_ptr + row_idx * stride_rm + col_offsets, mask=mask, other=0.0)
    
    # In-place add
    x = x + res
    # write back residual
    tl.store(residual_ptr + row_idx * stride_rm + col_offsets, x, mask=mask)
    
    # RMSNorm
    variance = tl.sum(x * x, axis=0) / N
    rsqrt = tl.math.rsqrt(variance + eps)
    
    x_norm = x * rsqrt
    
    # Multiply by weight
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)
    out = x_norm * weight
    
    tl.store(out_ptr + row_idx * stride_outm + col_offsets, out, mask=mask)

def fused_rmsnorm_residual(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_2d = x.view(-1, x.shape[-1])
    res_2d = residual.view(-1, residual.shape[-1])
    M, N = x_2d.shape
    out_2d = torch.empty_like(x_2d)
    
    BLOCK_N = triton.next_power_of_2(N)
    
    grid = (M, )
    fused_rmsnorm_residual_kernel[grid](
        x_2d, res_2d, weight, out_2d,
        M, N,
        x_2d.stride(0), res_2d.stride(0), out_2d.stride(0),
        eps, BLOCK_N=BLOCK_N
    )
    
    return out_2d.view(x.shape)
