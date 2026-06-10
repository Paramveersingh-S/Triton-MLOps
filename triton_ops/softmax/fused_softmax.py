import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128}, num_warps=2),
        triton.Config({'BLOCK_N': 256}, num_warps=4),
        triton.Config({'BLOCK_N': 512}, num_warps=8),
        triton.Config({'BLOCK_N': 1024}, num_warps=16),
        triton.Config({'BLOCK_N': 2048}, num_warps=16),
        triton.Config({'BLOCK_N': 4096}, num_warps=32),
    ],
    key=['N'],
)
@triton.jit
def fused_softmax_kernel(
    x_ptr,
    y_ptr,
    M,
    N,
    stride_m,
    BLOCK_N: tl.constexpr,
):
    # Each program handles ONE row
    row_idx = tl.program_id(0)

    # Load row into registers (masked for non-power-of-2 N)
    col_offsets = tl.arange(0, BLOCK_N)
    mask = col_offsets < N
    x = tl.load(x_ptr + row_idx * stride_m + col_offsets,
                 mask=mask, other=-float('inf'))

    # Online softmax in registers
    x_max = tl.max(x, axis=0)
    x_shifted = x - x_max
    numerator = tl.exp(x_shifted)
    denominator = tl.sum(numerator, axis=0)
    y = numerator / denominator

    # Single write back
    tl.store(y_ptr + row_idx * stride_m + col_offsets, y, mask=mask)

def fused_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Fused softmax implementation using Triton.
    1 read + 1 write vs 3 reads + 2 writes in naive implementation.
    """
    # Make sure x is 2D
    x_2d = x.view(-1, x.shape[-1])
    M, N = x_2d.shape
    y_2d = torch.empty_like(x_2d)
    
    grid = (M, )
    
    fused_softmax_kernel[grid](
        x_2d, y_2d,
        M, N,
        x_2d.stride(0),
    )
    return y_2d.view(x.shape)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128}, num_warps=2),
        triton.Config({'BLOCK_N': 256}, num_warps=4),
        triton.Config({'BLOCK_N': 512}, num_warps=8),
        triton.Config({'BLOCK_N': 1024}, num_warps=16),
        triton.Config({'BLOCK_N': 2048}, num_warps=16),
        triton.Config({'BLOCK_N': 4096}, num_warps=32),
    ],
    key=['N'],
)
@triton.jit
def fused_scale_mask_softmax_kernel(
    x_ptr, mask_ptr, y_ptr,
    M, N, scale,
    stride_x, stride_m, stride_y,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_N)
    mask_cond = col_offsets < N
    
    # Load x and scale
    x = tl.load(x_ptr + row_idx * stride_x + col_offsets, mask=mask_cond, other=-float('inf'))
    x = x * scale
    
    # Load mask and add
    m_val = tl.load(mask_ptr + row_idx * stride_m + col_offsets, mask=mask_cond, other=-float('inf'))
    x = x + m_val
    
    x_max = tl.max(x, axis=0)
    x_shifted = x - x_max
    numerator = tl.exp(x_shifted)
    denominator = tl.sum(numerator, axis=0)
    y = numerator / denominator
    
    tl.store(y_ptr + row_idx * stride_y + col_offsets, y, mask=mask_cond)

def fused_scale_mask_softmax(x: torch.Tensor, mask: torch.Tensor, scale: float) -> torch.Tensor:
    x_2d = x.view(-1, x.shape[-1])
    mask_2d = mask.view(-1, mask.shape[-1])
    M, N = x_2d.shape
    y_2d = torch.empty_like(x_2d)
    
    grid = (M, )
    fused_scale_mask_softmax_kernel[grid](
        x_2d, mask_2d, y_2d,
        M, N, scale,
        x_2d.stride(0), mask_2d.stride(0) if mask_2d.size(0) > 1 else 0, y_2d.stride(0),
    )
    return y_2d.view(x.shape)
