import torch
import triton
import triton.language as tl

@triton.jit
def fused_silu_and_mul_kernel(
    x_ptr,
    out_ptr,
    M, N,
    stride_xm,
    stride_outm,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_N)
    mask = col_offsets < N

    # Load gate and up projection
    # gate is x[:, :N], up is x[:, N:]
    gate = tl.load(x_ptr + row_idx * stride_xm + col_offsets, mask=mask, other=0.0)
    up = tl.load(x_ptr + row_idx * stride_xm + N + col_offsets, mask=mask, other=0.0)
    
    # SwiGLU: silu(gate) * up
    # silu(x) = x * sigmoid(x)
    sigmoid_gate = 1.0 / (1.0 + tl.exp(-gate))
    silu_gate = gate * sigmoid_gate
    
    out = silu_gate * up
    
    tl.store(out_ptr + row_idx * stride_outm + col_offsets, out, mask=mask)

def fused_swiglu(x: torch.Tensor) -> torch.Tensor:
    """
    Computes SwiGLU on a tensor where the last dimension is 2 * N.
    """
    x_2d = x.view(-1, x.shape[-1])
    M, N2 = x_2d.shape
    N = N2 // 2
    
    out_2d = torch.empty((M, N), dtype=x.dtype, device=x.device)
    BLOCK_N = triton.next_power_of_2(N)
    
    grid = (M, )
    fused_silu_and_mul_kernel[grid](
        x_2d, out_2d,
        M, N,
        x_2d.stride(0), out_2d.stride(0),
        BLOCK_N=BLOCK_N
    )
    
    out_shape = list(x.shape)
    out_shape[-1] = N
    return out_2d.view(*out_shape)
