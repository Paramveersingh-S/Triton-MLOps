import torch
from triton_ops.attention.merge_states import _merge_attn_states_triton

def merge_attn_states(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
) -> None:
    num_tokens, num_heads, head_size = output.shape
    
    partial_o = torch.stack([prefix_output, suffix_output], dim=2)
    # LSE shape: [num_heads, num_tokens] -> transpose to [num_tokens, num_heads]
    prefix_lse_t = prefix_lse.transpose(0, 1)
    suffix_lse_t = suffix_lse.transpose(0, 1)
    partial_lse = torch.stack([prefix_lse_t, suffix_lse_t], dim=2)
    
    partial_m = partial_lse
    partial_l = torch.ones_like(partial_lse)
    
    merged = _merge_attn_states_triton(partial_o, partial_m, partial_l)
    
    output.copy_(merged)
