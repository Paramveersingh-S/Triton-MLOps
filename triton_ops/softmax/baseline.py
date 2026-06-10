import torch

def naive_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Naive softmax implementation representing the 3-pass PyTorch ATen behavior.
    Pass 1: max_x = max(x)                    -> 1 GMEM read
    Pass 2: exp_shifted = exp(x - max_x)      -> 1 GMEM read + 1 write
    Pass 3: softmax = exp_shifted / sum(...)  -> 1 GMEM read + 1 write
    """
    # Pass 1
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    
    # Pass 2
    x_shifted = x - x_max
    numerator = torch.exp(x_shifted)
    
    # Pass 3
    denominator = torch.sum(numerator, dim=dim, keepdim=True)
    y = numerator / denominator
    
    return y
