import pytest
import torch

@pytest.fixture(autouse=True)
def setup_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Triton tests")
