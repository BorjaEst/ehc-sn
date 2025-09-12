import torch
from torch import Tensor


def seeds(n: int, seed: int, device: str) -> list[int]:
    gen = torch.Generator(device)
    gen.manual_seed(seed)
    return torch.randint(0, 2**32 - 1, (n,), generator=gen, device=device).tolist()
