import torch.nn as nn
import torch

class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
    
def l2_norm(x):
    # Normalize using L2 norm on the last dimension
    norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    return x / (norm + 1e-6)  # Add 1e-6 to avoid division by zero and improve numerical stability
