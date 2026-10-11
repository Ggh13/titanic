import torch
import torch.nn as nn
from omegaconf import DictConfig


class ClassifyModel(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 1024, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, 1)  
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
