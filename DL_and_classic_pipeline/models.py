import torch
import torch.nn as nn
from omegaconf import DictConfig


class ClassifyModel(nn.Module):
    """MLP-классификатор для бинарной классификации."""

    def __init__(self, in_features: int, hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.2):
        """
        Инициализирует MLP с переменным количеством скрытых слоёв.

        Args:
            in_features: количество входных признаков
            hidden_dim: ширина первого скрытого слоя
            num_layers: количество слоёв (включая выходной)
            dropout: вероятность dropout
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        layers = []
        current_dim = in_features


        for i in range(num_layers - 1):
            next_dim = hidden_dim // (2 ** i)
            layers.append(nn.Linear(current_dim, next_dim))
            layers.append(nn.BatchNorm1d(next_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            current_dim = next_dim


        layers.append(nn.Linear(current_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход. Принимает батч признаков, возвращает логиты."""
        return self.net(x)
