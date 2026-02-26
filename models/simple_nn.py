"""
simple_nn.py
Definición del modelo de red neuronal simple para clasificación de sentimientos.
"""

import torch
import torch.nn as nn


class SimpleSentimentNN(nn.Module):
    """
    Red neuronal simple (MLP) para clasificación de sentimientos.
    
    Arquitectura mínima:
      - Capa lineal de entrada (input_dim -> hidden_dim)
      - Activación ReLU
      - Capa lineal de salida (hidden_dim -> num_classes)
    
    Parámetros totales = input_dim * hidden_dim + hidden_dim + hidden_dim * num_classes + num_classes
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 3):
        """
        Args:
            input_dim   : Dimensión de entrada (num features TF-IDF + numéricas).
            hidden_dim  : Tamaño de la capa oculta.
            num_classes : Número de clases de salida (3).
        """
        super(SimpleSentimentNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def count_parameters(self) -> int:
        """Devuelve el número total de parámetros entrenables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(input_dim: int, hidden_dim: int = 64, num_classes: int = 3) -> SimpleSentimentNN:
    """Función de construcción del modelo."""
    model = SimpleSentimentNN(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
    return model
