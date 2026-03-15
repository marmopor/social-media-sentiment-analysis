import torch
import torch.nn as nn

class BetterDeepMLP(nn.Module):
    """
    MLP simplificado para evitar overfitting en datasets pequeños con muchas features.
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.5
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Dropout(0.2), # Input dropout para robustez
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def build_model(input_dim: int, num_classes: int = 3, hidden_dim: int = 64, dropout: float = 0.5):
    return BetterDeepMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout
    )
