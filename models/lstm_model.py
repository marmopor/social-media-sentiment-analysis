"""
lstm_model.py
LSTM bidireccional para clasificación de sentimientos en texto.
Captura dependencias secuenciales en ambas direcciones del texto.
"""

import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    """
    Clasificador basado en LSTM bidireccional para análisis de sentimientos.

    Arquitectura:
      - Embedding(vocab_size, embed_dim)
      - LSTM bidireccional (2 capas) con dropout entre capas
      - Se toma el último hidden state de ambas direcciones
      - Dropout -> Linear(hidden_dim*2 -> 64) -> ReLU -> Dropout -> Linear(64 -> num_classes)

    La bidireccionalidad permite capturar contexto tanto pasado como futuro
    de cada palabra en la secuencia.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.5,
        pad_idx: int = 0,
    ):
        """
        Args:
            vocab_size  : Tamaño del vocabulario.
            embed_dim   : Dimensión de los embeddings.
            hidden_dim  : Dimensión del estado oculto del LSTM.
            num_layers  : Número de capas LSTM apiladas.
            num_classes : Número de clases de salida.
            dropout     : Probabilidad de dropout.
            pad_idx     : Índice del token de padding.
        """
        super(BiLSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # hidden_dim * 2 porque es bidireccional
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Tensor de índices de palabras, shape (batch_size, seq_len)
        Returns:
            logits : shape (batch_size, num_classes)
        """
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)

        # LSTM output
        lstm_out, (hidden, _) = self.lstm(embedded)
        # hidden: (num_layers * 2, batch, hidden_dim)

        # Tomar los últimos hidden states de ambas direcciones (última capa)
        # Forward: hidden[-2], Backward: hidden[-1]
        hidden_fwd = hidden[-2]  # (batch, hidden_dim)
        hidden_bwd = hidden[-1]  # (batch, hidden_dim)
        hidden_cat = torch.cat([hidden_fwd, hidden_bwd], dim=1)  # (batch, hidden_dim*2)

        # Capas fully connected
        out = self.dropout(hidden_cat)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

    def count_parameters(self) -> int:
        """Devuelve el número total de parámetros entrenables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(
    vocab_size: int,
    embed_dim: int = 128,
    hidden_dim: int = 128,
    num_layers: int = 2,
    num_classes: int = 3,
    dropout: float = 0.5,
    pad_idx: int = 0,
) -> BiLSTMClassifier:
    """Función de construcción del modelo BiLSTM."""
    return BiLSTMClassifier(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        pad_idx=pad_idx,
    )
