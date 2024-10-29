import os
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

import torch.nn as nn

import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

# Transformer Model (for long-term analysis)
class TransformerStockModel(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, num_encoder_layers, window_size=45, dropout=0.1):
        super(TransformerStockModel, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = self._generate_positional_encoding(d_model, window_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, 1)  # Output score for long-term performance

    def forward(self, x):
        """
        Forward pass for the Transformer model.

        :param x: Input time-series data (batch_size, sequence_length, input_dim).
        """
        # Project input into the d_model space
        x = self.input_projection(x)  # (batch_size, sequence_length, d_model)

        # Add positional encoding
        x = x + self.positional_encoding[:x.size(1), :].unsqueeze(0).to(
            x.device)  # (batch_size, sequence_length, d_model)

        # Pass through the transformer encoder
        x = self.transformer_encoder(x)  # (batch_size, sequence_length, d_model)

        # Use only the last time step for the final prediction
        return self.fc(x[:, -1, :])  # (batch_size, 1)

    def _generate_positional_encoding(self, d_model, window_size, max_len=5000):
        """
        Generates positional encoding to add to the input embeddings.

        :param d_model: Dimension of the model's embeddings.
        :param window_size: Number of days in the sliding window.
        :param max_len: Maximum length of positional encodings.
        """
        positional_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        # Return the positional encoding for the given window size
        return positional_encoding[:window_size, :]


# LSTM Model (for short-term analysis)
class LSTMForecastingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super(LSTMForecastingModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()
        self.criterion = nn.SmoothL1Loss()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc1(out[:, -1, :])  # Use the last time step
        out = self.relu(out)
        out = self.fc2(out)
        return out