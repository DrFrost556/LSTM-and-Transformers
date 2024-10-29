import os
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import torch.nn as nn

import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

from Train import train_model
from Dataset import StockDataset
from Model import TransformerStockModel, LSTMForecastingModel

# Parameters for LSTM (short-term analysis)
lstm_lookback = 30  # 30-day sliding window
batch_size = 64
scaler = MinMaxScaler(feature_range=(-1, 1))
data_dir = r'C:\Users\Oscar\PycharmProjects\pythonProject\stock_data'
max_files = 10  # Only load 5 CSV files for testing

# Parameters for Transformer (long-term analysis)
transformer_lookback = 90  # 90-day sliding window (longer lookback for long-term trends)

# LSTM will focus on short-term windows of recent data (e.g., last 6 months)
recent_only = True
recent_period = 180  # Use the last 180 days of data for LSTM

dataset_lstm = StockDataset(data_dir=data_dir, lookback=lstm_lookback, scaler=scaler,
                            use_deltas=True, max_files=max_files, recent_only=recent_only, recent_period=recent_period)
dataset_transformer = StockDataset(data_dir=data_dir, lookback=transformer_lookback, scaler=scaler,
                                   use_deltas=False, max_files=max_files)

dataloader_lstm = DataLoader(dataset_lstm, batch_size=batch_size, shuffle=True)
dataloader_transformer = DataLoader(dataset_transformer, batch_size=batch_size, shuffle=True)

# Initialize models
input_dim = 1
hidden_dim = 64
output_dim = 1
d_model = 64  # Dimension of the Transformer model's embeddings
n_heads = 4   # Number of attention heads
num_encoder_layers = 3  # Number of encoder layers

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize LSTM model
lstm_model = LSTMForecastingModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
lstm_model.to(device)

# Initialize Transformer model with a 90-day window
transformer_model = TransformerStockModel(input_dim=input_dim, d_model=d_model, n_heads=n_heads,
                                          num_encoder_layers=num_encoder_layers, window_size=transformer_lookback)
transformer_model.to(device)

# Define loss function and optimizers
criterion = torch.nn.MSELoss() # You can replace this with MSELoss or any other
lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
transformer_optimizer = torch.optim.Adam(transformer_model.parameters(), lr=0.0001)

# Train the LSTM model
print("Training LSTM model...")
train_model(lstm_model, dataloader_lstm, lstm_optimizer, criterion, num_epochs=10, device=device)

# Train the Transformer model
print("Training Transformer model...")
train_model(transformer_model, dataloader_transformer, transformer_optimizer, criterion, num_epochs=5, device=device)

# Save the models
print("Saving the trained models...")
torch.save(lstm_model.state_dict(), 'LSTM_final.pt')
torch.save(transformer_model.state_dict(), 'Transformer_final.pt')
print("Models saved.")