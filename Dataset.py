import os
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

import torch.nn as nn

import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


class StockDataset(Dataset):
    def __init__(self, data_dir, lookback, scaler, use_deltas=False, max_files=None, recent_only=False,
                 recent_period=180):
        """
        :param data_dir: Directory containing stock data files.
        :param lookback: Lookback window size for the sequence data.
        :param scaler: Scaler for preprocessing stock prices.
        :param use_deltas: Boolean indicating if we should use price deltas instead of absolute prices.
        :param max_files: Limit the number of stock files to load (useful for testing).
        """
        self.data_dir = data_dir
        self.lookback = lookback
        self.scaler = scaler
        self.use_deltas = use_deltas
        self.recent_only = recent_only
        self.recent_period = recent_period
        self.stock_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]

        # If max_files is set, limit the number of files
        if max_files is not None:
            self.stock_files = self.stock_files[:max_files]

        self.data = []
        print(f"Processing files")

        for file in self.stock_files:
            stock_data = pd.read_csv(file, index_col=0)
            stock_data.sort_index(inplace=True)

            if len(stock_data) < 180:
                continue

            if self.use_deltas:
                stock_data['price_change'] = stock_data['adjclose'].diff()
                stock_data.dropna(inplace=True)
                if self.recent_only:
                    stock_data = stock_data[-self.recent_period:]
                scaled_data = scaler.fit_transform(stock_data[['price_change']].values.reshape(-1, 1))
            else:
                stock_data.dropna(inplace=True)
                scaled_data = scaler.fit_transform(stock_data[['adjclose']].values.reshape(-1, 1))

            for i in range(len(stock_data) - lookback):
                x = scaled_data[i:i + lookback]
                y = scaled_data[i + lookback]
                self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)