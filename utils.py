import pandas as pd
import numpy as np
from copy import deepcopy as dc
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def preprocess_data(data, scaler, lookback):
    data = dc(data)
    for i in range(1, lookback+1):
        data[f'Close_t-{i}'] = data['Close'].shift(i)

    data.dropna(inplace=True)

    data_np = data.to_numpy()
    data_scaled = scaler.transform(data_np)

    X = data_scaled[:, 1:]
    y = data_scaled[:, 0]

    X = dc(np.flip(X, axis=1))
    # 1 is the number of features or input_size or columns
    X = X.reshape(-1, lookback, 1)

    X = torch.tensor(X).float().to(device)

    return X
