import os
import plotly.express as px
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
import sqlite3

def get_ARIMA(Data):
    Data['Month'] = Data.Month.astype(str).str.pad(2,fillchar='0')
    Data['Year'] = Data['Year'].astype(str)
    Data['Date'] = pd.to_datetime(Data['Year'] + Data['Month'], format='%Y%m')
    train = Data.loc[Data['Date'] < pd.to_datetime(2019, format='%Y'),]
    test = Data.loc[Data['Date'] >= pd.to_datetime(2019, format='%Y'),]

    del train['Falls within']
    del train['Year']
    del train ['Month']
    del test['Falls within']
    del test['Year']
    del test ['Month']

    train = train.set_index(['Date'])
    test = test.set_index(['Date'])
    predicted = test.copy()

    smodel = pm.auto_arima(train,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=12,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=True,   # No Seasonality
                      start_P=0,
                      start_Q=0,
                      D=1,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)

    n_periods = len(test)
    ARIMA_predictions, confint = smodel.predict(n_periods=n_periods, return_conf_int=True)
    predicted["Crime_Number"] = ARIMA_predictions
    return train, test, predicted


def get_RNN(Data):
    def create_inout_sequences(input_data, tw):
        inout_seq = []
        L = len(input_data)
        for i in range(L - tw):
            train_seq = input_data[i:i + tw]
            train_label = input_data[i + tw:i + tw + 1]
            inout_seq.append((train_seq, train_label))
        return inout_seq

    Data['Month'] = Data.Month.astype(str).str.pad(2,fillchar='0')
    Data['Year'] = Data['Year'].astype(str)
    Data['Date'] = pd.to_datetime(Data['Year'] + Data['Month'], format='%Y%m')
    train = Data.loc[Data['Date'] < pd.to_datetime(2019, format='%Y'),]
    test = Data.loc[Data['Date'] >= pd.to_datetime(2019, format='%Y'),]

    del train['Falls within']
    del train['Year']
    del train ['Month']
    del test['Falls within']
    del test['Year']
    del test ['Month']

    train = train.set_index(['Date'])
    test = test.set_index(['Date'])
    predicted = test.copy()
    train_data_normalized = train.to_numpy()

    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data_normalized = scaler.fit_transform(train_data_normalized.reshape(-1, 1))
    train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
    train_window = 12
    train_inout_seq = create_inout_sequences(train_data_normalized, train_window)

    class LSTM(nn.Module):
        def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
            super().__init__()
            self.hidden_layer_size = hidden_layer_size

            self.lstm = nn.LSTM(input_size, hidden_layer_size)

            self.linear = nn.Linear(hidden_layer_size, output_size)

            self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                                torch.zeros(1, 1, self.hidden_layer_size))

        def forward(self, input_seq):
            lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
            predictions = self.linear(lstm_out.view(len(input_seq), -1))
            return predictions[-1]

    model = LSTM()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(model)

    epochs = 200

    for i in range(epochs):
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))

            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()
        if i % 50 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
    fut_pred = len(test)
    test_inputs = train_data_normalized[-train_window:].tolist()

    model.eval()
    for i in range(fut_pred):
        seq = torch.FloatTensor(test_inputs[-train_window:])
        with torch.no_grad():
            model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
            test_inputs.append(model(seq).item())

    RNN_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:]).reshape(-1, 1))
    predicted["Crime_Number"] = RNN_predictions.flatten()
    return train, test, predicted

def get_Mape(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100
