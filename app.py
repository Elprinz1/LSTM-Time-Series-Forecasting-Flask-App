from copy import deepcopy as dc
from model import LSTM
from utils import preprocess_data
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template
import pickle
import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend for Mac OS

# set plot style
plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'lightblue',
    'legend.loc': 'upper left'
})


app = Flask(__name__)

# Load the model and the scaler

model = torch.load('best_model.pth')
scaler = torch.load('scaler.pth')

# Define the lookback
lookback = 14


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Load the request file and preprocess the data
    if request.method == 'POST':
        data = request.files['data']
        if data.filename == '':
            return 'No selected file'
        if data.filename.split('.')[-1] not in ['csv', 'xlsx']:
            return 'Invalid file format'

        if data.filename.split('.')[-1] == 'csv':
            msg = "File uploaded successfully"
            data = pd.read_csv(data, index_col=0)
        else:
            msg = "File uploaded successfully"
            data = pd.read_excel(data, index_col=0)

    X = preprocess_data(data, scaler, lookback)
    model.eval()
    with torch.no_grad():
        y_pred = model(X.clone().detach())
        y_pred = y_pred.cpu().numpy()

    y_pred = y_pred.flatten()

    dummies = np.zeros((X.shape[0], lookback+1))
    dummies[:, 0] = y_pred
    dummies = scaler.inverse_transform(dummies)

    y_pred = dc(dummies[:, 0])

    date = data.index[-lookback:]

    # Plot the prediction and actual close price using the actual date
    plt.figure(figsize=(12, 6))
    plt.title('Amazon stock price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.plot(date, y_pred, label='Prediction')
    plt.plot(date, data['Close'][-lookback:], label='Actual')
    plt.text(date[-1], y_pred[-1], f'{y_pred[-1]:.2f}',
             fontsize=8, va='center', ha='left', color='navy')

    plt.xticks(rotation=45)
    plt.legend()

    plot_path = 'static/plot.png'
    plt.savefig(plot_path)
    plt.close()

    return render_template('index.html', prediction=y_pred[-1], plot_url=plot_path, msg=msg)


if __name__ == '__main__':
    app.run(debug=True)
