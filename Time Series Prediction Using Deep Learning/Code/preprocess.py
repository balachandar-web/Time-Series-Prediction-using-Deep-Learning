import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import dateutil.parser

def load_data(filepath):
    data = pd.read_csv('Dataset/retail_sales.csv')
    return data

def preprocess_data(data):
    data['Date'] = pd.to_datetime(data['Date'])

    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month

    data.drop('Date', axis=1, inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(predictions)

    return scaled_data, scaler

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

if __name__ == "__main__":
    data = load_data('Dataset/retail_sales.csv')
    scaled_data, scaler = preprocess_data(data)

    look_back = 12
    X, Y = create_dataset(scaled_data, look_back)

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    train_size = int(len(X) * 0.8)
    test_size = len(X) - train_size
    X_train, X_test = X[0:train_size], X[train_size:len(X)]
    Y_train, Y_test = Y[0:train_size], Y[train_size:len(Y)]

    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('Y_train.npy', Y_train)
    np.save('Y_test.npy', Y_test)
    np.save('scaler.npy', scaler)

