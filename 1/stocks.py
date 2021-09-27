import numpy as np
import pandas as pd
from convert import days
from tensorflow import keras as k

stock_days = days

def get_stock_data(fraction=1):
    data_file = "stocks.csv"

    labels_header = "BuySellSignal"

    stock_data = pd.read_csv(data_file, names=[*[f"{x} day" for x in range(1, stock_days+1)],labels_header])

    stock_data = stock_data.tail(int(len(stock_data)*fraction))

    all_stock_data_features = stock_data.copy()
    all_stock_data_features.pop(labels_header)
    normalize = k.layers.Normalization()
    normalize.adapt(all_stock_data_features)

    training_split = int(len(stock_data)*0.8)

    stock_train = stock_data[0:training_split]
    stock_test = stock_data[training_split:]

    stock_train_labels = stock_train.pop(labels_header)
    stock_train_features = np.array(stock_train)

    stock_test_labels = stock_test.pop(labels_header)
    stock_test_features = np.array(stock_test)

    return stock_train_features, stock_train_labels, stock_test_features, stock_test_labels, normalize