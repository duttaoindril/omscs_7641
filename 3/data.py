import numpy as np
import pandas as pd
from gen_stock_data import days
from tensorflow import keras as k
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

yoga_labels_count = 4
yoga_count = 24


def get_yoga_data(fraction=1, test_split=0.25, hot=False, normalize=False):
    data_file = "yoga.csv"
    labels_header = "Pose"
    return get_csv_data(data_file, labels_header, yoga_count, shuffle=True, fraction=fraction, test_split=test_split, hot=hot, normalize=normalize)


stock_labels_count = 3
stock_count = days


def get_stock_data(fraction=1, test_split=0.25, hot=False, normalize=False, normalized=False):
    data_file = "stock_normalized.csv" if normalized else "stock.csv"
    labels_header = "BuySellHoldSignal"
    return get_csv_data(data_file, labels_header, stock_count, shuffle=False, fraction=fraction, test_split=test_split, hot=hot, normalize=normalize)


def get_csv_data(data_file, labels_header, feature_count, shuffle, fraction=1, test_split=0.25, hot=False, normalize=False):
    data = pd.read_csv(
        data_file,
        names=[
            *[f"data {x}" for x in range(1, feature_count + 1)],
            labels_header,
        ],
    )

    data = data if not shuffle else data.sample(
        frac=fraction,
    ).reset_index(drop=True)

    data_labels = data.pop(labels_header)
    data_features = np.array(data)

    if normalize:
        normalize = k.layers.Normalization()
        normalize.adapt(data_features)
        data_features = normalize(data_features)

    if test_split == 0:
        data_train_features, data_test_features, data_train_labels, data_test_labels = data_features, None, data_labels, None
    else:
        data_train_features, data_test_features, data_train_labels, data_test_labels = train_test_split(
            data_features,
            data_labels,
            test_size=test_split,
            shuffle=shuffle,
        )

    if hot:
        one_hot = OneHotEncoder()
        data_train_labels_hot = one_hot.fit_transform(
            data_train_labels.values.reshape(-1, 1)
        ).todense()
        if(data_test_labels is not None):
            data_test_labels_hot = one_hot.transform(
                data_test_labels.values.reshape(-1, 1)
            ).todense()
        return data_train_features, data_train_labels, data_train_labels_hot, data_test_features, data_test_labels, data_test_labels_hot

    return data_train_features, data_train_labels, data_test_features, data_test_labels
