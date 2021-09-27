import numpy as np
import pandas as pd
from tensorflow import keras as k

yoga_count = 24

def get_yoga_data(fraction=1):
    data_file = "yoga.csv"

    labels_header = "Pose"

    yoga_data = pd.read_csv(data_file, names=[*[f"data {x}" for x in range(1, yoga_count + 1)], labels_header]).sample(frac=fraction).reset_index(drop=True)

    all_yoga_data_features = yoga_data.copy()
    all_yoga_data_features.pop(labels_header)
    normalize = k.layers.Normalization()
    normalize.adapt(all_yoga_data_features)

    training_split = int(len(yoga_data)*0.8)

    yoga_train = yoga_data[0:training_split]
    yoga_test = yoga_data[training_split:]

    yoga_train_labels = yoga_train.pop(labels_header)
    yoga_train_features = np.array(yoga_train)

    yoga_test_labels = yoga_test.pop(labels_header)
    yoga_test_features = np.array(yoga_test)

    return yoga_train_features, yoga_train_labels, yoga_test_features, yoga_test_labels, normalize