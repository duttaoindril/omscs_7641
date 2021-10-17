import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

yoga_count = 24


def get_yoga_data(fraction=1, test_split=0.2):
    data_file = "yoga.csv"

    labels_header = "Pose"

    yoga_data = pd.read_csv(
        data_file,
        names=[
            *[f"data {x}" for x in range(1, yoga_count + 1)],
            labels_header,
        ],
    ).sample(frac=fraction).reset_index(drop=True)

    yoga_labels = yoga_data.pop(labels_header)
    yoga_features = np.array(yoga_data)

    yoga_train_features, yoga_test_features, yoga_train_labels, yoga_test_labels = train_test_split(
        yoga_features,
        yoga_labels,
        test_size=test_split
    )

    one_hot = OneHotEncoder()

    yoga_train_labels_hot = one_hot.fit_transform(
        yoga_train_labels.values.reshape(-1, 1)
    ).todense()
    yoga_test_labels_hot = one_hot.transform(
        yoga_test_labels.values.reshape(-1, 1)
    ).todense()

    return yoga_train_features, yoga_train_labels_hot, yoga_test_features, yoga_test_labels_hot
