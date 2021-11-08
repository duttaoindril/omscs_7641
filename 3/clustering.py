import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from util import set_seed, start_time, mean, plt, plot
from data import yoga_labels_count, get_yoga_data, get_stock_data, stock_labels_count, OneHotEncoder


def kmeans(k, features):
    end = start_time('K Means')
    kmeans = KMeans(n_clusters=k).fit(features)
    predicted_labels = kmeans.predict(features)
    time = end()
    return predicted_labels, features


def em(k, features):
    end = start_time('EM')
    gmm = GaussianMixture(n_components=k).fit(features)
    predicted_labels = gmm.predict(features)
    time = end()
    return predicted_labels, features


def add_cluster_features(clusters, features):
    onehot_encoder = OneHotEncoder(sparse=False)
    cluster_encoded = clusters.reshape(len(clusters), 1)
    onehot_encoded = onehot_encoder.fit_transform(cluster_encoded)
    return np.append(features, onehot_encoded, axis=1)


clustering = {
    'kmeans': kmeans,
    'em': em,
}


def analyze_clusters(predicted, labels):
    mapped_labels = {}
    for i in range(len(predicted)):
        if(labels[i] not in mapped_labels):
            mapped_labels[labels[i]] = {}
        if(predicted[i] not in mapped_labels[labels[i]]):
            mapped_labels[labels[i]][predicted[i]] = 0
        mapped_labels[labels[i]][predicted[i]] += 1
    keys = mapped_labels.keys()
    accuracy = []
    for key in keys:
        counts = mapped_labels[key]
        max_key = max(counts, key=lambda key: counts[key])
        accuracy.append(counts[max_key] / sum(counts.values()))
    sorted_keys = list(keys)
    sorted_keys.sort()
    print("MAPPED LABELS: ", mapped_labels)
    print("AVERAGE ACCURACY: ", mean(accuracy))
    return sorted_keys, accuracy, mapped_labels


def run(name, features, labels, labels_count):
    labels = labels.values
    print(name+" CLUSTERING KMEANS")
    km_keys, km_accuracy, _ = analyze_clusters(
        kmeans(labels_count, features)[0],
        labels,
    )
    print(name+" CLUSTERING EXPECTATIONS MAXIMIZATION")
    em_keys, em_accuracy, _ = analyze_clusters(
        em(labels_count, features)[0],
        labels,
    )
    plt(km_keys, km_accuracy, label="K-Means")
    plt(em_keys, em_accuracy, label="Expectation Maximization")
    plot(
        name + ' K Means VS Expectation Maximization',
        xlabel="Label",
        ylabel="Accuracy",
    )


def yoga():
    features, labels, _, _ = get_yoga_data(test_split=0)
    run("Yoga", features, labels, yoga_labels_count)


def stock(norm=False):
    norm_string = "Normalized " if norm else ""
    features, labels, _, _ = get_stock_data(test_split=0, normalized=norm)
    run(norm_string + "Stock", features, labels, stock_labels_count)


def main():
    yoga()
    stock()
    stock(norm=True)


if __name__ == "__main__":
    set_seed()
    main()
