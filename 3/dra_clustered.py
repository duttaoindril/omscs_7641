from dra import dra
from clustering import clustering, analyze_clusters
from util import set_seed, start_time, mean, plt, plot
from data import yoga_labels_count, get_yoga_data, get_stock_data, stock_labels_count, OneHotEncoder


def run(name, features, labels, label_count, component_count):
    for clustering_method in clustering.values():
        for dra_method in dra.values():
            print('=========================================================')
            print(name, clustering_method.__name__, dra_method.__name__)
            transformed_features, _, _, _, _, _ = dra_method(
                features,
                component_count,
            )
            clustered_labels, _ = clustering_method(
                label_count,
                transformed_features,
            )
            keys, accuracy, _ = analyze_clusters(clustered_labels, labels)
            plt(
                keys,
                accuracy,
                label=clustering_method.__name__ + ' with ' + dra_method.__name__,
            )
    plot(
        name + ' DRA with Clustering Accuracy Comparisons',
        xlabel="Label",
        ylabel="Accuracy",
    )


yoga_component_count = 12


def yoga():
    features, labels, _, _ = get_yoga_data(test_split=0)
    run('Yoga', features, labels, yoga_labels_count, yoga_component_count)


stock_component_count = 45
normalized_stock_component_count = 35


def stock(norm=False):
    norm_string = "Normalized " if norm else ""
    features, labels, _, _ = get_stock_data(
        test_split=0,
        normalized=norm,
    )
    run(
        norm_string + 'Stock',
        features,
        labels,
        stock_labels_count,
        normalized_stock_component_count if norm else stock_component_count,
    )


def main():
    yoga()
    stock()
    stock(norm=True)


if __name__ == "__main__":
    set_seed()
    main()
