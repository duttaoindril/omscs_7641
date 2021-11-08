from dra import dra
from clustering import clustering, analyze_clusters
import numpy as np
import mlrose_hiive as ml
from util import plt, plot, set_seed, start_time
from sklearn.metrics import accuracy_score
from data import yoga_labels_count, get_yoga_data, get_stock_data, stock_labels_count
from clustering import kmeans, em, add_cluster_features

iteration_count = 5000


def neural_network(node_count, features, labels, test_features, test_labels,):
    model = ml.NeuralNetwork(
        hidden_nodes=[node_count*2, node_count*2],
        bias=True,
        curve=True,
        is_classifier=True,
        early_stopping=True,
        algorithm='gradient_descent',
        activation='sigmoid',
        learning_rate=0.001,
        max_attempts=500,
        max_iters=iteration_count,
    )
    print("Training...")
    end_time = start_time('NN TRAINING TIME')
    data = model.fit(features, labels)
    print("Results:")
    time = end_time()
    loss = data.loss
    train_accuracy = accuracy_score(labels, model.predict(features))
    test_accuracy = accuracy_score(test_labels, model.predict(test_features))
    print('loss:', data.loss)
    print('train_accuracy:', train_accuracy)
    print('test_accuracy:', test_accuracy)
    print('loss_curve:', len(data.fitness_curve))
    return time, loss, train_accuracy, test_accuracy, data.fitness_curve


def run(name, features, hot_labels, test_features, hot_test_labels, label_count, component_count):
    print('================================================================')
    print(name + " NN VANILLA")
    time, loss, train_accuracy, test_accuracy, loss_curve = neural_network(
        label_count, features, hot_labels, test_features, hot_test_labels,
    )
    loss_curve = loss_curve[10:]
    print(
        "time, loss, train_accuracy, test_accuracy:",
        time, loss, train_accuracy, test_accuracy,
    )
    plt(list(range(1, len(loss_curve)+1)), loss_curve, label='vanilla')
    for clustering_method in clustering.values():
        print('================================================================')
        print(name + " NN Clustering " + clustering_method.__name__)
        features_with_cluster = add_cluster_features(
            *clustering_method(label_count, features),
        )
        test_features_with_cluster = add_cluster_features(
            *clustering_method(label_count, test_features),
        )
        time, loss, train_accuracy, test_accuracy, loss_curve = neural_network(
            label_count,
            features_with_cluster,
            hot_labels,
            test_features_with_cluster,
            hot_test_labels,
        )
        print(
            "time, loss, train_accuracy, test_accuracy:",
            time, loss, train_accuracy, test_accuracy,
        )
        loss_curve = loss_curve[10:]
        plt(list(range(1, len(loss_curve)+1)),
            loss_curve, label=clustering_method.__name__)
    for dra_method in dra.values():
        print('================================================================')
        print(name + " NN DRA " + dra_method.__name__)
        features_with_dra = dra_method(features, component_count)[0]
        test_features_with_dra = dra_method(test_features, component_count)[0]
        time, loss, train_accuracy, test_accuracy, loss_curve = neural_network(
            label_count,
            features_with_dra,
            hot_labels,
            test_features_with_dra,
            hot_test_labels,
        )
        loss_curve = loss_curve[10:]
        print(
            "time, loss, train_accuracy, test_accuracy:",
            time, loss, train_accuracy, test_accuracy,
        )
        plt(list(range(1, len(loss_curve)+1)),
            loss_curve, label=dra_method.__name__)
    plot(
        name + ' DRA and or Clustering Fitness Loss Curves',
        xlabel="Iterations",
        ylabel="Fitness Loss",
    )


yoga_component_count = 12


def yoga():
    features, _, hot_labels, test_features, _, hot_test_labels = get_yoga_data(
        hot=True,
    )
    run(
        'Yoga',
        features,
        hot_labels,
        test_features,
        hot_test_labels,
        yoga_labels_count,
        yoga_component_count,
    )


def main():
    yoga()


if __name__ == "__main__":
    set_seed()
    main()
