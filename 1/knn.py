import numpy as np
import convert as util
from yoga import get_yoga_data
from stocks import get_stock_data
from sklearn.neighbors import KNeighborsClassifier

def decision_tree(data, k, weight, should_normalize=True):
    train_features, train_labels, test_features, test_labels, normalize = data
    knn = KNeighborsClassifier(n_neighbors=k, weights=weight)
    knn.fit(normalize(train_features) if should_normalize else train_features, train_labels)
    train_accuracy = knn.score(normalize(train_features) if should_normalize else train_features, train_labels)
    test_accuracy = knn.score(normalize(test_features) if should_normalize else test_features, test_labels)
    return train_accuracy, test_accuracy

def stock():
    k = 100
    weights = ['distance', 'uniform']
    util.handle_training_testing_plot(lambda weight: decision_tree(get_stock_data(1), k, weight), weights, "Stock KNN Weight vs Accuracy", 'Weight')

    weight = weights[0]
    ks = [*range(1,101)]
    util.handle_training_testing_plot(lambda k: decision_tree(get_stock_data(1), k, weight), ks, "Stock KNN K vs Accuracy", 'K')

    k = 100
    percents = [*np.linspace(0.1,1,10)]
    util.handle_training_testing_plot(lambda percent: decision_tree(get_stock_data(percent), k, weight), percents, "Stock KNN Percent vs Accuracy", 'Percent')

def yoga():
    k = 1
    weights = ['distance', 'uniform']
    util.handle_training_testing_plot(lambda weight: decision_tree(get_yoga_data(1), k, weight), weights, "Yoga KNN Weight vs Accuracy", 'Weight')

    weight = weights[0]
    ks = [*range(1,101)]
    util.handle_training_testing_plot(lambda k: decision_tree(get_yoga_data(1), k, weight), ks, "Yoga KNN K vs Accuracy", 'K')

    k = 1
    percents = [*np.linspace(0.1,1,10)]
    util.handle_training_testing_plot(lambda percent: decision_tree(get_yoga_data(percent), k, weight), percents, "Yoga KNN Percent vs Accuracy", 'Percent')

def main():
    stock()
    yoga()

if __name__ == "__main__":
    main()