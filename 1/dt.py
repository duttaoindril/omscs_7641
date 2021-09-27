import numpy as np
import convert as util
from sklearn import tree as dt
from yoga import yoga_count, get_yoga_data
from stocks import stock_days, get_stock_data

def decision_tree(data, min_samples_leaf, max_depth=None, criterion='gini', should_normalize=True):
    train_features, train_labels, test_features, test_labels, normalize = data
    learner = dt.DecisionTreeClassifier(criterion=criterion, min_samples_leaf=min_samples_leaf, max_depth=max_depth)
    learner = learner.fit(normalize(train_features) if should_normalize else train_features, train_labels)
    train_accuracy = learner.score(normalize(train_features) if should_normalize else train_features, train_labels)
    test_accuracy = learner.score(normalize(test_features) if should_normalize else test_features, test_labels)
    return train_accuracy, test_accuracy

def stock():
    max_depth = None
    leaf_sizes = [*range(1,10,1),*range(10,50,5),*range(50,501,50)]
    util.handle_training_testing_plot(lambda leaf_size: decision_tree(get_stock_data(1), leaf_size, max_depth), leaf_sizes, "Stock Decision Tree Leaf Size vs Accuracy", 'Leaf Size')

    leaf_size = 500
    max_depths = [1, *range(int(stock_days/4), int(stock_days*2)+1, int(stock_days/4))]
    util.handle_training_testing_plot(lambda leaf_size: decision_tree(get_stock_data(1), leaf_size, max_depth), max_depths, "Stock Decision Tree Max Depth vs Accuracy", 'Max Depth')

    max_depth = None
    percents = [*np.linspace(0.1,1,10)]
    util.handle_training_testing_plot(lambda percent: decision_tree(get_stock_data(percent), leaf_size, max_depth), percents, "Stock Decision Tree Percent vs Accuracy", 'Percent')

def yoga():
    max_depth = None
    leaf_sizes = [*range(1,10,1),*range(10,50,5),*range(50,501,50)]
    util.handle_training_testing_plot(lambda leaf_size: decision_tree(get_yoga_data(1), leaf_size, max_depth), leaf_sizes, "Yoga Decision Tree Leaf Size vs Accuracy", 'Leaf Size')

    leaf_size = 1
    max_depths = [1, *range(int(yoga_count/4), int(yoga_count*2)+1, int(yoga_count/4))]
    util.handle_training_testing_plot(lambda leaf_size: decision_tree(get_yoga_data(1), leaf_size, max_depth), max_depths, "Yoga Decision Tree Max Depth vs Accuracy", 'Max Depth')

    max_depth = None
    percents = [*np.linspace(0.1,1,10)]
    util.handle_training_testing_plot(lambda percent: decision_tree(get_yoga_data(percent), leaf_size, max_depth), percents, "Yoga Decision Tree Percent vs Accuracy", 'Percent')

def main():
    stock()
    yoga()

if __name__ == "__main__":
    main()