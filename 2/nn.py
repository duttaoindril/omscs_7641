import numpy as np
import mlrose_hiive as ml
from util import set_seed, start_time
from sklearn.metrics import accuracy_score
from yoga import yoga_count, get_yoga_data

algos = [
    'random_hill_climb',
    'simulated_annealing',
    'genetic_alg',
    'gradient_descent'
]
activations = ['tanh', 'tanh', 'relu', 'sigmoid']
learning_rates = [100, 100, 0.1, 0.001]
iterations = [10000, 10000, 1000, 10000]
attempts = [100, 100, 100, 10]


def neural_network(type):
    algo = algos[type]
    activation = activations[type]
    learning_rate = learning_rates[type]
    attempt = attempts[type]
    iteration = iterations[type]
    print(f"Running {activation} activation {algo} NN")
    print(f"Using LR {learning_rate}, {attempt}/{iteration} attempts/iters")
    train_features, train_labels, test_features, test_labels = get_yoga_data()
    model = ml.NeuralNetwork(
        # All Algorithms
        hidden_nodes=[yoga_count*2, yoga_count*2],
        bias=True,
        curve=True,
        is_classifier=True,
        early_stopping=True,
        # Pick Algorithm - RHC / SA / GA / GD
        algorithm=algo,
        activation=activation,
        learning_rate=learning_rate,
        max_attempts=attempt,
        max_iters=iteration,
        # RHC Only Parameter
        restarts=1,
        # SA Only Parameters
        schedule=ml.ExpDecay(exp_const=0.1),
        # GA Only Parameters
        pop_size=700,
        mutation_prob=0.1,
    )
    print("Training...")
    end_time = start_time()
    data = model.fit(train_features, train_labels)
    print("Results:")
    end_time()
    print(data.loss)
    print(data.fitness_curve)
    print(data.fitted_weights)
    print(data.predicted_probs)

    train_accuracy = accuracy_score(train_labels, model.predict(train_features))
    test_accuracy = accuracy_score(test_labels, model.predict(test_features))

    print('train_accuracy, test_accuracy')
    print(train_accuracy, test_accuracy)
    return train_accuracy, test_accuracy


def main():
    neural_network(0)
    neural_network(1)
    neural_network(2)
    neural_network(3)


if __name__ == "__main__":
    set_seed()
    main()
