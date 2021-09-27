import convert as util
from tensorflow import keras as k
from yoga import yoga_count, get_yoga_data
from stocks import stock_days, get_stock_data

# Gaussian: K(x, y) == exp(- square(x - y) / (2 * square(scale)))
# Laplacian: K(x, y) = exp(-abs(x - y) / scale))

accuracy_key = 'categorical_accuracy'

def support_vector_machine(data, kernel_function, learning_rate=1e-3, is_stock=False, with_history=False, epochs=100, validation_split=0.2, should_normalize=True):
    train_features, train_labels, test_features, test_labels, normalize = data
    model = k.Sequential(
        [
            normalize,
            k.Input(shape=(stock_days if is_stock else yoga_count)),
            k.layers.experimental.RandomFourierFeatures(output_dim=4096, kernel_initializer=kernel_function, trainable=True),
            k.layers.Dense(2 if is_stock else 4, activation='softmax')
        ]
    )
    model.compile(
        optimizer=k.optimizers.Adam(learning_rate=learning_rate),
        loss=k.losses.hinge,
        metrics=['mse',accuracy_key],
    )
    history = model.fit(train_features, train_labels, epochs=epochs, validation_split=validation_split, verbose=0)
    run_summary = model.evaluate(test_features, test_labels)
    train_accuracy = history.history[accuracy_key][-1]
    test_accuracy = run_summary[-1]
    if with_history:
        return train_accuracy, test_accuracy, history
    else:
        return train_accuracy, test_accuracy

def stock():
    learning_rate = 1e-1
    kernel_functions = ['gaussian', 'laplacian']
    util.handle_training_testing_plot(lambda kernel_function: support_vector_machine(get_stock_data(1), kernel_function, learning_rate, True), kernel_functions, "Stock SVM Kernel Function vs Accuracy", 'Kernel Function')

    kernel_function = kernel_functions[0]
    learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    util.handle_training_testing_plot(lambda learning_rate: support_vector_machine(get_stock_data(1), kernel_function, learning_rate, True), learning_rates, "Stock SVM Learning Rate vs Accuracy", 'Learning Rate')

    learning_rate = learning_rates[0]
    a, b, history = support_vector_machine(get_stock_data(1), kernel_function, learning_rate, True, True)
    epochs = [*range(1,history.params['epochs']+1)]
    training_accuracies = history.history[accuracy_key]
    testing_accuracies = history.history['val_'+accuracy_key]
    util.handle_training_testing_plot(lambda epoch: (training_accuracies[epoch-1], testing_accuracies[epoch-1]), epochs, "Stock SVM Epoch vs Accuracy", 'Epoch')

def yoga():
    learning_rate = 1e-3
    kernel_functions = ['gaussian', 'laplacian']
    util.handle_training_testing_plot(lambda kernel_function: support_vector_machine(get_yoga_data(1), kernel_function, learning_rate, False), kernel_functions, "Yoga SVM Kernel Function vs Accuracy", 'Kernel Function')

    kernel_function = kernel_functions[0]
    learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    util.handle_training_testing_plot(lambda learning_rate: support_vector_machine(get_yoga_data(1), kernel_function, learning_rate, False), learning_rates, "Yoga SVM Learning Rate vs Accuracy", 'Learning Rate')

    learning_rate = learning_rates[2]
    a, b, history = support_vector_machine(get_yoga_data(1), kernel_function, learning_rate, False, True)
    epochs = [*range(1,history.params['epochs']+1)]
    training_accuracies = history.history[accuracy_key]
    testing_accuracies = history.history['val_'+accuracy_key]
    util.handle_training_testing_plot(lambda epoch: (training_accuracies[epoch-1], testing_accuracies[epoch-1]), epochs, "Yoga SVM Epoch vs Accuracy", 'Epoch')

def main():
    stock()
    # yoga()

if __name__ == "__main__":
    main()