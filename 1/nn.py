import convert as util
from tensorflow import keras as k
from yoga import yoga_count, get_yoga_data
from stocks import stock_days, get_stock_data

accuracy_key = 'categorical_accuracy'

def neural_network(data, depth, width, is_stock=False, with_history=False, epochs=100, validation_split=0.2, should_normalize=True):
    train_features, train_labels, test_features, test_labels, normalize = data
    model = k.Sequential([normalize])
    for _ in range(depth):
        model.add(k.layers.Dense(width))
    model.add(k.layers.Dense(2 if is_stock else 4, activation='softmax'))
    model.compile(loss='kl_divergence' if is_stock else 'sparse_categorical_crossentropy', optimizer='adamax' if is_stock else 'adam', metrics=['mse', accuracy_key])
    history = model.fit(train_features, train_labels,  epochs=epochs, validation_split=validation_split, verbose=0)
    run_summary = model.evaluate(test_features, test_labels)
    train_accuracy = history.history[accuracy_key][-1]
    test_accuracy = run_summary[-1]
    if with_history:
        return train_accuracy, test_accuracy, history
    else:
        return train_accuracy, test_accuracy

def stock():
    width = stock_days*2
    depths = [*range(1, 6)]
    util.handle_training_testing_plot(lambda depth: neural_network(get_stock_data(1), depth, width, True), depths, "Stock NN Depth vs Accuracy", 'Depth')

    depth = depths[-1]
    widths = [*range(int(stock_days/4), int(stock_days*3)+1, int(stock_days/4))]
    util.handle_training_testing_plot(lambda width: neural_network(get_stock_data(1), depth, width, True), widths, "Stock NN Width vs Accuracy", 'Width')

    width = stock_days*2
    _, _, history = neural_network(get_stock_data(1), depth, width, True, True)
    epochs = [*range(1,history.params['epochs']+1)]
    training_accuracies = history.history[accuracy_key]
    testing_accuracies = history.history['val_'+accuracy_key]
    util.handle_training_testing_plot(lambda epoch: (training_accuracies[epoch-1], testing_accuracies[epoch-1]), epochs, "Stock NN Epoch vs Accuracy", 'Epoch')

def yoga():
    width = yoga_count*2
    depths = [*range(1, 6)]
    util.handle_training_testing_plot(lambda depth: neural_network(get_yoga_data(1), depth, width, False), depths, f"Yoga NN Depth vs Accuracy", 'Depth')

    depth = depths[-1]
    widths = [*range(int(yoga_count/4), int(yoga_count*3)+1, int(yoga_count/4))]
    util.handle_training_testing_plot(lambda width: neural_network(get_yoga_data(1), depth, width, False), widths, "Yoga NN Width vs Accuracy", 'Width')

    width = yoga_count*2
    _, _, history = neural_network(get_yoga_data(1), depth, width, False, True)
    epochs = [*range(1,history.params['epochs']+1)]
    training_accuracies = history.history[accuracy_key]
    testing_accuracies = history.history['val_'+accuracy_key]
    util.handle_training_testing_plot(lambda epoch: (training_accuracies[epoch-1], testing_accuracies[epoch-1]), epochs, "Yoga NN Epoch vs Accuracy", 'Epoch')

def main():
    stock()
    yoga()

if __name__ == "__main__":
    main()