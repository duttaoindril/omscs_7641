from tensorflow import keras as k
from yoga import yoga_count, get_yoga_data
from stocks import stock_days, get_stock_data

accuracy_key = 'categorical_accuracy'

def stock(epochs=100, validation_split=0.2, lossFn='kl_divergence', optimizerFn='adamax'):
    stock_train_features, stock_train_labels, stock_test_features, stock_test_labels, normalize = get_stock_data(1)

    results = {}

    for depth in range(1, 5):
        for width in range(int(stock_days/4), int(stock_days*8), int(stock_days/4)):
            print(f'RUNNING STOCK MODEL WITH depth {depth} AND width {width}')
            model = k.Sequential([normalize])
            for i in range(depth):
                model.add(k.layers.Dense(width))
            model.add(k.layers.Dense(2, activation='softmax'))
            model.compile(loss=lossFn, optimizer=optimizerFn, metrics=['mse', accuracy_key])
            model.fit(stock_train_features, stock_train_labels, validation_split=validation_split, epochs=epochs, verbose=0)
            model.summary()
            run_summary = model.evaluate(stock_test_features, stock_test_labels)
            results[f"{lossFn}_{optimizerFn}"] = run_summary

    print(results)

def yoga(epochs=100, validation_split=0.2, lossFn='sparse_categorical_crossentropy', optimizerFn='adam'):
    yoga_train_features, yoga_train_labels, yoga_test_features, yoga_test_labels, normalize = get_yoga_data(1)
            
    results = {}

    for depth in range(1, 5):
        for width in range(int(yoga_count/4), int(yoga_count*8), int(yoga_count/4)):
            print(f'RUNNING YOGA MODEL WITH depth {depth} AND width {width}')
            model = k.Sequential([normalize])
            for i in range(depth):
                model.add(k.layers.Dense(width))
            model.add(k.layers.Dense(4, activation='softmax'))
            model.compile(loss=lossFn, optimizer=optimizerFn, metrics=['mse', accuracy_key])
            model.fit(yoga_train_features, yoga_train_labels, validation_split=validation_split, epochs=epochs, verbose=0)
            model.summary()
            run_summary = model.evaluate(yoga_test_features, yoga_test_labels)
            results[f"{depth}_{width}"] = run_summary

    print(results)

if __name__ == "__main__":
    stock()
    # yoga()