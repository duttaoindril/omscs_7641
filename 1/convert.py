import numpy as np
import matplotlib.pyplot as plt

days = 15

def show_plot(df, title, xlabel, ylabel):
    """Show plot with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()

def plot(
    title,
    xlabel=None,
    ylabel=None,
    xlim=None,
    ylim=None,
):
    """
    Plot and Save a chart

    :param title: Title of figure to plot
    :type id: string
    :param xlabel: Name of the X label
    :type xlabel: string
    :param ylabel: Name of the y label
    :type ylabel: string
    """
    if (xlabel is not None):
        plt.xlabel(xlabel)
    if (xlim is not None):
        plt.xlim(xlim)
    if (ylabel is not None):
        plt.ylabel(ylabel)
    if (ylim is not None):
        plt.ylim(ylim)
    plt.title(title)
    plt.legend()
    plt.savefig(f'{title}_figure.png')
    plt.close()

def handle_training_testing_plot(run, xs, title, xlabel, ylabel='Accuracy'):
    training_accuracies = []
    testing_accuracies = []
    for x in xs:
        training_accuracy, testing_accuracy = run(x)
        training_accuracies.append(training_accuracy)
        testing_accuracies.append(testing_accuracy)
    plt.plot(
        xs,
        training_accuracies,
        label='Training',
    )
    plt.plot(
        xs,
        testing_accuracies,
        label='Testing',
    )
    plot(
        title,
        xlabel=xlabel,
        ylabel=ylabel,
    )

if __name__ == "__main__":
    stock_data = np.genfromtxt('../stocks/ge.us.txt', delimiter=',')[1:, -3]
    print(stock_data)
    stock_data_rearranged = []
    for i in range(len(stock_data) - (days + 1)):
        a = np.around(stock_data[i : i + days], 2)
        e = (a - np.mean(a)) / np.std(a)
        stock_data_rearranged.append(np.append(a, 0 if stock_data[i + days + 1] >= stock_data[i + days] else 1))
    print(stock_data_rearranged)
    np.savetxt("stocks.csv", stock_data_rearranged, delimiter=",", fmt='%s')
