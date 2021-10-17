import random
import numpy as np
from time import time
import matplotlib.pyplot as plt


def start_time():
    start = time()

    def end_time():
        diff = time() - start
        print("Time: {:.2f}s".format(diff))
        return diff

    return end_time


def set_seed():
    seed = 3+7641+903459041
    random.seed(seed)
    np.random.seed(seed)


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


def plot_runner(problem_name, func, functions, sizes=[*range(2, 10, 2), *range(10, 50, 10), 50, 100]):
    data = {
        "Time": {
            "RHC": [],
            "SA": [],
            "GA": [],
            "MIMIC": [],
        },
        "Fitness": {
            "RHC": [],
            "SA": [],
            "GA": [],
            "MIMIC": [],
        },
        "Iterations": {
            "RHC": [],
            "SA": [],
            "GA": [],
            "MIMIC": [],
        },
    }
    print(f"PROBLEM {problem_name}")
    for size in sizes:
        print(f"RUNNING PROBLEM SIZE {size}")
        problem = func(size)
        for name in functions:
            results = functions[name](problem)
            i = 0
            for key in data:
                data[key][name].append(results[i])
                i += 1
    for key in data:
        for name in data[key]:
            plt.plot(sizes, data[key][name], label=name)
        plot(
            f'{problem_name} Optimizer {key} by Problem Size',
            xlabel=f'Problem Size',
            ylabel=f'{key}',
        )
