import random
import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as pltr
from matplotlib.pyplot import plot as plt
from mpl_toolkits.mplot3d import axes3d
from statistics import mean

# percents = [0.01, *np.linspace(0.1,0.9,9), 0.99]
# leaf_sizes = [*range(1,10,1),*range(10,50,5),*range(50,501,50)]

def start_time(label):
    start = time()

    def end_time():
        diff = time() - start
        print(label, "⏲  Time: {:.2f}s".format(diff))
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
        pltr.xlabel(xlabel)
    if (xlim is not None):
        pltr.xlim(xlim)
    if (ylabel is not None):
        pltr.ylabel(ylabel)
    if (ylim is not None):
        pltr.ylim(ylim)
    pltr.title(title)
    pltr.legend()
    pltr.savefig(f'{title}_figure.png')
    pltr.close()

def plot3D(
    title,
    xlabel='x axis',
    x=None,
    dx=None,
    ylabel='y axis',
    y=None,
    dy=None,
    zlabel='z axis',
    z=None,
    dz=None,
):
    """
    Plot and Save a chart

    :param title: Title of figure to plot
    :type id: string
    :param xlabel: Name of the X label
    """
    fig = pltr.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    dx = [1 for i in range(len(x))] if dx is None else dx
    dy = [1 for i in range(len(y))] if dy is None else dy
    dz = [0 for i in range(len(z))] if dz is None else dz
    ax1.bar3d(x, y, dz, dx, dy, z, shade=True)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_zlabel(zlabel)
    pltr.title(title)
    pltr.legend()
    # pltr.savefig(f'{title}_figure.png')
    pltr.show()
    pltr.close()

def kurtosis(data):
    return mean(pd.DataFrame(data).kurtosis())

# def plot_runner(problem_name, func, functions, sizes=[*range(2, 10, 2), *range(10, 50, 10), 50, 100]):
#     data = {
#         "Time": {
#             "RHC": [],
#             "SA": [],
#             "GA": [],
#             "MIMIC": [],
#         },
#         "Fitness": {
#             "RHC": [],
#             "SA": [],
#             "GA": [],
#             "MIMIC": [],
#         },
#         "Iterations": {
#             "RHC": [],
#             "SA": [],
#             "GA": [],
#             "MIMIC": [],
#         },
#     }
#     print(f"PROBLEM {problem_name}")
#     for size in sizes:
#         print(f"RUNNING PROBLEM SIZE {size}")
#         problem = func(size)
#         for name in functions:
#             results = functions[name](problem)
#             i = 0
#             for key in data:
#                 data[key][name].append(results[i])
#                 i += 1
#     for key in data:
#         for name in data[key]:
#             plt.plot(sizes, data[key][name], label=name)
#         plot(
#             f'{problem_name} Optimizer {key} by Problem Size',
#             xlabel=f'Problem Size',
#             ylabel=f'{key}',
#         )
