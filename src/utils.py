import matplotlib.pyplot as plt
import numpy as np


def plot(time_elpsd, data, figure_title, y_axis_name, x_axis_name, path, save=True, variance=False, stdev=None):
    plt.figure(figure_title)
    plt.grid()
    # plt.xticks(np.arange(0, time_elpsd[-1], step=500))
    # plt.yticks(np.arange(min(list), max(list), step=0.01))
    plt.plot(time_elpsd, data)
    plt.ylabel(y_axis_name)
    plt.xlabel(x_axis_name)
    if variance:
        plt.fill_between(time_elpsd, np.array(data) - np.array(stdev), np.array(data) + np.array(stdev))
    if save:
        plt.savefig(path + figure_title)
        plt.show()
    else:
        plt.show()
