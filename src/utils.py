import matplotlib.pyplot as plt
import numpy as np
from hyperparams_ur10 import OFF_POLICY_BATCH_SIZE as BATCH_SIZE, DISCOUNT, ENTROPY_WEIGHT, HIDDEN_SIZE, LEARNING_RATE, MAX_STEPS, POLYAK_FACTOR, REPLAY_SIZE, UPDATE_INTERVAL, UPDATE_START, SAVE_INTERVAL, OFFLINE_UPDATES, TEST_NUM, ACTION_DURATION
import rospkg
import sys
import os
import math

def plot(time_elpsd, data, figure_title, y_axis_name, x_axis_name, path, save=True, variance=False, stdev=None, color='-ok', plt_type="simple"):
    plt.figure(figure_title)
    plt.grid()
    # plt.xticks(np.arange(0, time_elpsd[-1], step=500))
    # plt.yticks(np.arange(min(list), max(list), step=0.01))
    if plt_type=="hist":
        bins = np.linspace(min(data), max(data), 20) # fixed number of bins
        plt.hist(data, bins=bins, alpha=0.5)
    elif plt_type == "boxplot":
        plt.boxplot(data) 
    elif plt_type == "simple":
        plt.plot(time_elpsd, data, color)
    elif plt_type == "log":
        plt.plot(time_elpsd, data, color)
        plt.yscale('log')

    plt.ylabel(y_axis_name)
    plt.xlabel(x_axis_name)
    if variance:
        plt.fill_between(time_elpsd, np.array(data) - np.array(stdev), np.array(data) + np.array(stdev))
    if save:
        plt.savefig(path + figure_title)
        plt.show()
    else:
        plt.show()


if __name__ == "__main__":

    if sys.argv[1] == "trials":
        filename_mean = sys.argv[2]
        filename_sdtev = sys.argv[3]
        
        print(filename_mean)
        print(filename_sdtev)

        means_list = np.genfromtxt(filename_mean, delimiter=',')  
        stdev_list = np.genfromtxt(filename_sdtev, delimiter=',')  

        rospack = rospkg.RosPack()
        package_path = rospack.get_path("hand_direction")
        plot_directory = package_path + "/src/plots/"
        if not os.path.exists(plot_directory):
            print("Dir %s was not found. Creating it..." %(plot_directory))
            os.makedirs(plot_directory)
        
        plot(range(UPDATE_INTERVAL, MAX_STEPS+UPDATE_INTERVAL, UPDATE_INTERVAL), means_list, "trials", 'Tests Score', 'Number of Interactions', plot_directory, save=True, variance=True, stdev=stdev_list)
    elif sys.argv[1] == "actions":

        filename_actions = sys.argv[3]
        filename_time = sys.argv[4]

        actions = np.genfromtxt(filename_actions, delimiter=',')  
        time = np.genfromtxt(filename_time, delimiter=',')

        rospack = rospkg.RosPack()
        package_path = rospack.get_path("hand_direction")
        plot_directory = package_path + "/src/plots/"
        if not os.path.exists(plot_directory):
            print("Dir %s was not found. Creating it..." %(plot_directory))
            os.makedirs(plot_directory)
        if sys.argv[2] == "simple":
            plot(time, actions, "Actions_Simple", 'Actions', 'Timestamp', plot_directory, save=True)
        elif sys.argv[2] == "simple_diff":
            plot(time[1:], actions[1:] - actions[:-1], "Actions_Simple_diff", 'Actions Change', 'Timestamp', plot_directory, save=True)
        elif sys.argv[2] == "hist":
            plot(time, actions, "Actions_Hist", 'Actions', 'Timestamp', plot_directory, save=True, plt_type="hist")
        elif sys.argv[2] == "hist_diff":
            plot(time[1:], (actions[1:] - actions[:-1])/10, "Actions_Hist_diff", 'Actions Change', 'Timestamp', plot_directory, save=True, plt_type="hist")
        elif sys.argv[2] == "boxplot":
            plot(time, actions, "Actions_Boxplot", 'Actions', 'Timestamp', plot_directory, save=True, plt_type="boxplot") 
        elif sys.argv[2] == "boxplot_diff":
            plot(time[1:], (actions[1:] - actions[:-1])/10, "Actions_Boxplot_diff", 'Actions Change', 'Timestamp', plot_directory, save=True, plt_type="boxplot") 
        elif sys.argv[2] == "log":
            plot(time[1:], (actions[1:] - actions[:-1])/10, "Actions_Boxplot_diff", 'Actions Change', 'Timestamp', plot_directory, save=True, plt_type="log") 

