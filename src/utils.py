import matplotlib.pyplot as plt
import numpy as np
from hyperparams_ur10 import OFF_POLICY_BATCH_SIZE as BATCH_SIZE, DISCOUNT, ENTROPY_WEIGHT, HIDDEN_SIZE, LEARNING_RATE, MAX_STEPS, POLYAK_FACTOR, REPLAY_SIZE, UPDATE_INTERVAL, UPDATE_START, SAVE_INTERVAL, OFFLINE_UPDATES, TEST_NUM, ACTION_DURATION
import rospkg
import sys
import os

def plot(time_elpsd, data, figure_title, y_axis_name, x_axis_name, path, save=True, variance=False, stdev=None, color='-xk'):
    plt.figure(figure_title)
    plt.grid()
    # plt.xticks(np.arange(0, time_elpsd[-1], step=500))
    # plt.yticks(np.arange(min(list), max(list), step=0.01))
    plt.plot(time_elpsd, data, color)
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
    filename_mean = sys.argv[1]
    filename_sdtev = sys.argv[2]
    
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
