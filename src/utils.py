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
        print (path + figure_title)
        plt.savefig(path + figure_title, dpi=150)
        # plt.show()
    else:
        plt.show()

def plot_hist(data, path, title):
    plt.figure()
    plt.title(title)
    plt.grid()
    bins = np.linspace(min(data), max(data), 40) # fixed number of bins
    plt.hist(data, bins=bins, alpha=1)
    plt.savefig(path, dpi=150)


def subplot(plot_directory, turtle_pos, turtle_vel, turtle_acc, time_turtle_pos, time_turtle_vel, time_turtle_acc, real_act_list, time_real_act_list, axis, plot_type):
    fig, axs = plt.subplots(4,figsize=(15,7))
    # fig.tight_layout()
    
    plt.sca(axs[0])
    axs[0].set_ylim([-10, 800])
    axs[0].set_xlim([min(time_real_act_list), max(time_real_act_list)])
    if axis == "x":
        plt.title("Human")
    elif axis == "y":
        plt.title("Agent")
    plt.yticks(np.arange(0, 800, step=100))
    plt.xticks(np.arange(math.ceil(min(time_real_act_list)), math.ceil(max(time_real_act_list)), 1))
    axs[0].axes.xaxis.set_ticklabels([])

    plt.ylabel("Turtle Pos " + axis)
    # plt.xlabel("System Time")

    # axs[0].set_title('Turtle Position X')
    # axs[0].plot(time_turtle_pos, turtle_pos, "-ok")
    axs[0].scatter(time_turtle_pos, turtle_pos, s=1)
    
    axs[0].grid()

    plt.sca(axs[1])
    # axs[1].set_ylim([-.10, .11])
    axs[1].set_xlim([min(time_real_act_list), max(time_real_act_list)])

    # plt.yticks(np.arange(-.10, .11, step=.25))
    plt.xticks(np.arange(math.ceil(min(time_real_act_list)), math.ceil(max(time_real_act_list)), 1))
    axs[1].axes.xaxis.set_ticklabels([])


    plt.ylabel("Turtle Vel " + axis)
    # plt.xlabel("System Time")

    # axs[1].set_title('Turtle Velocity X')
    # axs[1].plot(time_turtle_vel, turtle_vel, "-ok")
    axs[1].scatter(time_turtle_vel, turtle_vel, s=1)

    axs[1].grid()

    plt.sca(axs[2])
    # axs[2].set_ylim([-0.18, 0.19])
    # axs[2].set_ylim([-0.11, 0.11])
    axs[2].set_xlim([min(time_real_act_list), max(time_real_act_list)])

    # plt.yticks(np.arange(-0.18, 0.19, step=0.06))
    # plt.yticks(np.arange(-0.1, 0.11, step=0.05))
    plt.xticks(np.arange(math.ceil(min(time_real_act_list)), math.ceil(max(time_real_act_list)), 1))

    axs[2].axes.xaxis.set_ticklabels([])

    plt.ylabel("Turtle Accel " + axis)
    # plt.xlabel("System Time")

    # axs[2].set_title('Turtle Acceleration X')
    # axs[2].plot(time_turtle_acc, turtle_acc, "-ok")
    axs[2].scatter(time_turtle_acc, turtle_acc, s=1)
    
    axs[2].grid()

    plt.sca(axs[3])
    
    # axs[3].set_ylim([-1.1, 1.1])
    # axs[3].set_ylim([-8, 9])
    axs[3].set_xlim([min(time_real_act_list), max(time_real_act_list)])
    if plot_type == "accell":
        if axis == "x":
            axs[3].set_ylim([-0.1, 0.11])
            plt.yticks(np.arange(-0.1, 0.11, step=0.05))
        elif axis == "y":
            axs[3].set_ylim([-1.1, 1.1])
            plt.yticks(np.arange(-1.1, 1.1, step=0.5))
    elif plot_type == "accell_dir":
        axs[3].set_ylim([-1.1, 1.1])
        plt.yticks(np.arange(-1, 0.1, step=1))
        
    # plt.yticks(np.arange(-1, 1.1, step=1))
    # plt.yticks(np.arange(-8, 9, step=2))
    plt.xticks(np.arange(math.ceil(min(time_real_act_list)), math.ceil(max(time_real_act_list)), 1),rotation=30)

    plt.ylabel(axis + " Action")
    plt.xlabel("Time")
    
    # axs[3].set_title('Hand Movement X')
    # axs[3].plot(time_real_act_list, real_act_list, "-ok")
    axs[3].scatter(time_real_act_list, real_act_list, s=2)

    axs[3].grid()
    if axis == "x":
        plt.savefig(plot_directory + "Turtle_Human_Comparison",dpi=150)
        print (plot_directory + "Turtle_Human_Comparison")
    elif axis == "y":
        plt.savefig(plot_directory + "Turtle_Agent_Comparison",dpi=150)
        print (plot_directory + "Turtle_Agent_Comparison")

    # plt.show()



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
        experiment_num = sys.argv[4] 
        plot_type = sys.argv[3]
        
        try:
            filename_actions = sys.argv[5]
            actions = np.genfromtxt(filename_actions, delimiter=',')
        except Exception:
            print("Action argument not given")
        try:
            filename_time = sys.argv[6]
            time = np.genfromtxt(filename_time, delimiter=',')
        except Exception:
            print("Time argument not given")

        rospack = rospkg.RosPack()
        package_path = rospack.get_path("hand_direction")
        plot_directory = package_path + "/src/plots/"+ plot_type + "/"
        plot_directory += "Experiment_" + experiment_num + "/"


        if not os.path.exists(plot_directory):
            print("Dir %s was not found. Creating it..." %(plot_directory))
            os.makedirs(plot_directory)

        if sys.argv[2] == "simple":
            plot(range(len(actions)), actions, "Actions_Simple", 'Actions', 'Timestamp', plot_directory, save=True)
        
        elif sys.argv[2] == "simple_diff":
            plot(time[1:], actions[1:] - actions[:-1], "Actions_Simple_diff", 'Actions Change', 'Timestamp', plot_directory, save=True)
        
        elif sys.argv[2] == "hist":
            # fps = np.genfromtxt(plot_directory+"fps_list.csv", delimiter=',')
            plot(None, actions, "fps_list", 'Histogram', 'FPS', plot_directory, save=True, plt_type="hist")
        
        elif sys.argv[2] == "hist_diff":
            plot(time[1:], (actions[1:] - actions[:-1])/10, "Actions_Hist_diff", 'Actions Change', 'Timestamp', plot_directory, save=True, plt_type="hist")
        
        elif sys.argv[2] == "boxplot":
            plot(time, actions, "Actions_Boxplot", 'Actions', 'Timestamp', plot_directory, save=True, plt_type="boxplot") 
        
        elif sys.argv[2] == "boxplot_diff":
            plot(time[1:], (actions[1:] - actions[:-1])/10, "Actions_Boxplot_diff", 'Actions Change', 'Timestamp', plot_directory, save=True, plt_type="boxplot") 
        
        elif sys.argv[2] == "log":
            plot(time[1:], (actions[1:] - actions[:-1])/10, "Actions_Boxplot_diff", 'Actions Change', 'Timestamp', plot_directory, save=True, plt_type="log") 

    elif sys.argv[1] == "comparison":

        plot_type = sys.argv[2]

        axis = sys.argv[3]

        rospack = rospkg.RosPack()
        package_path = rospack.get_path("hand_direction")
        plot_directory = package_path + "/src/plots/" + plot_type +"/"
        plot_directory += "/Experiment_" + str(exp_num) + run_subfolder + "/turtle_dynamics/"

        exp_num = len([name for name in os.listdir('.') if os.path.isfile(name)])


        for exp in range(exp_num):
            run_subfolder = "game_" + run_num + "/"
            plot_directory += "Experiment_" + str(exp) + run_subfolder + "/turtle_dynamics/"

            turtle_pos = np.genfromtxt(plot_directory+"turtle_pos_"+axis+".csv", delimiter=',')
            turtle_vel = np.genfromtxt(plot_directory+"turtle_vel_"+axis+".csv", delimiter=',')
            turtle_acc = np.genfromtxt(plot_directory+"turtle_accel_"+axis+".csv", delimiter=',')

            time_turtle_pos = np.genfromtxt(plot_directory+"time_turtle_pos.csv", delimiter=',')
            time_turtle_vel = np.genfromtxt(plot_directory+"time_turtle_vel.csv", delimiter=',')
            time_turtle_acc = np.genfromtxt(plot_directory+"time_turtle_acc.csv", delimiter=',')

            if axis == "x":
                real_act_list = np.genfromtxt(plot_directory+"human_act_list.csv", delimiter=',')  
                time_real_act_list = np.genfromtxt(plot_directory + "action_timesteps.csv", delimiter=',')
            elif axis == "y":
                real_act_list = np.genfromtxt(plot_directory+"agent_act_list.csv", delimiter=',')  
                time_real_act_list = np.genfromtxt(plot_directory + "action_timesteps.csv", delimiter=',')
            
            # div = 4
            # turtle_pos = turtle_pos[:len(turtle_pos)/div]
            # turtle_vel = turtle_vel[:len(turtle_vel)/div]
            # turtle_acc = turtle_acc[:len(turtle_acc)/div]
            # real_act_list = real_act_list[:len(real_act_list)/div]

            # time_turtle_pos = time_turtle_pos[:len(time_turtle_pos)/div]
            # time_turtle_vel = time_turtle_vel[:len(time_turtle_vel)/div]
            # time_turtle_acc = time_turtle_acc[:len(time_turtle_acc)/div]
            # time_real_act_list = time_real_act_list[:len(time_real_act_list)/div]

            subplot(plot_directory, turtle_pos, turtle_vel, turtle_acc, time_turtle_pos, time_turtle_vel, time_turtle_acc, real_act_list, time_real_act_list, axis, plot_type)


    elif sys.argv[1] == "plot_x":

        plot_type = sys.argv[2]
        rospack = rospkg.RosPack()
        package_path = rospack.get_path("hand_direction")
        plot_directory = package_path + "/src/plots/" + plot_type + "/"

        turtle_pos_x = np.genfromtxt(plot_directory+"x_points_"+plot_type+".csv", delimiter=',')[:1000]
        time_turtle_pos_x = np.genfromtxt(plot_directory+"x_points_time_"+plot_type+".csv", delimiter=',')[:1000]

        human_action = np.genfromtxt(plot_directory+'human_action_'+plot_type+'.csv', delimiter=',')[:1000]
        human_action_time = np.genfromtxt(plot_directory+'human_action_time'+plot_type+'.csv', delimiter=',')[:1000]

        turtle_acc_x = np.genfromtxt(plot_directory+'x_turtle_accel_'+plot_type+'.csv', delimiter=',')[:1000]
        time_turtle_acc_x = np.genfromtxt(plot_directory+'x_turtle_accel_time_' + plot_type + '.csv', delimiter=',')[:1000]

        plt.figure()
        plt.title('Human Hand keypoints on x-axis-'+ plot_type)
        ax = plt.gca()
        # ax.set_ylim([-10, 800])
        # ax.set_xlim([min(time_turtle_pos_x), max(time_turtle_pos_x)+1])

        plt.yticks(np.arange(min(turtle_pos_x), max(turtle_pos_x)+1, step=0.07))
        # plt.xticks(np.arange(min(time_real_act_list), max(time_real_act_list)+1, 3))
       
        plt.ylabel("Pos X")
        plt.xlabel("System Time")

        # axs[0].set_title('Turtle Position X')
        ax.plot(time_turtle_pos_x, turtle_pos_x, '-ok')
        ax.grid()
        plt.savefig(plot_directory + "human_hand_pos_x_" + plot_type,dpi=300)

        fig, axs = plt.subplots(3,figsize=(15,10))
        # fig.tight_layout()
        plt.sca(axs[0])
        plt.title('Human Hand keypoints on x-axis and Actions-'+ plot_type)

        # axs[0].set_ylim([-10, 800])
        axs[0].set_xlim([min(time_turtle_pos_x), max(time_turtle_pos_x)])

        plt.yticks(np.arange(min(turtle_pos_x)-1, max(turtle_pos_x)+1, step=0.07))
        # plt.xticks(np.arange(math.ceil(min(time_turtle_pos_x))-1, math.ceil(max(time_turtle_pos_x))+1, 0.2))
        axs[0].axes.xaxis.set_ticklabels([])

        plt.ylabel("Pos X Human Hand")
        # plt.xlabel("System Time")

        # axs[0].set_title('Turtle Position X')
        axs[0].plot(time_turtle_pos_x, turtle_pos_x, '-ok')
        axs[0].grid()

        plt.sca(axs[1])
        if plot_type == "direction":
            axs[1].set_ylim([min(human_action)-0.5, max(human_action)+1])
        elif plot_type == "accel":
            axs[1].set_ylim([min(human_action)-0.1, max(human_action)+0.1])
        # axs[1].set_ylim([min(human_action)-0.5, max(human_action)+1])
        axs[1].set_xlim([min(time_turtle_pos_x), max(time_turtle_pos_x)])

        if plot_type == "direction":
            plt.yticks(np.arange(min(human_action), max(human_action)+1, step=1))
        elif plot_type == "accel":
            plt.yticks(np.arange(min(human_action), max(human_action)+0.1, step=0.05))
        # plt.yticks(np.arange(min(human_action), max(human_action)+1, step=1))
        # plt.xticks(np.arange(math.ceil(min(time_turtle_pos_x))-1, math.ceil(max(time_turtle_pos_x))+1, 0.2))
        axs[1].axes.xaxis.set_ticklabels([])

        plt.ylabel("Action Human")
        # plt.xlabel("System Time")

        # axs[1].set_title('Turtle Velocity X')
        axs[1].scatter(human_action_time, human_action)
        axs[1].grid()

        plt.sca(axs[2])
        # axs[2].set_ylim([min(human_action)-0.5, max(human_action)+1])
        axs[2].set_xlim([min(time_turtle_pos_x), max(time_turtle_pos_x)])


        # plt.yticks(np.arange(min(human_action), max(human_action)+1, step=1))

        # plt.xticks(np.arange(math.ceil(min(time_turtle_pos_x))-1, math.ceil(max(time_turtle_pos_x))+1, 0.2))
        # axs[1].axes.xaxis.set_ticklabels([])

        plt.ylabel("Turtle Accel")
        plt.xlabel("System Time")

        # axs[1].set_title('Turtle Velocity X')
        axs[2].plot(time_turtle_acc_x, turtle_acc_x, "-ok")
        axs[2].grid()

        plt.savefig(plot_directory + "human_hand_pos_x_action_" + plot_type,dpi=300)
        # plt.show()
    elif sys.argv[1] == "plot_rl":
        plot_type = sys.argv[2]
        exp_num = sys.argv[3]
        rospack = rospkg.RosPack()
        package_path = rospack.get_path("hand_direction")
        plot_directory = package_path + "/src/plots/" + plot_type +"/"
        plot_directory += "/Experiment_" + str(exp_num) + '/rl_dynamics/'

        alpha_values = np.genfromtxt(plot_directory + 'alpha_values.csv', delimiter=',')
        policy_loss_list = np.genfromtxt(plot_directory + 'policy_loss.csv', delimiter=',')
        value_loss_list = np.genfromtxt(plot_directory + 'value_loss.csv', delimiter=',')
        rewards_list = np.genfromtxt(plot_directory + 'rewards_list.csv', delimiter=',')
        turn_list = np.genfromtxt(plot_directory + 'turn_list.csv', delimiter=',')


        critics_lr_list = np.genfromtxt(plot_directory + 'critics_lr_list.csv', delimiter=',')
        value_critic_lr_list = np.genfromtxt(plot_directory + 'value_critic_lr_list.csv', delimiter=',')
        actor_lr_list = np.genfromtxt(plot_directory + 'actor_lr_list.csv', delimiter=',')

        mean_list = np.genfromtxt(plot_directory + 'means.csv', delimiter=',')
        stdev_list = np.genfromtxt(plot_directory + 'stdev.csv', delimiter=',')

        plot(range(len(alpha_values)), alpha_values, "alpha_values", 'Alpha Value', 'Number of Gradient Updates', plot_directory, save=True)
        plot(range(len(policy_loss_list)), policy_loss_list, "policy_loss", 'Policy loss', 'Number of Gradient Updates', plot_directory, save=True)
        plot(range(len(value_loss_list)), value_loss_list, "value_loss_list", 'Value loss', 'Number of Gradient Updates', plot_directory, save=True)
        plot(range(len(rewards_list)), rewards_list, "Rewards_per_game", 'Total Rewards per Game', 'Number of Games', plot_directory, save=True)
        plot(range(len(turn_list)), turn_list, "Steps_per_game", 'Turns per Game', 'Number of Games', plot_directory, save=True) 

        plot(range(len(critics_lr_list)), critics_lr_list, "critics_lr_list", 'Critic lr', 'Number of Gradient Updates', plot_directory, save=True)
        plot(range(len(value_critic_lr_list)), value_critic_lr_list, "value_critic_lr_list", 'Value lr', 'Number of Gradient Updates', plot_directory, save=True)
        plot(range(len(actor_lr_list)), actor_lr_list, "actor_lr_list", 'Actor lr', 'Number of Gradient Updates', plot_directory, save=True) 

        plot(range(UPDATE_INTERVAL, MAX_STEPS + UPDATE_INTERVAL, UPDATE_INTERVAL), mean_list, "trials", 'Tests Score', 'Number of Interactions', plot_directory, save=True, variance=True, stdev=stdev_list)     

