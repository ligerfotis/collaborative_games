#!/mnt/34C28480C28447D6/PycharmProjects/maze3d_collaborative/venv/bin/python
import sys
import rospkg
import os
rospack = rospkg.RosPack()
package_path = rospack.get_path("collaborative_games")
os.chdir(package_path + "/src/maze3D_game")

from experiment import Experiment
import rospy
from datetime import timedelta
from maze3D.Maze3DEnv import Maze3D
from maze3D.assets import *
from rl_models.utils import get_config, get_plot_and_chkpt_dir, get_sac_agent
from maze3D.utils import convert_actions, save_logs_and_plot
import time

class RL_Maze3D:
    def __init__(self, argv):
        # get configuration
        self.config = get_config(argv[0])
        # creating environment
        self.maze = Maze3D(config_file=argv[0])

        # create the checkpoint and plot directories for this experiment
        self.chkpt_dir, self.plot_dir, self.timestamp = get_plot_and_chkpt_dir(self.config)

        # create the SAC agent
        self.sac = get_sac_agent(self.config, self.maze, self.chkpt_dir)

        # create the experiment
        self.experiment = Experiment(self.config, self.maze, self.sac)
        self.start_experiment = time.time()

    def main(self):
        # training loop
        loop = self.config['Experiment']['loop']
        if loop == 1:
            # Experiment 1
            self.experiment.loop_1()
        else:
            # Experiment 2
            self.experiment.loop_2()

        end_experiment = time.time()
        experiment_duration = timedelta(
            seconds=end_experiment - self.start_experiment - self.experiment.duration_pause_total)

        print('Total Experiment time: {}'.format(experiment_duration))

        # save training logs to a pickle file
        self.experiment.df_training_logs.to_pickle(self.plot_dir + '/training_logs.pkl')
        self.experiment.df_timing_x_logs.to_pickle(self.plot_dir + '/timing_x_logs.pkl')
        self.experiment.df_timing_y_logs.to_pickle(self.plot_dir + '/timing_y_logs.pkl')

        if not self.config['game']['test_model']:
            total_games = self.experiment.max_episodes if loop == 1 else self.experiment.game
            # save rest of the experiment logs and plot them
            save_logs_and_plot(self.experiment, self.chkpt_dir, self.plot_dir, total_games)
            self.experiment.save_info(self.chkpt_dir, experiment_duration, total_games)
        pg.quit()


if __name__ == '__main__':
    """ The manin caller of the file."""
    rospy.init_node('Maze3D_wrapper', anonymous=True)
    ctrl = RL_Maze3D(sys.argv[1:])
    if not ctrl.main():
        exit(0)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
