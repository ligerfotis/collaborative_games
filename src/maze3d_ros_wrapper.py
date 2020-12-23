#!/mnt/34C28480C28447D6/PycharmProjects/maze3d_collaborative/venv/bin/python
import rospkg
import os
rospack = rospkg.RosPack()
package_path = rospack.get_path("collaborative_games")
os.chdir(package_path + "/src/")

import rospy
import std_msgs
from std_msgs.msg import Int16, Float32
# from environment import Game 
from collaborative_games.msg import observation, action_agent, reward_observation, action_msg
from std_srvs.srv import Empty,EmptyResponse, Trigger
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from collections import deque
import csv
from datetime import timedelta
from maze3D.Maze3DEnv import Maze3D
from maze3D.assets import *
from maze3D.config import pause
from rl_models.sac_agent import Agent
from rl_models.sac_discrete_agent import DiscreteSACAgent
from rl_models.utils import get_config, get_plot_and_chkpt_dir, plot_learning_curve, plot
from maze3D.utils import convert_actions
import time

import torch 

class RL_Maze3D:
    def __init__(self):
        # print("init")
        self.config = get_config()

        self.discrete = True
        # creating environment
        self.maze = Maze3D()
        self.chkpt_dir, self.plot_dir, self.timestamp = get_plot_and_chkpt_dir(self.config['game']['load_checkpoint'],
                                                                self.config['game']['checkpoint_name'], self.discrete)

        if self.discrete:
            self.sac = DiscreteSACAgent(config=self.config, env=self.maze, input_dims=self.maze.observation_shape,
                                   n_actions=self.maze.action_space.actions_number,
                                   chkpt_dir=self.chkpt_dir)
        else:
            self.sac = Agent(config=self.config, env=self.maze, input_dims=self.maze.observation_shape, n_actions=self.maze.action_space.shape,
                        chkpt_dir=self.chkpt_dir)
        """
        initialize human & agent actions
        """
        self.action_human = 0.0
        self.action_agent = 0.0
        """
        Create subscribers for human action
        act_human_sub_y is for the case that the agent's action is not used
        """
        self.act_human_sub = rospy.Subscriber("/rl/action_x", action_msg, self.set_human_action)
        # self.act_human_sub_y = rospy.Subscriber("/rl/action_y", action_msg, self.set_human_action_y)
        """
        Create publishers for turtle's acceleration, agent's action, 
        robot reset signal and turtle's position on x axis
        """
        self.turtle_accel_pub = rospy.Publisher("/rl/turtle_accel", Float32, queue_size = 10)
        self.act_agent_pub = rospy.Publisher("/rl/action_y", action_msg, queue_size = 10)
        self.reset_robot_pub = rospy.Publisher("/robot_reset", Int16, queue_size = 1)
        self.turtle_state_pub = rospy.Publisher("/rl/turtle_pos_X", Int16, queue_size = 10)

        random_seed = None
        if random_seed:
            torch.manual_seed(random_seed)

        
        self.best_score = -100 - 1 * self.config['Experiment']['max_timesteps']
        self.best_reward = self.best_score
        self.best_score_episode = -1
        self.best_score_length = -1
        # logging variables
        self.running_reward = 0
        self.avg_length = 0
        self.timestep = 1
        self.total_steps = 0

        # self.training_epochs_per_update = 128
        self.action_history = []
        self.score_history = []
        self.episode_duration_list = []
        self.length_list = []
        self.grad_updates_durations = []
        self.info = {}
        if self.config['game']['load_checkpoint']:
            self.sac.load_models()
            # env.render(mode='human')

        self.max_episodes = self.config['Experiment']['max_episodes']
        self.max_timesteps = self.config['Experiment']['max_timesteps']
        self.flag = True
        self.start_experiment = time.time()
        self.duration_pause_total = 0
        
    def set_human_action(self,action_human):
        """
        Gets the human action from the publisher.
        """
        if action_human.action != 0.0:
            self.action_human = -action_human.action
            # self.action_human_timestamp = action_human.header.stamp.to_sec()
            # self.time_real_act_list.append(self.action_human_timestamp)
            # self.real_act_list.append(self.action_human)
            # self.transmit_time_list.append(rospy.get_rostime().to_sec()  - action_human.header.stamp.to_sec())

    def main(self):
        # training loop
        for i_episode in range(1, self.max_episodes + 1):
            observation = self.maze.reset()
            timedout = False
            episode_reward = 0
            start = time.time()
            grad_updates_duration = 0
            if i_episode < self.config['Experiment']['start_training_step_on_episode']:  # Pure exploration
                print("Using Random Agent")
            else:
                if self.flag:
                    print("Using SAC Agent")
                    self.flag = False
            # actions = [0, 0, 0, 0]  # all keys not pressed
            duration_pause = 0
            save_models = True
            for self.timestep in range(self.max_timesteps + 1):
                self.total_steps += 1

                if self.discrete:
                    if i_episode < self.config['Experiment']['start_training_step_on_episode']:  # Pure exploration
                        agent_action = np.random.randint(0, self.maze.action_space.actions_number)
                        save_models = False
                    else:  # Explore with actions_prob
                        save_models = True
                        agent_action = self.sac.actor.sample_act(observation)
                else:
                    save_models = True
                    agent_action = self.sac.choose_action(observation)
                """
                Add the human part here
                """
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        return 1
                    if event.type == pg.KEYDOWN:
                        if event.key == pg.K_SPACE:
                            # print("space")
                            start_pause = time.time()
                            pause()
                            end_pause = time.time()
                            duration_pause += end_pause - start_pause
                #         if event.key in self.maze.keys:
                #             actions[self.maze.keys_fotis[event.key]] = 1
                #             # action_human += maze.keys[event.key]
                #     if event.type == pg.KEYUP:
                #         if event.key in self.maze.keys:
                #             actions[self.maze.keys_fotis[event.key]] = 0
                #             # action_human -= maze.keys[event.key]
                # # print(action)
                # # agent_action, human_action = action

                # agent_action = maze.action_space.sample()[0]
                # human_action = convert_actions(actions)[1]
                human_action = self.action_human
                action = [agent_action, human_action]
                self.action_history.append(action)

                if self.timestep ==self.max_timesteps:
                    timedout = True

                if self.discrete:
                    observation_, reward, done = self.maze.step(action, timedout, self.config['Experiment']['action_duration'])
                    self.sac.memory.add(observation, agent_action, reward, observation_, done)
                else:
                    observation_, reward, done = self.maze.step(action, timedout, self.config['Experiment']['action_duration'])
                    self.sac.remember(observation, agent_action, reward, observation_, done)

                if not self.config['game']['test_model']:
                    if self.discrete:
                        self.sac.learn()
                        self.sac.soft_update_target()
                    else:
                        self.sac.learn([observation, agent_action, reward, observation_, done])
                observation = observation_


                # ifself.total_steps >= start_training_step andself.total_steps % sac.target_update_interval == 0:
                #     sac.soft_update_target()

                self.running_reward += reward
                episode_reward += reward
                # if render:
                #     env.render()
                if done:
                    break

            end = time.time()
            if self.best_reward < episode_reward:
                self.best_reward = episode_reward
            self.duration_pause_total += duration_pause
            episode_duration = end - start - duration_pause
            self.episode_duration_list.append(episode_duration)
            self.score_history.append(episode_reward)
            avg_grad_updates_duration = grad_updates_duration /self.timestep
            self.grad_updates_durations.append(avg_grad_updates_duration)

            log_interval = self.config['Experiment']['log_interval']
            avg_ep_duration = np.mean(self.episode_duration_list[-log_interval:])
            avg_score = np.mean(self.score_history[-log_interval:])

            if avg_score >self.best_score:
                self.best_score = avg_score
                self.best_score_episode = i_episode
                self.best_score_length =self.timestep
                if not self.config['game']['test_model'] and save_models:
                    sac.save_models()

            self.length_list.append(self.timestep)
            self.avg_length +=self.timestep
            if not self.config['game']['test_model']:
                # off policy learning
                start_grad_updates = time.time()
                update_cycles = self.config['Experiment']['update_cycles']

                # ifself.total_steps >= self.config['Experiment'][
                #     'start_training_step'] andself.total_steps % sac.update_interval == 0:
                if i_episode % self.sac.update_interval == 0 and update_cycles > 0:
                    print("Performing {} updates".format(update_cycles))
                    for e in tqdm(range(update_cycles)):
                        if self.discrete:
                            self.sac.learn()
                            self.sac.soft_update_target()
                        else:
                            self.sac.learn()

                end_grad_updates = time.time()
                grad_updates_duration += end_grad_updates - start_grad_updates

            # logging
            if not self.config['game']['test_model']:
                if i_episode % log_interval == 0:
                    self.avg_length = int(self.avg_length / log_interval)
                    self.running_reward = int((self.running_reward / log_interval))

                    print('Episode {} \t avg length: {} \t Total reward(last {} episodes): {} \t Best Score: {} \t avg '
                          'episode duration: {} avg grad updates duration: {}'.format(i_episode,self.avg_length, log_interval,
                                                                                     self.running_reward,self.best_score,
                                                                                      timedelta(seconds=avg_ep_duration),
                                                                                      timedelta(
                                                                                          seconds=avg_grad_updates_duration)))
                    self.running_reward = 0
                    self.avg_length = 0
        end_experiment = time.time()
        experiment_duration = timedelta(seconds=end_experiment -self.start_experiment -self.duration_pause_total)
        self.info['experiment_duration'] = experiment_duration
        self.info['best_score'] =self.best_score
        self.info['best_score_episode'] =self.best_score_episode
        self.info['best_reward'] =self.best_reward
        self.info['best_score_length'] =self.best_score_length
        self.info['total_steps'] = self.total_steps
        self.info['fps'] = self.maze.fps

        print('Total Experiment time: {}'.format(experiment_duration))

        if not self.config['game']['test_model']:
            x = [i + 1 for i in range(len(self.score_history))]
            np.savetxt(self.chkpt_dir + '/scores.csv', np.asarray(self.score_history), delimiter=',')

            actions = np.asarray(self.action_history)
            # action_main = actions[0].flatten()
            # action_side = actions[1].flatten()
            x_actions = [i + 1 for i in range(len(actions))]
            # Save logs in files
            np.savetxt(self.chkpt_dir + '/actions.csv', actions, delimiter=',')
            # np.savetxt('tmp/sac_' + timestamp + '/action_side.csv', action_side, delimiter=',')
            np.savetxt(self.chkpt_dir + '/epidode_durations.csv', np.asarray(self.episode_duration_list), delimiter=',')
            np.savetxt(self.chkpt_dir + '/avg_length_list.csv', np.asarray(self.length_list), delimiter=',')
            w = csv.writer(open(self.chkpt_dir + '/rest_info.csv', "w"))
            for key, val in self.info.items():
                w.writerow([key, val])
            np.savetxt(self.chkpt_dir + '/grad_updates_durations.csv',self.grad_updates_durations, delimiter=',')

            plot_learning_curve(x,self.score_history, self.plot_dir + "/scores.png")
            # plot_actions(x_actions, action_main, plot_dir + "/action_main.png")
            # plot_actions(x_actions, action_side, plot_dir + "/action_side.png")
            plot(self.length_list, self.plot_dir + "/length_list.png", x=[i + 1 for i in range(self.max_episodes)])
            plot(self.episode_duration_list, self.plot_dir + "/epidode_durations.png", x=[i + 1 for i in range(self.max_episodes)])
            plot(self.grad_updates_durations, self.plot_dir + "/grad_updates_durations.png", x=[i + 1 for i in range(self.max_episodes)])

        pg.quit()
        return 0

if __name__ == '__main__':
    """ The manin caller of the file."""
    rospy.init_node('Maze3D_wrapper', anonymous=True)
    ctrl = RL_Maze3D()
    # ctrl.game.game_intro()

    if not ctrl.main():
        exit(0)

    # while ctrl.game.running:
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


