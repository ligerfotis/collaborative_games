import csv
import math
from statistics import mean
import pandas as pd
import time
from datetime import timedelta
from maze3D.Maze3DEnv import Maze3D
from maze3D.assets import *
from maze3D.config import pause
import numpy as np
from tqdm import tqdm
from maze3D.utils import convert_actions
import rospy
from collaborative_games.msg import observation, action_agent, reward_observation, action_msg

log_column_names = ["actions_x", "actions_y", "tray_rot_x", "tray_rot_y", "tray_rot_vel_x", "tray_rot_vel_y",
                            "ball_pos_x", "ball_pos_y", "ball_vel_x", "ball_vel_y"]
time_column_names_x = ["action_x", "act_x_created", "transmit_time_act_x"]
time_column_names_y = ["action_x", "act_y_created", "transmit_time_act_y"]


class Experiment:
    def __init__(self, config, environment, agent):
        self.config = config
        self.env = environment
        self.agent = agent
        self.best_score = None
        self.best_reward = None
        self.best_score_episode = -1
        self.best_score_length = -1
        self.total_steps = 0
        self.action_history = []
        self.score_history = []
        self.episode_duration_list = []
        self.length_list = []
        self.grad_updates_durations = []
        self.discrete = config['SAC']['discrete']
        self.second_human = config['game']['second_human']
        self.duration_pause_total = 0
        if self.config['game']['load_checkpoint']:
            self.agent.load_models()
        # self.df = pd.DataFrame(columns=column_names)
        self.max_episodes = None
        self.max_timesteps = None
        self.grad_updates_durations = []
        self.avg_grad_updates_duration = 0
        self.human_actions = None
        self.agent_action = None
        self.total_timesteps = None
        self.max_timesteps_per_game = None
        self.save_models = True
        self.game = None

        # initialize human & agent actions
        self.human_action = 0.0
        self.action_second_human = 0.0
        self.agent_action = 0.0
        
        # Create subscribers for human action
        self.act_human_sub = rospy.Subscriber("/rl/action_x", action_msg, self.set_human_action_x)
        self.act_human_sub_y = rospy.Subscriber("/rl/action_y", action_msg, self.set_human_action_y)

        self.df_training_logs = pd.DataFrame(columns=log_column_names)
        self.df_timing_x_logs = pd.DataFrame(columns=time_column_names_x)
        self.df_timing_y_logs = pd.DataFrame(columns=time_column_names_y)

        self.human_act_transmition_time_list = []

    def set_human_action_x(self,action_human):
        """
        Gets the human action from the publisher.
        """
        if action_human.action != 0.0:
            self.human_action = -action_human.action

            act_x_time_created = action_human.header.stamp.to_sec()
            self.human_act_transmition_time_list.append(time.time() - act_x_time_created)
            # self.act_x_time_created_list.append(act_x_time_created)
            # self.transmit_time_act_x_list.append(rospy.get_rostime().to_sec()  - action_human.header.stamp.to_sec())
            new_row = {'action_x':self.human_action, 'act_x_created': act_x_time_created, 'transmit_time_act_x':rospy.get_rostime().to_sec()  - action_human.header.stamp.to_sec()}
            self.df_timing_x_logs = self.df_timing_x_logs.append(new_row, ignore_index=True)
    
    def set_human_action_y(self,action_human):
        """
        Gets the human action from the publisher.
        """
        if action_human.action != 0.0:
            self.action_second_human = action_human.action

            act_y_time_created = action_human.header.stamp.to_sec()
            # self.act_y_time_created_list.append(act_x_time_created)
            # self.transmit_time_act_y_list.append(rospy.get_rostime().to_sec()  - action_human.header.stamp.to_sec())
            new_row = {'action_y':self.action_second_human, 'act_y_created': act_y_time_created, 'transmit_time_act_y':rospy.get_rostime().to_sec()  - action_human.header.stamp.to_sec()}
            self.df_timing_y_logs = self.df_timing_y_logs.append(new_row, ignore_index=True)

    # Experiment 1 loop
    def loop_1(self):
        # Experiment 1 loop
        flag = True
        current_timestep = 0
        running_reward = 0
        avg_length = 0

        self.best_score = -100 - 1 * self.config['Experiment']['loop_1']['max_timesteps']
        self.best_reward = self.best_score
        self.max_episodes = self.config['Experiment']['loop_1']['max_episodes']
        self.max_timesteps = self.config['Experiment']['loop_1']['max_timesteps']

        for i_episode in range(1, self.max_episodes + 1):
            observation = self.env.reset()
            timedout = False
            episode_reward = 0
            start = time.time()

            actions = [0, 0, 0, 0]  # all keys not pressed
            duration_pause = 0
            self.save_models = True
            for timestep in range(1, self.max_timesteps + 1):
                self.total_steps += 1
                current_timestep += 1

                # compute agent's action
                if not self.second_human:
                    randomness_threshold = self.config['Experiment']['loop_1']['start_training_step_on_episode']
                    randomness_critirion = i_episode
                    flag = self.compute_agent_action(observation, randomness_critirion, randomness_threshold, flag)
                # compute keyboard action
                duration_pause = self.getKeyboard(actions, duration_pause)
                # get final action pair
                action = self.get_action_pair()

                if timestep == self.max_timesteps:
                    timedout = True

                # Environment step
                observation_, reward, done = self.env.step(action, timedout,
                                                           self.config['Experiment']['loop_1']['action_duration'])
                # add experience to buffer and train
                self.save_experience_and_train(observation, reward, observation_, done)

                observation = observation_
                new_row = {'actions_x': action[0], 'actions_y': action[1], "ball_pos_x": observation[0],
                           "ball_pos_y": observation[1], "ball_vel_x": observation[2], "ball_vel_y": observation[3],
                           "tray_rot_x": observation[4], "tray_rot_y": observation[5], "tray_rot_vel_x": observation[6],
                           "tray_rot_vel_y": observation[7]}
                # append row to the dataframe
                self.df_training_logs = self.df_training_logs.append(new_row, ignore_index=True)
                # if total_steps >= start_training_step and total_steps % sac.target_update_interval == 0:
                #     sac.soft_update_target()

                running_reward += reward
                episode_reward += reward

                if done:
                    break

            end = time.time()
            if self.best_reward < episode_reward:
                self.best_reward = episode_reward
            self.duration_pause_total += duration_pause
            episode_duration = end - start - duration_pause

            self.episode_duration_list.append(episode_duration)
            self.score_history.append(episode_reward)

            log_interval = self.config['Experiment']['loop_1']['log_interval']
            avg_ep_duration = np.mean(self.episode_duration_list[-log_interval:])
            avg_score = np.mean(self.score_history[-log_interval:])

            # best score logging
            self.save_best_model(avg_score, i_episode, current_timestep)

            self.length_list.append(current_timestep)
            avg_length += current_timestep

            # off policy learning
            if not self.config['game']['test_model']:
                grad_updates_duration = self.grad_updates()
                self.grad_updates_durations.append(grad_updates_duration)

            # logging
            if not self.config['game']['test_model']:
                running_reward, avg_length = self.print_logs(i_episode, running_reward, avg_length, log_interval,
                                                             avg_ep_duration)
            current_timestep = 0
        if not self.second_human:
            self.avg_grad_updates_duration = mean(self.grad_updates_durations)


    # Experiment 2 loop
    def loop_2(self):
        # Experiment 2 loop
        flag = True
        current_timestep = 0
        observation = self.env.reset()
        timedout = False
        episode_reward = 0
        actions = [0, 0, 0, 0]  # all keys not pressed

        self.best_score = -50 - 1 * self.config['Experiment']['loop_2']['max_timesteps_per_game']
        self.best_reward = self.best_score
        self.total_timesteps = self.config['Experiment']['loop_2']['total_timesteps']
        self.max_timesteps_per_game = self.config['Experiment']['loop_2']['max_timesteps_per_game']

        avg_length = 0
        duration_pause = 0
        self.save_models = True
        self.game = 0
        running_reward = 0
        start = time.time()

        for timestep in range(1, self.total_timesteps + 1):
            self.total_steps += 1
            current_timestep += 1

            # get agent's action
            if not self.second_human:
                randomness_threshold = self.config['Experiment']['loop_2']['start_training_step_on_timestep']
                randomness_critirion = timestep
                flag = self.compute_agent_action(observation, randomness_critirion, randomness_threshold, flag)
            # compute keyboard action
            duration_pause = self.getKeyboard(actions, duration_pause)
            # get final action pair
            action = self.get_action_pair()

            if current_timestep == self.max_timesteps_per_game:
                timedout = True

            # Environment step
            observation_, reward, done = self.env.step(action, timedout,
                                                       self.config['Experiment']['loop_2']['action_duration'])
            # add experience to buffer and train
            self.save_experience_and_train(observation, reward, observation_, done)

            new_row = {'actions_x': action[0], 'actions_y': action[1], "ball_pos_x": observation[0],
                       "ball_pos_y": observation[1], "ball_vel_x": observation[2], "ball_vel_y": observation[3],
                       "tray_rot_x": observation[4], "tray_rot_y": observation[5], "tray_rot_vel_x": observation[6],
                       "tray_rot_vel_y": observation[7]}
            # append row to the dataframe
            self.df_training_logs = self.df_training_logs.append(new_row, ignore_index=True)
            observation = observation_
            # if total_steps >= start_training_step and total_steps % sac.target_update_interval == 0:
            #     sac.soft_update_target()

            # off policy learning
            if not self.config['game']['test_model']:
                grad_updates_duration = self.grad_updates()
                self.grad_updates_durations.append(grad_updates_duration)

            running_reward += reward
            episode_reward += reward

            if done:
                end = time.time()
                self.game += 1
                if self.best_reward < episode_reward:
                    self.best_reward = episode_reward
                self.duration_pause_total += duration_pause
                episode_duration = end - start - duration_pause

                self.episode_duration_list.append(episode_duration)
                self.score_history.append(episode_reward)

                log_interval = self.config['Experiment']['loop_2']['log_interval']
                avg_ep_duration = np.mean(self.episode_duration_list[-log_interval:])
                avg_score = np.mean(self.score_history[-log_interval:])

                # best score logging
                self.save_best_model(avg_score, self.game, current_timestep)

                self.length_list.append(current_timestep)
                avg_length += current_timestep

                # logging
                if not self.config['game']['test_model']:
                    running_reward, avg_length = self.print_logs(self.game, running_reward, avg_length, log_interval,
                                                                 avg_ep_duration)

                current_timestep = 0
                observation = self.env.reset()
                timedout = False
                episode_reward = 0
                actions = [0, 0, 0, 0]  # all keys not pressed
                start = time.time()

        if not self.second_human:
            self.avg_grad_updates_duration = mean(self.grad_updates_durations)

    def getKeyboard(self, actions, duration_pause):
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
        return duration_pause

    def save_info(self, chkpt_dir, experiment_duration, total_games):
        info = {}
        info['experiment_duration'] = experiment_duration
        info['best_score'] = self.best_score
        info['best_score_episode'] = self.best_score_episode
        info['best_reward'] = self.best_reward
        info['best_score_length'] = self.best_score_length
        info['total_steps'] = self.total_steps
        info['total_games'] = total_games
        info['fps'] = self.env.fps
        info['avg_grad_updates_duration'] = self.avg_grad_updates_duration
        info['avg_human_act_transmition_time'] =mean(self.human_act_transmition_time_list)
        w = csv.writer(open(chkpt_dir + '/rest_info.csv', "w"))
        for key, val in info.items():
            w.writerow([key, val])

    def get_action_pair(self):
        if self.second_human:
            action = [self.action_second_human, self.human_action]
        else:
            action = [self.agent_action, self.human_action]
        self.action_history.append(action)
        self.action_second_human, self.human_action = [0, 0]
        return action

    def save_experience_and_train(self, observation, reward, observation_, done):
        if not self.second_human:
            if self.discrete:
                self.agent.memory.add(observation, self.agent_action, reward, observation_, done)
            else:
                # observation_, reward, done = self.maze.step(action, timedout, self.config['Experiment']['loop_1']['action_duration'])
                self.agent.remember(observation, self.agent_action, reward, observation_, done)

            if not self.config['game']['test_model']:
                if self.discrete:
                    self.agent.learn()
                    self.agent.soft_update_target()
                else:
                    self.agent.learn([observation, self.agent_action, reward, observation_, done])
                    self.agent.learn([observation, self.agent_action, reward, observation_, done])

    def save_best_model(self, avg_score, game, current_timestep):
        if avg_score > self.best_score:
            self.best_score = avg_score
            self.best_score_episode = game
            self.best_score_length = current_timestep
            if not self.config['game']['test_model'] and self.save_models and not self.second_human:
                self.agent.save_models()

    def grad_updates(self):
        start_grad_updates = time.time()
        update_cycles = math.ceil(
            self.config['Experiment']['loop_2']['update_cycles'] / self.agent.batch_size)
        if not self.second_human:
            if self.total_steps % self.agent.update_interval == 0 and update_cycles > 0:
                print("Performing {} updates".format(update_cycles))
                for _ in tqdm(range(update_cycles)):
                    if self.discrete:
                        self.agent.learn()
                        self.agent.soft_update_target()
                    else:
                        self.agent.learn()

        end_grad_updates = time.time()
        return end_grad_updates - start_grad_updates

    def print_logs(self, game, running_reward, avg_length, log_interval, avg_ep_duration):
        if game % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            log_reward = int((running_reward / log_interval))

            print(
                'Episode {} \tTotal timesteps {} \t avg length: {} \t Total reward(last {} episodes): {} \t Best Score: {} \t avg '
                'episode duration: {}'.format(game, self.total_steps, avg_length,
                                              log_interval,
                                              log_reward, self.best_score,
                                              timedelta(
                                                  seconds=avg_ep_duration)))
            running_reward = 0
            avg_length = 0
        return running_reward, avg_length

    def compute_agent_action(self, observation, randomness_critirion, randomness_threshold, flag):
        if self.discrete:
            if randomness_critirion < randomness_threshold:
                # Pure exploration
                self.agent_action = np.random.randint(self.env.action_space.actions_number)
                self.save_models = False
                if flag:
                    print("Using Random Agent")
                    flag = False
            else:  # Explore with actions_prob
                self.save_models = True
                self.agent_action = self.agent.actor.sample_act(observation)
                if not flag:
                    print("Using SAC Agent")
                    flag = True
        else:
            self.save_models = True
            self.agent_action = self.agent.choose_action(observation)
        return flag
