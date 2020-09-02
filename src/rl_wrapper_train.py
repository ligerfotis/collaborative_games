#!/usr/bin/env python
import rospy
import std_msgs
from std_msgs.msg import Int16, Float32
from environment import Game 
from hand_direction.msg import observation, action_agent, reward_observation, action_msg
from std_srvs.srv import Empty,EmptyResponse, Trigger
import time
from statistics import mean, stdev
from sac import SAC
from hyperparams_ur10 import OFF_POLICY_BATCH_SIZE as BATCH_SIZE, DISCOUNT, ENTROPY_WEIGHT, HIDDEN_SIZE, LEARNING_RATE, MAX_STEPS, POLYAK_FACTOR, REPLAY_SIZE, UPDATE_INTERVAL, UPDATE_START, SAVE_INTERVAL, OFFLINE_UPDATES, TEST_NUM, ACTION_DURATION, CONTROL_MODE, ACCEL_RATE, REWARD_TYPE, INTER_NUM
import matplotlib.pyplot as plt
import numpy as np
from utils import plot, plot_hist, subplot
from tqdm import tqdm
import rospkg
import os
from collections import deque
import csv


control_mode = CONTROL_MODE

offline_updates_num = OFFLINE_UPDATES
test_num = TEST_NUM

action_duration = ACTION_DURATION

class controller:

	def __init__(self):
		# print("init")
		self.game = Game()
		self.agent = SAC()

		self.action_human = 0.0
		self.action_agent = 0.0

		self.act_human_sub = rospy.Subscriber("/rl/action_x", action_msg, self.set_human_action)

		self.turtle_accel_pub = rospy.Publisher("/rl/turtle_accel", Float32, queue_size = 10)
		self.act_agent_pub = rospy.Publisher("/rl/action_y", action_msg, queue_size = 10)
		self.reset_robot_pub = rospy.Publisher("/robot_reset", Int16, queue_size = 1)
		self.turtle_state_pub = rospy.Publisher("/rl/turtle_pos_X", Int16, queue_size = 10)

		self.transmit_time_list = []

		self.rewards_list = []
		self.turn_list = []
		self.interaction_time_list = []
		self.interaction_training_time_list = []
		self.mean_list = []
		self.stdev_list = []
		self.alpha_values = []
		self.policy_loss_list = []
		self.value_loss_list = []
		self.q_loss_list = []
		self.critics_lr_list = []
		self.value_critic_lr_list = []
		self.actor_lr_list = []
		self.trials_list = []
		self.agent_act_list = []
		self.human_act_list = []
		self.action_timesteps = []
		self.turtle_pos_x = []
		self.turtle_vel_x = []
		self.turtle_acc_x = []
		self.turtle_pos_y = []
		self.turtle_vel_y = []
		self.turtle_acc_y = []
		self.position_timesteps = []
		self.exec_time_list = []
		self.act_human_list = []
		self.real_act_list = []
		self.time_real_act_list = []
		self.time_turtle_pos = []
		self.time_turtle_vel = []
		self.time_turtle_acc = []
		self.exec_time = None

		self.interaction_counter = 0
		self.iter_num = 0

		self.run_num = 1

		self.turtle_pos_x_dict = {}

		self.fps_list = []

		x1 = np.linspace(0, np.pi, 10)
		x2 = np.linspace(0, -np.pi, 10)		
		tmp_list = []
		for _ in range(15):
			tmp_list = np.concatenate((tmp_list, np.sin(x2)/5.71), axis=None)
			tmp_list = np.concatenate((tmp_list, np.sin(x1)/5.71), axis=None)


		self.agent_act_synthetic_accel = deque(tmp_list) 
		# print(self.agent_act_synthetic_accel)
		
		rospack = rospkg.RosPack()
		package_path = rospack.get_path("hand_direction")

		self.plot_directory = package_path + "/src/plots/" + control_mode +"/"
		if not os.path.exists(self.plot_directory):
			print("Dir %s was not found. Creating it..." %(self.plot_directory))
			os.makedirs(self.plot_directory)

		exp_num = 1
		while 1:
			if os.path.exists(self.plot_directory + "Experiment_" + str(exp_num) + "/"):
				exp_num += 1
			else:
				self.plot_directory += "Experiment_" + str(exp_num) + "/"
				os.makedirs(self.plot_directory)
				break

		if not os.path.exists(self.plot_directory + 'rl_dynamics/'):
			print("Dir %s was not found. Creating it..." %(self.plot_directory + 'rl_dynamics/'))
			os.makedirs(self.plot_directory + 'rl_dynamics/')
		# if not os.path.exists(self.plot_directory + "turtle_dynamics/"):
		# 	print("Dir %s was not found. Creating it..." %(self.plot_directory + 'turtle_dynamics/'))
		# 	os.makedirs(self.plot_directory + 'turtle_dynamics/')

		

	def set_human_action(self,action_human):
		if action_human.action != 0.0:
			self.action_human = action_human.action
			self.time_real_act_list.append(rospy.get_rostime().to_sec())
			self.real_act_list.append(self.action_human)

			self.transmit_time_list.append(rospy.get_rostime().to_sec()  - action_human.header.stamp.to_sec())


	def game_loop(self):
		first_update = True
		global_time = rospy.get_rostime().to_sec()

		# # run trials
		self.reset_lists()
		score_list =  self.test()

		self.mean_list.append(mean(score_list))
		self.stdev_list.append(stdev(score_list))
		self.trials_list.append(score_list)

		self.resetGame()
		self.interaction_counter = 0

		for exp in range(MAX_STEPS+1):

			self.check_goal_reached()

			if self.game.running:
				self.interaction_counter += 1

				start_interaction_time = time.time()

				self.game.experiment = exp
				self.turns += 1

				state = self.getState()
				agent_act = self.agent.next_action(state)

				self.turtle_state_pub.publish(state[2])
				self.publish_agent_action(agent_act)

				tmp_time = time.time()
				act_human = self.action_human # self.action_human is changing while the loop is running

				count  = 0
				while ((time.time() - tmp_time) < action_duration) and self.game.running:
					count += 1
					# control mode is "accel" or "vel"
					self.exec_time = self.game.play([act_human, agent_act.item()],control_mode=control_mode)
					# exec_time = self.game.play([act_human, 0],control_mode=control_mode)
					
					self.agent_act_list.append(agent_act)
					self.human_act_list.append(act_human)
					self.action_timesteps.append(tmp_time)

					self.save_stats()
					self.check_goal_reached()
					
				# print count
					
				reward = self.getReward()
				next_state = self.getState()
				done = self.game.finished
				episode = [state, reward, agent_act, next_state, done]

				self.agent.update_rw_state(episode)

				self.total_reward_per_game += reward 

				self.interaction_time_list.append(time.time() - start_interaction_time)

				# when replay buffer has enough samples update gradient at every turn
				if first_update and len(self.agent.D) >= BATCH_SIZE and exp > UPDATE_START:

					if first_update:
						print("\nStarting updates")
						first_update = False

					[alpha, policy_loss, value_loss, q_loss, critics_lr, value_critic_lr, actor_lr] = self.agent.train(sample=episode)
					self.save_rl_data(alpha, policy_loss, value_loss, q_loss, critics_lr, value_critic_lr, actor_lr)

					self.interaction_training_time_list.append(time.time() - start_interaction_time)

				# run "offline_updates_num" offline gradient updates every "UPDATE_INTERVAL" steps
				elif not first_update and len(self.agent.D) >= BATCH_SIZE and exp % UPDATE_INTERVAL == 0:

					print(str(offline_updates_num) + " Gradient upadates")
					self.game.waitScreen("Training... Please Wait.")

					pbar = tqdm(xrange(1, offline_updates_num + 1), unit_scale=1, smoothing=0)
					for _ in pbar:
						[alpha, policy_loss, value_loss, q_loss, critics_lr, value_critic_lr, actor_lr] = self.agent.train(verbose=False)
						self.interaction_training_time_list.append(time.time() - start_interaction_time)

						self.save_rl_data(alpha, policy_loss, value_loss, q_loss, critics_lr, value_critic_lr, actor_lr)

					# # run trials
					self.reset_lists()
					score_list =  self.test()

					self.mean_list.append(mean(score_list))
					self.stdev_list.append(stdev(score_list))
					self.trials_list.append(score_list)

					self.resetGame()
			else:
				self.save_and_plot_stats_environment(self.run_num)
				self.run_num += 1
				self.turn_list.append(self.turns)
				self.rewards_list.append(self.total_reward_per_game)
				
				# reset game
				self.resetGame()
		
		self.save_and_plot_stats_rl()
		
		print("Average Human Action Transmission time per interaction: %f milliseconds(stdev: %f). \n" % (mean(self.transmit_time_list) * 1e3, stdev(self.transmit_time_list) * 1e3))
		print("Average Execution time per interaction: %f milliseconds(stdev: %f). \n" % (mean(self.interaction_time_list) * 1e3, stdev(self.interaction_time_list) * 1e3))
		print("Average Execution time per interaction and online update: %f milliseconds(stdev: %f). \n" % (mean(self.interaction_training_time_list) * 1e3, stdev(self.interaction_training_time_list) * 1e3))

		print("Total time of experiments is: %d minutes and %d seconds.\n" % ( ( rospy.get_rostime().to_sec() - global_time )/60, ( rospy.get_rostime().to_sec() - global_time )%60 )  )
		
		self.game.endGame()

	def test(self):

		score_list = []
		self.iter_num += 1
		for game in range(test_num):
			score = INTER_NUM

			self.resetGame("Testing Model. Trial %d of %d." % (game+1,test_num))
			self.interaction_counter = 0
			while self.game.running:
				self.interaction_counter += 1
				self.game.experiment = "Test: " + str(game+1)

				state = self.getState()
				agent_act = self.agent.next_action(state, stochastic=False) # take only the mean
				act_human = self.action_human

				tmp_time = time.time()
				self.agent_act_list.append(agent_act)
				self.human_act_list.append(act_human)
				self.action_timesteps.append(tmp_time)
				# print(agent_act)
				self.publish_agent_action(agent_act)

				while ((time.time() - tmp_time) < action_duration) and self.game.running:
					self.exec_time = self.game.play([act_human, agent_act.item()], total_games=test_num, control_mode=control_mode)
					self.save_stats()
					self.check_goal_reached()

				score -= 1
				self.check_goal_reached()
				

			score_list.append(score)
			self.save_and_plot_stats_environment("Test_"+ str(self.iter_num) + "_" + str(game))


		return score_list

	def save_stats(self):
		self.turtle_accel_pub.publish(self.game.accel_x)

		self.turtle_pos_x.append(self.game.real_turtle_pos[0])
		self.turtle_pos_y.append(self.game.real_turtle_pos[1])
		self.time_turtle_pos.append(rospy.get_rostime().to_sec())

		self.turtle_vel_x.append(self.game.vel_x)
		self.turtle_vel_y.append(self.game.vel_y)
		self.time_turtle_vel.append(rospy.get_rostime().to_sec())

		self.turtle_acc_x.append(self.game.accel_x)
		self.turtle_acc_y.append(self.game.accel_y)
		self.time_turtle_acc.append(rospy.get_rostime().to_sec())

		self.fps_list.append(self.game.current_fps)

		self.turtle_pos_x_dict[self.game.turtle_pos[0]] = self.exec_time

		self.exec_time_list.append(self.exec_time)

	def getReward(self):
		if self.game.finished:
			if self.game.timedOut and REWARD_TYPE=="penalty":
				return -50
			else:
				return 100
		else:
			return -1

	def getState(self):
		return [self.game.accel_x, self.game.accel_y, self.game.turtle_pos[0], self.game.turtle_pos[1], self.game.vel_x, self.game.vel_y]

	def check_goal_reached(self, name="turtle"):
		if name is "turtle":
			if self.game.time_dependend and (self.interaction_counter % INTER_NUM == 0) and self.interaction_counter != 0:
				self.game.running = 0
				self.game.exitcode = 1
				self.game.timedOut = True
				self.game.finished = True
				self.interaction_counter = 0

			# if self.game.width - 40 > self.game.turtle_pos[0] > self.game.width - (80 + 40) \
			# and 20 < self.game.turtle_pos[1] < (80 + 60 / 2 - 32):

			if (self.game.width - 160) <= self.game.turtle_pos[0] <= (self.game.width - 60 - 64) and 40 <= self.game.turtle_pos[1] <= (140 - 64):
				self.game.running = 0
				self.game.exitcode = 1
				self.game.finished = True  # This means final state achieved
				self.interaction_counter = 0


	def resetGame(self, msg=None):
		wait_time = 3
		# self.reset_robot_pub.publish(1)
		self.game.waitScreen(msg1="Put Right Wrist on starting point.", msg2=msg, duration=wait_time)
		self.game = Game()
		self.TIME = ACTION_DURATION * INTER_NUM
		self.action_human = 0.0
		self.game.start_time = time.time()
		self.total_reward_per_game = 0
		self.turns = 0
		# self.reset_robot_pub.publish(1)

	def save_and_plot_stats_rl(self):
		np.savetxt(self.plot_directory + 'rl_dynamics/' + 'alpha_values.csv', self.alpha_values, delimiter=',', fmt='%f')
		np.savetxt(self.plot_directory + 'rl_dynamics/' + 'policy_loss.csv', self.policy_loss_list, delimiter=',', fmt='%f')
		np.savetxt(self.plot_directory + 'rl_dynamics/' + 'value_loss.csv', self.value_loss_list, delimiter=',', fmt='%f')
		np.savetxt(self.plot_directory + 'rl_dynamics/' + 'q_loss.csv', self.q_loss_list, delimiter=',', fmt='%f')

		np.savetxt(self.plot_directory + 'rl_dynamics/' + 'rewards_list.csv', self.rewards_list, delimiter=',', fmt='%f')
		np.savetxt(self.plot_directory + 'rl_dynamics/' + 'turn_list.csv', self.turn_list, delimiter=',', fmt='%f')
		np.savetxt(self.plot_directory + 'rl_dynamics/' + 'means.csv', self.mean_list, delimiter=',', fmt='%f')		
		np.savetxt(self.plot_directory + 'rl_dynamics/' + 'stdev.csv', self.stdev_list, delimiter=',', fmt='%f')
		np.savetxt(self.plot_directory + 'rl_dynamics/' + 'critics_lr_list.csv', self.critics_lr_list, delimiter=',', fmt='%f')
		np.savetxt(self.plot_directory + 'rl_dynamics/' + 'value_critic_lr_list.csv', self.value_critic_lr_list, delimiter=',', fmt='%f')
		np.savetxt(self.plot_directory + 'rl_dynamics/' + 'actor_lr_list.csv', self.actor_lr_list, delimiter=',', fmt='%f')
		np.savetxt(self.plot_directory + 'rl_dynamics/' + 'trials_list.csv', self.trials_list, delimiter=',', fmt='%f')

		np.savetxt(self.plot_directory + 'fps_list.csv', self.fps_list, delimiter=',', fmt='%f')
		plot_hist(self.fps_list, self.plot_directory, 'fps_list_' + str(self.game.fps))
		
		np.savetxt(self.plot_directory + 'exec_time_list.csv', self.exec_time_list, delimiter=',', fmt='%f')

		plot(range(len(self.alpha_values)), self.alpha_values, "alpha_values", 'Alpha Value', 'Number of Gradient Updates', self.plot_directory, save=True)
		plot(range(len(self.policy_loss_list)), self.policy_loss_list, "policy_loss", 'Policy loss', 'Number of Gradient Updates', self.plot_directory, save=True)
		plot(range(len(self.value_loss_list)), self.value_loss_list, "value_loss_list", 'Value loss', 'Number of Gradient Updates', self.plot_directory, save=True)
		plot(range(len(self.rewards_list)), self.rewards_list, "Rewards_per_game", 'Total Rewards per Game', 'Number of Games', self.plot_directory, save=True)
		plot(range(len(self.turn_list)), self.turn_list, "Steps_per_game", 'Turns per Game', 'Number of Games', self.plot_directory, save=True)	

		plot(range(len(self.critics_lr_list)), self.critics_lr_list, "critics_lr_list", 'Critic lr', 'Number of Gradient Updates', self.plot_directory, save=True)
		plot(range(len(self.value_critic_lr_list)), self.value_critic_lr_list, "value_critic_lr_list", 'Value lr', 'Number of Gradient Updates', self.plot_directory, save=True)
		plot(range(len(self.actor_lr_list)), self.actor_lr_list, "actor_lr_list", 'Actor lr', 'Number of Gradient Updates', self.plot_directory, save=True)	

		plot(range(UPDATE_INTERVAL, MAX_STEPS + UPDATE_INTERVAL, UPDATE_INTERVAL), self.mean_list, "trials", 'Tests Score', 'Number of Interactions', self.plot_directory, save=True, variance=True, stdev=self.stdev_list)		


	def save_and_plot_stats_environment(self, run_num):

		run_subfolder = "game_" + str(run_num) + "/"
		os.makedirs(self.plot_directory + run_subfolder + "turtle_dynamics/")

		self.exp_details()

		np.savetxt(self.plot_directory + run_subfolder + "turtle_dynamics/"+ 'agent_act_list.csv', self.agent_act_list, delimiter=',', fmt='%f')
		np.savetxt(self.plot_directory + run_subfolder + "turtle_dynamics/"+ 'human_act_list.csv', self.human_act_list, delimiter=',', fmt='%f')
		np.savetxt(self.plot_directory + run_subfolder + "turtle_dynamics/"+ 'action_timesteps.csv', self.action_timesteps, delimiter=',', fmt='%f')
		human_act_list = np.genfromtxt(self.plot_directory + run_subfolder + "turtle_dynamics/"+ 'human_act_list.csv', delimiter=',')
		agent_act_list = np.genfromtxt(self.plot_directory + run_subfolder + "turtle_dynamics/"+ 'agent_act_list.csv', delimiter=',')
		plot_hist(agent_act_list, self.plot_directory + run_subfolder + "turtle_dynamics/"+ 'agent_act_hist_' "game_" + str(run_num), 'Agent Action Histogram')
		plot_hist(human_act_list, self.plot_directory + run_subfolder + "turtle_dynamics/"+ 'human_act_hist_' "game_" + str(run_num), 'Human Action Histogram')

		np.savetxt(self.plot_directory + run_subfolder + "turtle_dynamics/" + 'turtle_pos_x.csv', self.turtle_pos_x, delimiter=',', fmt='%f')
		np.savetxt(self.plot_directory + run_subfolder + "turtle_dynamics/" + 'turtle_pos_y.csv', self.turtle_pos_y, delimiter=',', fmt='%f')
		np.savetxt(self.plot_directory + run_subfolder + "turtle_dynamics/" + 'time_turtle_pos.csv', self.time_turtle_pos, delimiter=',', fmt='%f')
		
		np.savetxt(self.plot_directory + run_subfolder + "turtle_dynamics/" + 'turtle_vel_x.csv', self.turtle_vel_x, delimiter=',', fmt='%f')
		np.savetxt(self.plot_directory + run_subfolder + "turtle_dynamics/" + 'turtle_vel_y.csv', self.turtle_vel_y, delimiter=',', fmt='%f')
		np.savetxt(self.plot_directory + run_subfolder + "turtle_dynamics/" + 'time_turtle_vel.csv', self.time_turtle_vel, delimiter=',', fmt='%f')
		
		np.savetxt(self.plot_directory + run_subfolder + "turtle_dynamics/" + 'turtle_accel_x.csv', self.turtle_acc_x, delimiter=',', fmt='%f')
		np.savetxt(self.plot_directory + run_subfolder + "turtle_dynamics/" + 'turtle_accel_y.csv', self.turtle_acc_y, delimiter=',', fmt='%f')
		np.savetxt(self.plot_directory + run_subfolder + "turtle_dynamics/" + 'time_turtle_acc.csv', self.time_turtle_acc, delimiter=',', fmt='%f')
		
		np.savetxt(self.plot_directory + run_subfolder + "turtle_dynamics/" + 'real_act_list.csv', self.real_act_list, delimiter=',', fmt='%f')
		np.savetxt(self.plot_directory + run_subfolder + "turtle_dynamics/" + 'time_real_act_list.csv', self.time_real_act_list, delimiter=',', fmt='%f')

		subplot(self.plot_directory + run_subfolder + "turtle_dynamics/", self.turtle_pos_x, self.turtle_vel_x, self.turtle_acc_x, self.time_turtle_pos, self.time_turtle_vel, self.time_turtle_acc, human_act_list, self.action_timesteps, "x", control_mode)
		subplot(self.plot_directory + run_subfolder + "turtle_dynamics/", self.turtle_pos_y, self.turtle_vel_y, self.turtle_acc_y, self.time_turtle_pos, self.time_turtle_vel, self.time_turtle_acc, agent_act_list, self.action_timesteps, "y", control_mode)
		
		self.reset_lists()

	def publish_agent_action(self, agent_act):
		h = std_msgs.msg.Header()
		h.stamp = rospy.Time.now() 

		act = action_msg()
		act.action = agent_act
		act.header = h
		
		self.act_agent_pub.publish(act)

	def reset_lists(self):
		self.agent_act_list = []
		self.human_act_list = []
		self.action_timesteps = []
		self.turtle_pos_x = []
		self.turtle_pos_y = []
		self.time_turtle_pos = []
		self.turtle_vel_x = []
		self.turtle_vel_y = []
		self.time_turtle_vel = []
		self.turtle_acc_x = []
		self.turtle_acc_y = []
		self.time_turtle_acc = []
		self.real_act_list = []
		self.time_real_act_list = []	

	def save_rl_data(self, alpha, policy_loss, value_loss, q_loss, critics_lr, value_critic_lr, actor_lr):
		self.alpha_values.append(alpha.item())
		self.policy_loss_list.append(policy_loss.item())
		self.value_loss_list.append(value_loss.item())
		self.q_loss_list.append(q_loss.item())
		
		self.critics_lr_list.append(critics_lr)
		self.value_critic_lr_list.append(value_critic_lr)
		self.actor_lr_list.append(actor_lr)

	def exp_details(self):
		exp_details = {}
		exp_details["MAX_STEPS"] = MAX_STEPS
		exp_details["CONTROL_MODE"] = CONTROL_MODE
		exp_details["ACCEL_RATE"] = ACCEL_RATE
		exp_details["ACTION_DURATION"] = ACTION_DURATION
		exp_details["BATCH_SIZE"] = BATCH_SIZE
		exp_details["REPLAY_SIZE"] = REPLAY_SIZE
		exp_details["OFFLINE_UPDATES"] = OFFLINE_UPDATES
		exp_details["UPDATE_START"] = UPDATE_START


		csv_file = "exp_details.csv"
		try:
			w = csv.writer(open(self.plot_directory+csv_file, "w"))
			for key, val in exp_details.items():
				w.writerow([key, val])
		except IOError:
			print("I/O error")



if __name__ == '__main__':
	rospy.init_node('rl_wrapper', anonymous=True)
	ctrl = controller()
	ctrl.game.game_intro()

	ctrl.game_loop()
	# while ctrl.game.running:
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")


