#!/usr/bin/env python
import rospy
import gym
import std_msgs
from std_msgs.msg import Int16, Float32
from environment import Game 
from keypoint_3d_matching_msgs.msg import Keypoint3d_list
from hand_direction.msg import observation, action_agent, reward_observation, action_msg
from std_srvs.srv import Empty,EmptyResponse, Trigger
import time
from statistics import mean, stdev
from sac import SAC
from hyperparams_ur10 import OFF_POLICY_BATCH_SIZE as BATCH_SIZE, DISCOUNT, ENTROPY_WEIGHT, HIDDEN_SIZE, LEARNING_RATE, MAX_STEPS, POLYAK_FACTOR, REPLAY_SIZE, UPDATE_INTERVAL, UPDATE_START, SAVE_INTERVAL
import matplotlib.pyplot as plt
import numpy as np
from utils import plot
from tqdm import tqdm
import rospkg
import os

# path = "/home/fligerakis/catkin_ws/src/hand_direction/plots/"
rospack = rospkg.RosPack()
package_path = rospack.get_path("hand_direction")

offline_updates_num = 20000
test_num = 10

action_duration = 0.2 # 200 milliseconds

class controller:

	def __init__(self):
		# print("init")
		self.game = Game()
		self.agent = SAC()

		self.action_human = 0.0
		self.action_agent = 0.0

		self.act_human_sub = rospy.Subscriber("/rl/action_x", action_msg, self.set_human_agent)

		self.transmit_time_list = []

		self.plot_directory = package_path + "/src/plots/"
		if not os.path.exists(self.plot_directory):
			print("Dir %s was not found. Creating it..." %(self.plot_directory))
			os.makedirs(self.plot_directory)

	def set_human_agent(self,action_agent):
		if action_agent.action != 0.0:
			self.action_human = action_agent.action
			self.transmit_time_list.append(rospy.get_rostime().to_sec()  - action_agent.header.stamp.to_sec())


	def game_loop(self):
		first_update = True
		rewards_list = []
		turn_list = []
		interaction_time_list = []
		interaction_training_time_list = []
		mean_list = []
		stdev_list = []
		global_time = rospy.get_rostime().to_sec()

		self.resetGame()

		for exp in range(MAX_STEPS+1):

			if self.game.running:
				start_interaction_time = time.time()

				self.game.experiment = expa
				self.turns += 1

				state = self.game.getState()
				agent_act = self.agent.next_action(state)

				tmp_time = time.time()
				act_human = self.action_human

				while time.time() - tmp_time < action_duration :
					exec_time = self.game.play([act_human, agent_act.item()])
					
				reward = self.game.getReward()
				next_state = self.game.getState()
				done = self.game.finished
				# episode = state, reward, agent_act, next_state, done

				self.agent.update_rw_state(state, reward, agent_act, next_state, done)
				self.total_reward_per_game += reward 

				interaction_time_list.append(time.time() - start_interaction_time)

				# when replay buffer has enough samples update gradient at every turn
				if len(self.agent.D) >= BATCH_SIZE:

					if first_update:
						print("\nStarting updates")
						first_update = False

					self.agent.train()

					interaction_training_time_list.append(time.time() - start_interaction_time)

				# run "offline_updates_num" offline gradient updates every "UPDATE_INTERVAL" steps
				if len(self.agent.D) >= BATCH_SIZE and exp % UPDATE_INTERVAL == 0:

					print(str(offline_updates_num) + " Gradient upadates")
					self.game.waitScreen("Training... Please Wait.")

					pbar = tqdm(xrange(1, offline_updates_num + 1), unit_scale=1, smoothing=0)
					for _ in pbar:
						self.agent.train(verbose=False)

					# run trials
					mean_score, stdev_score =  self.test()

					mean_list.append(mean_score)
					stdev_list.append(stdev_score)

					self.resetGame()
			else:
				turn_list.append(self.turns)
				rewards_list.append(self.total_reward_per_game)
				
				# reset game
				self.resetGame()

		

		plot(range(len(rewards_list)), rewards_list, "Rewards_per_game", 'Total Rewards per Game', 'Number of Games', self.plot_directory, save=True)
		plot(range(len(turn_list)), turn_list, "Steps_per_game", 'Steps per Game', 'Number of Games', self.plot_directory, save=True)		

		print(mean_list)
		print(stdev_list)
		plt.plot(range(0,MAX_STEPS, UPDATE_INTERVAL), mean_list, 'k')
		plt.fill_between(range(0,MAX_STEPS, UPDATE_INTERVAL), np.array(mean_list) - np.array(stdev_list), np.array(mean_list) + np.array(stdev_list))
		plt.savefig( self.plot_directory + "trials")
		plt.show()
		

		print("Average Execution time per interaction: %f milliseconds(stdev: %f). \n" % (mean(interaction_time_list) * 1e3, stdev(interaction_time_list) * 1e3))
		print("Average Execution time per interaction and online update: %f milliseconds(stdev: %f). \n" % (mean(interaction_training_time_list) * 1e3, stdev(interaction_training_time_list) * 1e3))

		print("Total time of experiments is: %d minutes and %d seconds.\n" % ( ( rospy.get_rostime().to_sec() - global_time )/60, ( rospy.get_rostime().to_sec() - global_time )%60 )  )
		
		self.game.endGame()

	def test(self):

		score_list = []
		for game in range(test_num):
			score = 200

			self.resetGame("Testing Model. Trial %d of %d." % (game+1,test_num))

			while self.game.running:
				self.game.experiment = "Test: " + str(game+1)

				state = self.game.getState()
				agent_act = self.agent.next_action(state, stochastic=False) # take only the mean
				# print(agent_act)
				tmp_time = time.time()
				while time.time() - tmp_time < 0.2 :
					exec_time = self.game.play([self.action_human, agent_act.item()], total_games=test_num)

				score -= 1

			score_list.append(score)

		return [mean(score_list), stdev(score_list)]

	def resetGame(self, msg=None):
		wait_time = 3
		self.game.waitScreen(msg1="Put Right Wrist on starting point.", msg2=msg, duration=wait_time)
		self.game = Game()
		self.action_human = 0.0
		self.game.start_time = time.time()
		self.total_reward_per_game = 0
		self.turns = 0



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