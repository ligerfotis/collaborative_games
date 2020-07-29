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
from hyperparams_ur10 import OFF_POLICY_BATCH_SIZE as BATCH_SIZE, DISCOUNT, ENTROPY_WEIGHT, HIDDEN_SIZE, LEARNING_RATE, MAX_STEPS, POLYAK_FACTOR, REPLAY_SIZE, UPDATE_INTERVAL, UPDATE_START, SAVE_INTERVAL, OFFLINE_UPDATES, TEST_NUM, ACTION_DURATION
import matplotlib.pyplot as plt
import numpy as np
from utils import plot
from tqdm import tqdm
import rospkg
import os

# path = "/home/fligerakis/catkin_ws/src/hand_direction/plots/"
rospack = rospkg.RosPack()
package_path = rospack.get_path("hand_direction")

offline_updates_num = OFFLINE_UPDATES
test_num = TEST_NUM

action_duration = ACTION_DURATION # 200 milliseconds

class controller:

	def __init__(self):
		# print("init")
		self.game = Game()
		self.agent = SAC()

		self.action_human = 0.0
		self.action_agent = 0.0

		self.act_human_sub = rospy.Subscriber("/rl/action_x", action_msg, self.set_human_action)

		self.act_agent_pub = rospy.Publisher('/rl/action_y', action_msg, queue_size = 10)

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
		self.critics_lr_list = []
		self.value_critic_lr_list = []
		self.actor_lr_list = []
		self.trials_list = []

		self.plot_directory = package_path + "/src/plots/"
		if not os.path.exists(self.plot_directory):
			print("Dir %s was not found. Creating it..." %(self.plot_directory))
			os.makedirs(self.plot_directory)

	def set_human_action(self,action_human):
		if action_agent.action != 0.0:
			self.action_human = action_human.action
			self.transmit_time_list.append(rospy.get_rostime().to_sec()  - action_human.header.stamp.to_sec())


	def game_loop(self):
		first_update = True
		global_time = rospy.get_rostime().to_sec()

		self.resetGame()

		for exp in range(MAX_STEPS+1):
			self.check_goal_reached()

			if self.game.running:
				start_interaction_time = time.time()

				self.game.experiment = exp
				self.turns += 1

				state = self.getState()
				agent_act = self.agent.next_action(state)

				self.publish_agent_action(agent_act)

				tmp_time = time.time()
				act_human = self.action_human

				while time.time() - tmp_time < action_duration :
					exec_time = self.game.play([act_human, agent_act.item()])
					
				reward = self.getReward()
				next_state = self.getState()
				done = self.game.finished
				episode = [state, reward, agent_act, next_state, done]

				self.agent.update_rw_state(episode)

				self.total_reward_per_game += reward 

				self.interaction_time_list.append(time.time() - start_interaction_time)

				# when replay buffer has enough samples update gradient at every turn
				if len(self.agent.D) >= BATCH_SIZE:

					if first_update:
						print("\nStarting updates")
						first_update = False

					[alpha, policy_loss, value_loss, critics_lr, value_critic_lr, actor_lr] = self.agent.train(sample=episode)
					self.alpha_values.append(alpha.item())
					self.policy_loss_list.append(policy_loss.item())
					self.value_loss_list.append(value_loss.item())
					self.critics_lr_list.append(critics_lr)
					self.value_critic_lr_list.append(value_critic_lr)
					self.actor_lr_list.append(actor_lr)

					self.interaction_training_time_list.append(time.time() - start_interaction_time)

				# run "offline_updates_num" offline gradient updates every "UPDATE_INTERVAL" steps
				if len(self.agent.D) >= BATCH_SIZE and exp % UPDATE_INTERVAL == 0:

					print(str(offline_updates_num) + " Gradient upadates")
					self.game.waitScreen("Training... Please Wait.")

					pbar = tqdm(xrange(1, offline_updates_num + 1), unit_scale=1, smoothing=0)
					for _ in pbar:
						[alpha, policy_loss, value_loss, critics_lr, value_critic_lr, actor_lr] = self.agent.train(verbose=False)
						self.alpha_values.append(alpha.item())
						self.policy_loss_list.append(policy_loss.item())
						self.value_loss_list.append(value_loss.item())
						self.critics_lr_list.append(critics_lr)
						self.value_critic_lr_list.append(value_critic_lr)
						self.actor_lr_list.append(actor_lr)

					# run trials
					score_list =  self.test()

					self.mean_list.append(mean(score_list))
					self.stdev_list.append(stdev(score_list))
					self.trials_list.append(score_list)

					self.resetGame()
			else:
				self.turn_list.append(self.turns)
				self.rewards_list.append(self.total_reward_per_game)
				
				# reset game
				self.resetGame()

		self.save_and_plot_stats()
		
		print("Average Human Action Transmission time per interaction: %f milliseconds(stdev: %f). \n" % (mean(self.transmit_time_list) * 1e3, stdev(self.transmit_time_list) * 1e3))
		print("Average Execution time per interaction: %f milliseconds(stdev: %f). \n" % (mean(self.interaction_time_list) * 1e3, stdev(self.interaction_time_list) * 1e3))
		print("Average Execution time per interaction and online update: %f milliseconds(stdev: %f). \n" % (mean(self.interaction_training_time_list) * 1e3, stdev(self.interaction_training_time_list) * 1e3))

		print("Total time of experiments is: %d minutes and %d seconds.\n" % ( ( rospy.get_rostime().to_sec() - global_time )/60, ( rospy.get_rostime().to_sec() - global_time )%60 )  )
		
		self.game.endGame()

	def test(self):

		score_list = []
		for game in range(test_num):
			score = 200

			self.resetGame("Testing Model. Trial %d of %d." % (game+1,test_num))

			while self.game.running:
				self.game.experiment = "Test: " + str(game+1)

				state = self.getState()
				agent_act = self.agent.next_action(state, stochastic=False) # take only the mean
				# print(agent_act)
				tmp_time = time.time()
				while time.time() - tmp_time < 0.2 :
					exec_time = self.game.play([self.action_human, agent_act.item()], total_games=test_num)

				score -= 1
				self.check_goal_reached()

			score_list.append(score)


		return score_list

	def getReward(self):
		if self.game.finished:
			return 10
		else:
			return -1

	def getState(self):
		return [self.game.accel_x, self.game.accel_y, self.game.turtle_pos[0], self.game.turtle_pos[1], self.game.vel_x, self.game.vel_y]

	def check_goal_reached(self, name="turtle"):
		if name is "turtle":
			if self.game.time_dependend and self.game.time_elapsed >= self.game.TIME:
				self.game.running = 0
				self.game.exitcode = 1
				self.game.timedOut = True
				self.game.finished = True

			# if self.game.width - 40 > self.game.turtle_pos[0] > self.game.width - (80 + 40) \
			# and 20 < self.game.turtle_pos[1] < (80 + 60 / 2 - 32):

			if self.game.width - 160 < self.game.turtle_pos[0] < self.game.	width - 60 - 64 and 40 < self.game.turtle_pos[1] < 140 -64:
				self.game.running = 0
				self.game.exitcode = 1
				self.game.finished = True  # This means final state achieved


	def resetGame(self, msg=None):
		wait_time = 3
		self.game.waitScreen(msg1="Put Right Wrist on starting point.", msg2=msg, duration=wait_time)
		self.game = Game()
		self.action_human = 0.0
		self.game.start_time = time.time()
		self.total_reward_per_game = 0
		self.turns = 0

	def save_and_plot_stats(self):
		np.savetxt(self.plot_directory + 'alpha_values.csv', self.alpha_values, delimiter=',', fmt='%f')
		np.savetxt(self.plot_directory + 'policy_loss.csv', self.policy_loss_list, delimiter=',', fmt='%f')
		np.savetxt(self.plot_directory + 'value_loss.csv', self.value_loss_list, delimiter=',', fmt='%f')
		np.savetxt(self.plot_directory + 'rewards_list.csv', self.rewards_list, delimiter=',', fmt='%f')
		np.savetxt(self.plot_directory + 'turn_list.csv', self.turn_list, delimiter=',', fmt='%f')
		np.savetxt(self.plot_directory + 'means.csv', self.mean_list, delimiter=',', fmt='%f')		
		np.savetxt(self.plot_directory + 'stdev.csv', self.stdev_list, delimiter=',', fmt='%f')
		np.savetxt(self.plot_directory + 'critics_lr_list.csv', self.critics_lr_list, delimiter=',', fmt='%f')
		np.savetxt(self.plot_directory + 'value_critic_lr_list.csv', self.value_critic_lr_list, delimiter=',', fmt='%f')
		np.savetxt(self.plot_directory + 'actor_lr_list.csv', self.actor_lr_list, delimiter=',', fmt='%f')
		np.savetxt(self.plot_directory + 'trials_list.csv', self.trials_list, delimiter=',', fmt='%f')



		plot(range(len(self.alpha_values)), self.alpha_values, "alpha_values", 'Alpha Value', 'Number of Gradient Updates', self.plot_directory, save=True)
		plot(range(len(self.policy_loss_list)), self.policy_loss_list, "policy_loss", 'Policy loss', 'Number of Gradient Updates', self.plot_directory, save=True)
		plot(range(len(self.value_loss_list)), self.value_loss_list, "value_loss_list", 'Value loss', 'Number of Gradient Updates', self.plot_directory, save=True)
		plot(range(len(self.rewards_list)), self.rewards_list, "Rewards_per_game", 'Total Rewards per Game', 'Number of Games', self.plot_directory, save=True)
		plot(range(len(self.turn_list)), self.turn_list, "Steps_per_game", 'Turns per Game', 'Number of Games', self.plot_directory, save=True)	

		plot(range(len(self.critics_lr_list)), self.critics_lr_list, "critics_lr_list", 'Critic lr', 'Number of Gradient Updates', self.plot_directory, save=True)
		plot(range(len(self.value_critic_lr_list)), self.value_critic_lr_list, "value_critic_lr_list", 'Value lr', 'Number of Gradient Updates', self.plot_directory, save=True)
		plot(range(len(self.actor_lr_list)), self.actor_lr_list, "actor_lr_list", 'Actor lr', 'Number of Gradient Updates', self.plot_directory, save=True)		

		plot(range(UPDATE_INTERVAL, MAX_STEPS+UPDATE_INTERVAL, UPDATE_INTERVAL), self.mean_list, "trials", 'Tests Score', 'Number of Interactions', self.plot_directory, save=True, variance=True, stdev=self.stdev_list)		

	def publish_agent_action(self, agent_act):
		h = std_msgs.msg.Header()
		h.stamp = rospy.Time.now() 

		act = action_msg()
		act.action = agent_act
		act.header = h
		
		self.act_agent_pub.publish(act)

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


