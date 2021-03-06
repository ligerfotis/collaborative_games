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

from utils import plot
class controller:

	def __init__(self):
		print("init")
		self.game = Game()
		self.agent = SAC()

		self.action_human = 0.0
		self.action_agent = 0.0

		self.act_human_sub = rospy.Subscriber("/rl/action_x", action_msg, self.set_human_agent)

		self.transmit_time_list = []

	def set_human_agent(self,action_agent):
		self.action_human = action_agent.action
		self.transmit_time_list.append(rospy.get_rostime().to_sec()  - action_agent.header.stamp.to_sec())


	def game_loop(self):
		score = 200

		for game in range(10):
			# print("Interaction %d" % (exp + 1))
			
			turns = 0
			if self.game.running:
				self.game.experiment = exp
				turns += 1
				start_time = time.time()
				state = self.game.getState()
				agent_act = self.agent.next_action(state)
				tmp_time = time.time()
				while time.time() - tmp_time < 0.2 :
					exec_time = self.game.play([self.action_human, agent_act.item()])


				reward = self.game.getReward()
				next_state = self.game.getState()
				done = self.game.finished

				self.agent.update_rw_state(state, reward, agent_act, next_state, done)
				total_rewards.append(reward)

				# print(len(self.agent.D))
				if len(self.agent.D) >= UPDATE_START and exp % UPDATE_INTERVAL == 0:
					if first_update:
						print("\nStarting updates")
						first_update = False

					print("Training")

					self.agent.train()
					lr_list.append(self.agent.lr)
					# print("%d milliseconds per cycle." % ((time.time() - start_time) * 1000))
				total_time.append(time.time() - start_time)
			else:
				turn_list.append(turns)
				avg_rewards.append(mean(total_rewards))
				total_rewards = []
				# print("Average time from publishing to receiveing is %f milliseconds. \n" % (mean(self.transmit_time_list)* 1e3))
				# self.transmit_time_list = []
				# reset game
				self.game = Game()
				self.game.start_time = time.time()

		plot(range(len(avg_rewards)), avg_rewards, "Average_Reward_per_Turn", 'Average Reward per Turn', 'Experiments Number', "/home/liger/catkin_ws/src/hand_direction/plots/", save=True)
		plot(range(len(total_time)), total_time, "Duration_per_turn", 'Duration_per_turn', 'Training steps', "/home/liger/catkin_ws/src/hand_direction/plots/", save=True)
		plot(range(len(turn_list)), turn_list, "Steps_per_turn", 'Steps per Turn', 'Experiments Number', "/home/liger/catkin_ws/src/hand_direction/plots/", save=True)


		print("Average Execution time for play funtion is %f milliseconds(stdev: %f). \n" % (mean(total_time) * 1e3, stdev(total_time) * 1e3))

		print("Total time of experiments is: %d minutes and %d seconds.\n" % ( ( rospy.get_rostime().to_sec() - global_time )/60, ( rospy.get_rostime().to_sec() - global_time )%60 )  )
		
		self.game.endGame()


		# print("Average Execution time for play funtion is %f milliseconds. \n" % ( mean(total_time)*1e3 ))			
		# print("Average time from publishing to receiveing is %f milliseconds. \n" % (mean(self.transmit_time_list)* 1e3))

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