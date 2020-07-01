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
from statistics import mean 

class controller:

	def __init__(self):
		print("init")
		# self.human_sub = rospy.Subscriber("/RW_x_direction", Int16, self.simulate)
		self.game = Game()
		self.action_human = 0.0
		self.action_agent = 0.0
		self.human_sub = rospy.Subscriber("/rl/hand_action_x", action_msg, self.set_action_human)
		self.agent_sub = rospy.Subscriber("/rl/action_y", action_msg, self.set_action_agent)

		# self.reward_pub = rospy.Publisher('/rl/reward', Int16, queue_size = 10)
		self.obs_robot_pub = rospy.Publisher('/rl/reward_and_observation_game', reward_observation, queue_size = 10, latch=True)
		
		self.transmit_time_list = []
		
		self.publish_reward_and_observations()

	def set_action_human(self,action_human):
		self.action_human = action_human.action
		self.transmit_time_list.append(rospy.get_rostime().to_sec()  - action_human.header.stamp.to_sec())

	def set_action_agent(self,action_agent):
		self.action_agent = action_agent.action
		self.transmit_time_list.append(rospy.get_rostime().to_sec()  - action_agent.header.stamp.to_sec())


	def publish_reward_and_observations(self):
		h = std_msgs.msg.Header()
		h.stamp = rospy.Time.now() 

		new_obs = reward_observation()
		new_obs.header = h
		new_obs.observations = self.game.getObservations()
		new_obs.final_state = self.game.finished
		new_obs.reward = self.game.getReward()

		self.obs_robot_pub.publish(new_obs)

	def game_loop(self):
		total_time = []
		# print self.action_human
		while self.game.running:
			exec_time = self.game.play([self.action_human, self.action_agent])
			total_time.append(exec_time)
			self.publish_reward_and_observations()
			
		self.game.endGame()
		self.publish_reward_and_observations()

		print("Average Execution time for play funtion is %f milliseconds. \n" % ( mean(total_time)*1e3 ))			
		print("Average time from publishing to receiveing is %f milliseconds. \n" % (mean(self.transmit_time_list)* 1e3))

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