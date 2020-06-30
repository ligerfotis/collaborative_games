#!/usr/bin/env python
import rospy
import gym
import std_msgs
from std_msgs.msg import Int16, Float32
from environment import Game 
from keypoint_3d_matching_msgs.msg import Keypoint3d_list
from hand_direction.msg import observation, action_agent, reward_observation, action_human
import time
from statistics import mean 

class controller:

	def __init__(self):
		print("init")
		# self.human_sub = rospy.Subscriber("/RW_x_direction", Int16, self.simulate)
		self.game = Game()
		self.action_human1 = 0.0
		self.action_human2 = 0.0
		self.human_sub = rospy.Subscriber("/rl/action_x", action_human, self.set_action_human1)
		self.human_sub = rospy.Subscriber("/rl/action_y", action_human, self.set_action_human2)

		# self.reward_pub = rospy.Publisher('/rl/reward', Int16, queue_size = 10)
		self.obs_robot_pub = rospy.Publisher('/rl/reward_observation_robot', reward_observation, queue_size = 10, latch=True)
		self.transmit_time_list = []
		
		
	def set_action_human1(self, act):
		self.action_human1 = act.action
		self.transmit_time_list.append(rospy.get_rostime().to_sec()  - act.header.stamp.to_sec())

	def set_action_human2(self, act):
		self.action_human2 = act.action
		self.transmit_time_list.append(rospy.get_rostime().to_sec()- act.header.stamp.to_sec())


	def play_next_agent(self):
		# print self.action_human
		total_time = []
		while self.game.running:
			exec_time = self.game.play([self.action_human1, self.action_human2])
			
			total_time.append(exec_time)

		print("Average Execution time for play funtion is %f milliseconds. \n" % ( mean(total_time)*1e3 ))			
		self.game.endGame()

		print("Average time from publishing to receiveing is %f milliseconds. \n" % (mean(self.transmit_time_list)* 1e3))



if __name__ == '__main__':
	rospy.init_node('rl_wrapper', anonymous=True)
	ctrl = controller()
	ctrl.game.play([0, 0])

	ctrl.play_next_agent()
	# while ctrl.game.running:
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")