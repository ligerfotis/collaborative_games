#!/usr/bin/env python
import rospy
import gym
import std_msgs
from std_msgs.msg import Int16, Float32
from environment import Game 
from keypoint_3d_matching_msgs.msg import Keypoint3d_list
from hand_direction.msg import observation, action_agent, reward_observation, action_human
from std_srvs.srv import Empty,EmptyResponse, Trigger

class controller:

	def __init__(self):
		print("init")
		# self.human_sub = rospy.Subscriber("/RW_x_direction", Int16, self.simulate)
		self.game = Game()
		self.action_human1 = 0.0
		self.action_human2 = 0.0
		self.human_sub = rospy.Subscriber("/rl/action_x", Float32, self.set_action_human1)
		self.human_sub = rospy.Subscriber("/rl/action_y", Float32, self.set_action_human2)

		# self.reward_pub = rospy.Publisher('/rl/reward', Int16, queue_size = 10)
		self.obs_robot_pub = rospy.Publisher('/rl/reward_observation_robot', reward_observation, queue_size = 10, latch=True)
		
		
	def set_action_human1(self,action_human):
		self.action_human1 = action_human.data

	def set_action_human2(self,action_human):
			self.action_human2 = action_human.data


	def play_next_agent(self):
		# print self.action_human
		while self.game.running:
			self.game.play([self.action_human1, self.action_human2])
			
		self.game.endGame()

		

		# self.play_next_lock = True


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