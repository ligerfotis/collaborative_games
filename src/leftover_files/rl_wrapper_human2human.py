#!/mnt/34C28480C28447D6/PycharmProjects/maze3d_collaborative/venv/bin/python
import rospy
# import gym
import std_msgs
from std_msgs.msg import Int16, Float32
from environment import Game 
import torch
# from keypoint_3d_matching_msgs.msg import Keypoint3d_list
from collaborative_games.msg import observation, action_agent, reward_observation, action_human
import time
from statistics import mean 

class controller:

	def __init__(self):
		print("init")
		self.experiments_num = 10
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



	def game_loop(self):
		total_time = []
		# print self.action_human
		for exp in range(self.experiments_num):
			print("Experiment %d" % exp)
			while self.game.running:
				exec_time = self.game.play([self.action_human1, self.action_human2])
				total_time.append(exec_time)
				
			# reset game
			self.game = Game()
			self.game.start_time = time.time()

		self.game.endGame()

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
