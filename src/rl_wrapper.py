#!/usr/bin/env python
import rospy
import gym
from std_msgs.msg import Int16
from environment import Game 
from keypoint_3d_matching_msgs.msg import Keypoint3d_list
from hand_direction.msg import reward_observation, action_human, action_agent


class controller:

	def __init__(self):
		print("init")
		# self.human_sub = rospy.Subscriber("/RW_x_direction", Int16, self.simulate)
		self.game = Game()
		self.human_sub = rospy.Subscriber("/rl/action_human", action_human, self.play_next_human)
		self.agent_sub = rospy.Subscriber("/rl/action_agent", action_agent, self.play_next_agent)

		self.reward_pub = rospy.Publisher('/rl/reward_game', reward_observation, queue_size = 10)

		self. play_next_lock = True
		# self.game.play(0,0)

		# r_0 = reward_observation()
		# r_0.reward = 0
		# r_0.observations = self.game.getObservations()
		# r_0.final_state = self.game.finished
		# if self.reward_pub.get_num_connections():
		# 	print "here"
		# 	self.reward_pub.publish(r_0)
		

	def play_next_human(self, action_human):
		# print("Action Human: " + str(action_human.action))
		if self.play_next_lock:
			self.play_next_lock = False
			if self.game.running:
				self.game.play(action_human.action, 0)
			else:
				self.game.endGame()
				
			r_0 = reward_observation()
			r_0.reward = self.game.getReward()
			r_0.observations = self.game.getObservations()
			r_0.final_state = self.game.finished
			self.reward_pub.publish( r_0 )
		self.play_next_lock = True


	def play_next_agent(self, action_agent):
		# print("Action Agent: " + str(action_agent.action))
		if self.play_next_lock:
			self.play_next_lock = False
			if self.game.running:
				self.game.play(0, action_agent.action)
			else:
				self.game.endGame()

			r_0 = reward_observation()
			r_0.reward = self.game.getReward()
			r_0.observations = self.game.getObservations()
			r_0.final_state = self.game.finished
			self.reward_pub.publish( r_0 )

		self.play_next_lock = True


if __name__ == '__main__':
	rospy.init_node('rl_wrapper', anonymous=True)
	ctrl = controller()
	# while ctrl.game.running:
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")