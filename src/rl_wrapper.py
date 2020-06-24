#!/usr/bin/env python
import rospy
import gym
import std_msgs
from std_msgs.msg import Int16
from environment import Game 
from keypoint_3d_matching_msgs.msg import Keypoint3d_list
from hand_direction.msg import observation, action_agent, reward_observation
from std_srvs.srv import Empty,EmptyResponse, Trigger

class controller:

	def __init__(self):
		print("init")
		# self.human_sub = rospy.Subscriber("/RW_x_direction", Int16, self.simulate)
		self.game = Game()
		# self.human_sub = rospy.Subscriber("/rl/action_human", action_human, self.play_next_human)
		self.agent_sub = rospy.Subscriber("/rl/action", action_agent, self.play_next_agent)

		# self.reward_pub = rospy.Publisher('/rl/reward', Int16, queue_size = 10)
		self.obs_robot_pub = rospy.Publisher('/rl/reward_observation_robot', reward_observation, queue_size = 10, latch=True)
		
		# rospy.sleep(0.2)
		self.publish_reward_observations()


	def publish_reward_observations(self):
		h = std_msgs.msg.Header()
		h.stamp = rospy.Time.now() 

		rew_obs = reward_observation()
		rew_obs.header = h
		rew_obs.observations = self.game.getObservations()
		rew_obs.final_state = self.game.finished
		rew_obs.reward = self.game.getReward()

		self.obs_robot_pub.publish(rew_obs)

	def play_next_agent(self, action_agent):
		# print("Action Agent: " + str(action_agent.action))
		# if self.play_next_lock:
			# self.play_next_lock = False
		if self.game.running:
			self.game.play(action_agent.action)
		else:
			self.game.endGame()

		self.publish_reward_observations()


		# self.play_next_lock = True


if __name__ == '__main__':
	rospy.init_node('rl_wrapper', anonymous=True)
	ctrl = controller()
	
	ctrl.game.play([0, 0])
	# while ctrl.game.running:
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")