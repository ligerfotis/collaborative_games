#!/usr/bin/env python
import rospy
import gym
from std_msgs.msg import Int16
from main import Game 
from keypoint_3d_matching_msgs.msg import Keypoint3d_list


class controller:

	def __init__(self):
		print("init")
		# self.human_sub = rospy.Subscriber("/RW_x_direction", Int16, self.simulate)
		self.game = Game()
		self.human_sub = rospy.Subscriber("/topic_transform", Keypoint3d_list, self.simulate)
		self.prev_x = None
		self.speed = 1500
		
	def getShift(self,pos_x):
		if self.prev_x == None:
			self.prev_x = shift = pos_x
		else:
			shift = pos_x - self.prev_x
			self.prev_x = pos_x
		if abs(shift) < 0.004:
			return 0
		else:
			return shift * self.speed 

	def normalize(self,x_data):
		if  0 < x_data < 0.35:
			return x_data
		else:
			return 0

	def simulate(self, data):
		pos_x = data.keypoints[0].points.point.y # yes it is "y" because of the setup in lab
		shift = self.getShift(self.normalize(pos_x))

		if self.game.running:
			self.game.play(shift)
		else:
			self.game.endGame()


if __name__ == '__main__':
	rospy.init_node('hand_direction', anonymous=True)
	ctrl = controller()
	# while ctrl.game.running:
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")