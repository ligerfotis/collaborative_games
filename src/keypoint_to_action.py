#!/usr/bin/env python
import rospy
from std_msgs.msg import Int16
from keypoint_3d_matching_msgs.msg import Keypoint3d_list
from hand_direction.msg import action_msg
import numpy as np
from std_msgs.msg import Float32
import std_msgs

class Converter:

	def __init__(self):
		# print("init")
		self.keypoint_sub = rospy.Subscriber("/topic_transform", Keypoint3d_list, self.callback)
		self.action_human_pub = rospy.Publisher('/rl/action_x', action_msg, queue_size = 10)
		self.prev_x = None
		# self.speed = 1500
		

	def getShift(self, pos_x):
			if self.prev_x == None:
				self.prev_x = shift = pos_x
			else:
				shift = pos_x - self.prev_x

			if abs(shift) < 0.10:
				return 0
			else:
				# return shift * self.speed
				self.prev_x = pos_x
				if shift < 0:
					return -1
				else:
					return 1

	def normalize(self, x_data):
		if  0 < x_data < 0.35:
			return x_data
		else:
			return 0

	def callback(self, data):
		h = std_msgs.msg.Header()
		h.stamp = rospy.Time.now() 

		pos_x = data.keypoints[0].points.point.y # yes it is "y" because of the setup in lab
		shift = self.getShift(self.normalize(pos_x))
		
		act = action_msg()
		act.action = np.asarray([float(shift)])
		act.header = h
		
		self.action_human_pub.publish(act)
		

if __name__ == '__main__':
	rospy.init_node('keypoint_to_action', anonymous=True)
	converter = Converter()
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")