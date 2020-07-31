#!/usr/bin/env python
import rospy
from std_msgs.msg import Int16
from keypoint_3d_matching_msgs.msg import Keypoint3d_list
from hand_direction.msg import action_msg
import numpy as np
from std_msgs.msg import Float32
import std_msgs
import time


offset = 0.07

class Converter:

	def __init__(self):
		# print("init")
		self.keypoint_sub = rospy.Subscriber("/topic_transform", Keypoint3d_list, self.callback)
		self.action_human_pub = rospy.Publisher('/rl/action_x', action_msg, queue_size = 1)
		self.prev_x = None
		self.start_time = None

	def getShift(self, pos_x):
			if self.prev_x == None:
				self.prev_x = pos_x
				shift = 0
				# shift = pos_x
			else:
				shift = pos_x - self.prev_x

			if abs(shift) < offset:
				return 0
			else:
				self.prev_x = pos_x
				# if self.start_time is not None:
				# 	print(time.time() - self.start_time)
				# self.start_time = time.time()
				# return shift
				if shift < 0:
					return 1
				else:
					return -1

	def normalize(self, x_data):
		if  0 < abs(x_data) < 0.35:
			return x_data
		else:
			return 0

	def callback(self, data):
		h = std_msgs.msg.Header()
		h.stamp = rospy.Time.now() 
		'''
		pos_x = data.keypoints[0].points.point.x
		'''
		pos_x = data.keypoints[0].points.point.x # yes it is "y" because of the setup in lab
		if 0.0 < pos_x:
			pos_x = 0
		elif pos_x < - 0.35:
			pos_x = 0.35

		pos_x = pos_x + 0.175

		act = action_msg()
		act.action = pos_x/2
		act.header = h
		
		self.action_human_pub.publish(act)
		

if __name__ == '__main__':
	rospy.init_node('keypoint_to_action', anonymous=True)
	converter = Converter()
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")