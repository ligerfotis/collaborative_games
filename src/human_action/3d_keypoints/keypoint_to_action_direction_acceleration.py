#!/usr/bin/env python
import rospy
from std_msgs.msg import Int16
from keypoint_3d_matching_msgs.msg import Keypoint3d_list
from collaborative_games.msg import action_msg
import numpy as np
from std_msgs.msg import Float32
import std_msgs
import time


offset = 0.04

class Converter:

	def __init__(self):
		# print("init")
		self.keypoint_sub = rospy.Subscriber("/topic_transform", Keypoint3d_list, self.callback)
		self.action_human_pub = rospy.Publisher('/rl/action_x', action_msg, queue_size = 1)
		self.prev_x = None
		self.start_time = None
		self.prev_shift = None

	def getShift(self, pos_x):
			
			if abs(pos_x) < offset:
				return 0
			else:
				if pos_x < 0:
					return -1
				else:
					return 1


	def callback(self, data):
		h = std_msgs.msg.Header()
		h.stamp = rospy.Time.now() 
		
		pos_x = data.keypoints[0].points.point.x 
		pos_x = pos_x + 0.28

		pos_x = self.getShift(pos_x)

		if pos_x != 0 and (self.prev_x is None or self.prev_x != pos_x):
			act = action_msg()
			act.action = pos_x
			act.header = h
			
			self.action_human_pub.publish(act)

			self.prev_x = pos_x
			

if __name__ == '__main__':
	rospy.init_node('keypoint_to_action', anonymous=True)
	converter = Converter()
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")
