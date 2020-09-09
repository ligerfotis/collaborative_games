#!/usr/bin/env python
import rospy
from std_msgs.msg import Int16
from keypoint_3d_matching_msgs.msg import Keypoint3d_list
from hand_direction.msg import action_msg
import numpy as np
from std_msgs.msg import Float32
import std_msgs
import time
import math

offset = 0.07

class Converter:

	def __init__(self):
		# print("init")
		self.keypoint_sub = rospy.Subscriber("/topic_transform", Keypoint3d_list, self.callback)
		self.action_human_pub = rospy.Publisher('/rl/action_x', action_msg, queue_size = 1)
		self.prev_x = None
		self.start_time = None
		
	def callback(self, data):
		# h = std_msgs.msg.Header()
		# h.stamp = rospy.Time.now() 
		h = data.keypoints[0].points.header
		
		pos_x = data.keypoints[0].points.point.x 
		pos_x = pos_x + 0.26

		if 0.20 < pos_x:
			pos_x = 0.20
		elif pos_x < - 0.20:
			pos_x = - 0.20

		pos_x = pos_x / 2
		
		pos_x = round(pos_x * 100)/10

		act = action_msg()
		act.action = pos_x
		act.header = h
		
		self.action_human_pub.publish(act)
		

if __name__ == '__main__':
	rospy.init_node('keypoint_to_action', anonymous=True)
	converter = Converter()
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")