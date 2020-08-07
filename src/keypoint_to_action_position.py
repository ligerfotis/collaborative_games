#!/usr/bin/env python
import rospy
from std_msgs.msg import Int16
from keypoint_3d_matching_msgs.msg import Keypoint3d_list
from hand_direction.msg import action_msg
import numpy as np
from std_msgs.msg import Float32, Int16
import std_msgs
import time
from robot_action import convertCm2Pixels, regulate

human_p = 1e-2

class Converter:

	def __init__(self):
		# print("init")
		self.keypoint_sub = rospy.Subscriber("/topic_transform", Keypoint3d_list, self.callback)
		self.keypoint_sub = rospy.Subscriber("/rl/turtle_pos_X", Int16, self.get_turtle_pos_x)
		self.action_human_pub = rospy.Publisher('/rl/action_x', action_msg, queue_size = 10)
		self.prev_x = None
		self.start_time = None
		self.turtle_pos_x = 0

	def get_turtle_pos_x(self, pos_x):
		self.turtle_pos_x = pos_x.data

	def callback(self, data):
		# if self.turtle_pos_x is not None:
		h = std_msgs.msg.Header()
		h.stamp = rospy.Time.now() 
		'''
		pos_x = data.keypoints[0].points.point.x
		'''
		pos_x = 0.35 + data.keypoints[0].points.point.x # yes it is "y" because of the setup in lab
		if 0.0 > pos_x:
			pos_x = 0
		elif pos_x > 0.35:
			pos_x = 0.35

		x_new, _ = convertCm2Pixels(pos_x, 0)
		# print ((float(self.turtle_pos_x))/float(x_new))
		cmd_vel_x = human_p  * ( x_new - self.turtle_pos_x)

		act = action_msg()
		act.action = cmd_vel_x
		act.header = h
		
		self.action_human_pub.publish(act)
	

if __name__ == '__main__':
	rospy.init_node('keypoint_to_action_position', anonymous=True)
	converter = Converter()
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")