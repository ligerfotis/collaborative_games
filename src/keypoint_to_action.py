#!/usr/bin/env python
import rospy
from std_msgs.msg import Int16
from keypoint_3d_matching_msgs.msg import Keypoint3d_list
from hand_direction.msg import reward_observation, action_human

class Converter:

	def __init__(self):
		print("init")
		self.keypoint_sub = rospy.Subscriber("/topic_transform", Keypoint3d_list, self.callback)
		self.action_pub = rospy.Publisher('/rl/action_human', action_human, queue_size = 10)
		self.prev_x = None
		self.speed = 1500
		

	def getShift(self, pos_x):
			if self.prev_x == None:
				self.prev_x = shift = pos_x
			else:
				shift = pos_x - self.prev_x
				self.prev_x = pos_x
			if abs(shift) < 0.004:
				return 0
			else:
				return shift * self.speed 

	def normalize(self, x_data):
		if  0 < x_data < 0.35:
			return x_data
		else:
			return 0

	def callback(self, data):
			pos_x = data.keypoints[0].points.point.y # yes it is "y" because of the setup in lab
			shift = self.getShift(self.normalize(pos_x))
			act = action_human()
			act.action = shift
			self.action_pub.publish(act)
		

if __name__ == '__main__':
	rospy.init_node('keypoint_to_action', anonymous=True)
	converter = Converter()
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")