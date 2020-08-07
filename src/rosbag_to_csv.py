#!/usr/bin/env python
import rospy 
from keypoint_3d_matching_msgs.msg import Keypoint3d_list
import rospkg
import numpy as np
import time
from std_msgs.msg import Int16, Float32
import sys
from hand_direction.msg import observation, action_agent, reward_observation, action_msg

rospack = rospkg.RosPack()
package_path = rospack.get_path("hand_direction")

class Listener:
	def __init__(self):

		self.keypoint_sub = rospy.Subscriber("/topic_transform", Keypoint3d_list, self.callback_x_points)
		self.act_human_sub = rospy.Subscriber("/rl/action_x", action_msg, self.set_human_action)
		self.act_human_sub = rospy.Subscriber("/rl/turtle_accel", Float32, self.set_turtle_accel)

		self.list_x = []
		self.list_time = []
		self.human_action = []
		self.human_time = []
		self.turtle_accel = []
		self.turtle_accel_time = []

		self.plot_directory = package_path + "/src/plots/"

	def callback_x_points(self, data):
		self.list_x.append(data.keypoints[0].points.point.x)
		self.list_time.append(rospy.get_rostime().to_sec())

	def set_human_action(self, action_human):
		self.human_time.append(rospy.get_rostime().to_sec())
		self.human_action.append(action_human.action)

	def set_turtle_accel(self, turtle_accel):
		self.turtle_accel_time.append(rospy.get_rostime().to_sec())
		self.turtle_accel.append(turtle_accel.data)

		
if __name__ == '__main__':
	plot_type = sys.argv[1]
	rospy.init_node('listener', anonymous=True)
	converter = Listener()
	np.savetxt(converter.plot_directory + 'x_points.csv', converter.list_x, delimiter=',', fmt='%f')
	try:
		while not rospy.is_shutdown():
			rospy.spin()
		np.savetxt(converter.plot_directory + plot_type + "/"+ 'human_action_'+plot_type+'.csv', converter.human_action, delimiter=',', fmt='%f')
		np.savetxt(converter.plot_directory + plot_type + "/"+ 'human_action_time_'+plot_type+'.csv', converter.human_time, delimiter=',', fmt='%f')
		
		np.savetxt(converter.plot_directory + plot_type + "/"+ 'x_points_'+plot_type+'.csv', converter.list_x, delimiter=',', fmt='%f')
		np.savetxt(converter.plot_directory + plot_type + "/"+ 'x_points_time_'+plot_type+'.csv', converter.list_time, delimiter=',', fmt='%f')

		np.savetxt(converter.plot_directory + plot_type + "/"+ 'x_turtle_accel_' + plot_type + '.csv', converter.turtle_accel, delimiter=',', fmt='%f')
		np.savetxt(converter.plot_directory + plot_type + "/"+ 'x_turtle_accel_time_' + plot_type + '.csv', converter.turtle_accel_time, delimiter=',', fmt='%f')

	except KeyboardInterrupt:
		print("Shutting down")