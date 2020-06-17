#!/usr/bin/env python
import rospy
from openpose_ros_msgs.msg import OpenPoseHumanList, OpenPoseHuman, PointWithProb
import sys
from std_msgs.msg import Int16

class hand_direction:

    def __init__(self):
        self.direction_pub = rospy.Publisher('/RW_x_direction', Int16, queue_size = 10)
        self.human_sub = rospy.Subscriber("/openpose_ros/human_list", OpenPoseHumanList, self.callback)
        self.current_x = None;

    def getDirection(self, point):
        direction = None #pixels
        
        if self.current_x is None:
        	self.current_x = point.x
        	direction = 0
        else:
        	direction = int(point.x - self.current_x)
        	# if abs(direction) > 2:
        	# 	direction = 0
        return direction 


    def callback(self, human_list):
   
		# publish the coords of the right wrist
		#self.direction_pub.publish(human_list.human_list[0].body_key_points_with_prob[4])
		dir = self.getDirection(human_list.human_list[0].body_key_points_with_prob[4])
		self.direction_pub.publish(dir)


def listener(args):
    rospy.init_node('hand_direction', anonymous=True)
    ic = hand_direction()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    listener(sys.argv)