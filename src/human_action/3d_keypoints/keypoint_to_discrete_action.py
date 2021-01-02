#!/usr/bin/env python
import sys
import rospy
from keypoint_3d_matching_msgs.msg import Keypoint3d_list
from collaborative_games.msg import action_msg
import yaml


def get_config(config_file='config_keypoints.yaml'):
    try:
        with open(config_file) as file:
            yaml_data = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file')

    return yaml_data


class HumanAction:

    def __init__(self, config="config_keypoints.yaml"):
        self.config = get_config("../../maze3D_game/" + config)
        topic_to_listen_to = self.config['topic_to_listen_to']
        self.keypoint_sub = rospy.Subscriber(topic_to_listen_to, Keypoint3d_list, self.callback)
        self.action_human_pub = rospy.Publisher('/rl/action_x', action_msg, queue_size=1)
        self.prev_x = None
        self.start_time = None

    def callback(self, pos_x):

        # get the header from the received message
        h = pos_x.keypoints[0].points.header
        # get the keypoint of wrist
        pos_x = pos_x.keypoints[0].points.point.x

        # set the starting point of the wrist
        starting_point = self.config['start_keypoint']
        offset = self.config['offset']
        """
        s: starting point
        o/2: offset/2
        ================================================
                -1      |      0       |       1      
        ================================================
               LEFT     |    CENTER    |     RIGHT      
        ------------------------------------------------
        ||<---offset--->|              |              ||
        ||              |      s       |              ||
        ||       <-o/2->|<-o/2-><-o/2->|<-o/2->       ||
        ------------------------------------------------
        """
        if pos_x < starting_point - (offset / 2):
            action = -1
        elif pos_x > starting_point + (offset / 2):
            action = 1
        else:
            action = 0

        act = action_msg()
        act.action = action
        act.header = h
        self.action_human_pub.publish(act)
        return action


if __name__ == '__main__':
    rospy.init_node('keypoint_to_action', anonymous=True)
    action_listener = HumanAction()  # take first argument
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
