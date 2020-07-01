#!/usr/bin/env python
import rospy
from hand_direction.msg import action_msg, action_agent, reward_observation
from getch import getch, pause
import numpy as np
import message_filters
from std_msgs.msg import Int16, Float32
from hand_direction.msg import observation

class Observations(object):
    def __init__(self):
        self.obs_robot_sub = message_filters.Subscriber("/rl/reward_and_observation_game", reward_observation)
        self.act_human_sub = message_filters.Subscriber("/rl/hand_action_x", action_msg)


        self.ts = message_filters.ApproximateTimeSynchronizer([self.obs_robot_sub, self.act_human_sub], 1, 1)
        self.ts.registerCallback(self.publish_full_observation_reward)

        self.obs_pub = rospy.Publisher("/rl/environment_response", reward_observation, queue_size=10)


    def publish_full_observation_reward(self, obs_game, act_human):
        new_obs = obs_game

        # just concat the human's input on the state vector
        new_obs.observations = np.concatenate([new_obs.observations,act_human.action],axis=None)
        self.obs_pub.publish(new_obs)


if __name__ == '__main__':
    rospy.init_node('Joint_Observations_Rewards', anonymous=True)
    agent = Observations()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")