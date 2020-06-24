#!/usr/bin/env python
import rospy
from hand_direction.msg import observation, action_agent, reward_observation
from getch import getch, pause
import numpy as np
import message_filters
from std_msgs.msg import Int16, Float32
from hand_direction.msg import observation

class Observations(object):
    def __init__(self):
        self.obs_robot_sub = message_filters.Subscriber("/rl/reward_observation_robot", reward_observation)
        self.obs_human_sub = message_filters.Subscriber("/rl/observation_human", observation)


        self.ts = message_filters.ApproximateTimeSynchronizer([self.obs_robot_sub, self.obs_human_sub], 1, 1)
        self.ts.registerCallback(self.publish_full_observation_reward)

        self.obs_pub = rospy.Publisher("/rl/observation_reward", reward_observation, queue_size=10)


    def publish_full_observation_reward(self, rew_obs_robot, obs_human):
        new_rew_obs = rew_obs_robot

        # just concat the human's input on the state vector
        new_rew_obs.observations = np.concatenate([rew_obs_robot.observations,obs_human.observations],axis=None)
        
        self.obs_pub.publish(new_rew_obs)


if __name__ == '__main__':
    rospy.init_node('Joint_Observations_Rewards', anonymous=True)
    agent = Observations()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")