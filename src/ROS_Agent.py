#!/usr/bin/env python
import rospy
from hand_direction.msg import reward_observation, action_agent, action_human
from getch import getch, pause
import numpy as np

# define hotkeys
escape = '\x1b'
exit = '\x03'

ACTION_SPACE = [0, 1] 

class RosAgent(object):
    def __init__(self):
        self.obs_sub = rospy.Subscriber("/rl/reward_game", reward_observation, self.policy, queue_size=10)
        self.act_pub = rospy.Publisher("/rl/action_agent", action_agent, queue_size=10)
        #self.act_pub = rospy.Publisher("/rl/action_human", action_human, queue_size=1)

    def policy(self, response):
        rate = rospy.Rate(1)
        agent_act = np.random.randint(low=ACTION_SPACE[0], high=ACTION_SPACE[1]+1)
        act = action_agent().action = agent_act*5
        self.act_pub.publish(act)
        rate.sleep()


if __name__ == '__main__':
    rospy.init_node('ros_agent', anonymous=True)
    agent = RosAgent()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")