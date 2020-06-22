#!/usr/bin/env python
import rospy
from hand_direction.msg import reward_observation, action_agent, action_human
from getch import getch, pause

# define hotkeys
escape = '\x1b'
exit = '\x03'


class RosAgent(object):
    def __init__(self):
        self.obs_sub = rospy.Subscriber("/rl/reward_game", reward_observation, self.obs_callback, queue_size=1)
        self.act_pub = rospy.Publisher("/rl/action_agent", action_agent, queue_size=1)
        self.act_pub = rospy.Publisher("/rl/action_human", action_human, queue_size=1)

    def obs_callback(self):
        if self.readKeyboard() == "w":
            print("1")
            self.action.action = +1
        elif self.readKeyboard() == "s":

            print("-1")
            self.action.action = -1

        self.act_pub.publish(self.action)

    def readKeyboard(self):
        key = getch()
        if key == "w":
            return "up"
        elif key == "s":
            return "down"
        # if key == "m" or key == escape or key == exit:
        #     break

if __name__ == '__main__':
    rospy.init_node('RosAgent', anonymous=True)
    agent = RosAgent()
    while not rospy.is_shutdown():
        print "here"
        agent.obs_callback()

    try:
        talker()
    except rospy.ROSInterruptException:
        pass