#!/usr/bin/env python
import rospy
from hand_direction.msg import reward, observation, action_agent, action_human, reward_observation
from getch import getch, pause
import numpy as np
from std_msgs.msg import Int16, Float32

# define hotkeys
escape = '\x1b'
exit = '\x03'

ACTION_SPACE = [0, 1] 

class RosAgent(object):
    def __init__(self):
        self.observation = rospy.Subscriber("/rl/observation_reward", reward_observation, self.update, queue_size=10)
        self.act_pub = rospy.Publisher("/rl/action", action_agent, queue_size=10)
        self.state = [0, 800 - 64] # Initial state for game
        self.reward  = 0
        self.final_state = False
        #self.act_pub = rospy.Publisher("/rl/action_human", action_human, queue_size=1)



    def update_reward(self, reward):
        self.reward = reward

    def update_state(self, state, final_state):
        self.state = state
        self.final_state = final_state
    

    def update(self, obs_reward):
        self.update_state(obs_reward.observations, obs_reward.final_state)
        self.update_reward(obs_reward.reward)
        human_act = obs_reward.observations[2]
        # rate = rospy.Rate(1)
        agent_act = np.random.randint(low=ACTION_SPACE[0], high=ACTION_SPACE[1]+1)
        act = action_agent()

        act.action = [human_act, float(agent_act)]

        self.act_pub.publish(act)
        # rate.sleep()


if __name__ == '__main__':
    rospy.init_node('ros_agent', anonymous=True)
    agent = RosAgent()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")