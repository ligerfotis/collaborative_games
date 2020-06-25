#!/usr/bin/env python
import rospy
from hand_direction.msg import reward, observation, action_agent, action_human, reward_observation
from getch import getch, pause
import numpy as np
from std_msgs.msg import Int16, Float32
from sac import SAC
import random


ACTION_SPACE = [0, 1] 

class RosAgent:
    def __init__(self):
        self.observation_reward = rospy.Subscriber("/rl/observation_reward", reward_observation, self.policy, queue_size=10)
        self.act_pub = rospy.Publisher("/rl/action", action_agent, queue_size=10)
        self.state = [0, 800 - 64] # Initial state for game
        self.reward  = 0
        self.final_state = False
        self.agent = SAC()

    def policy(self, obs_reward):
        self.agent.update_rw_state(obs_reward.observations, obs_reward.reward, obs_reward.final_state)
        self.reward = obs_reward.reward
        self.state = obs_reward.observations
        human_act = self.state[2]
        # # rate = rospy.Rate(1)
        agent_act = np.random.randint(low=ACTION_SPACE[0], high=ACTION_SPACE[1]+1)

        # agent_act = self.agent.next_action()
        print("reward: %d" %(self.reward))

        act = action_agent()

        act.action = [human_act, agent_act]

        self.act_pub.publish(act)
        # rate.sleep()


if __name__ == '__main__':
    rospy.init_node('ros_agent', anonymous=True)
    agent = RosAgent()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")