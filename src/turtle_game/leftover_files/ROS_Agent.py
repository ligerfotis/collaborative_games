#!/usr/bin/env python
import rospy
from collaborative_games.msg import reward, observation, action_agent, action_human, reward_observation
from getch import getch, pause
import numpy as np
from std_msgs.msg import Int16, Float32
from sac import SAC
import random
import std_msgs

ACTION_SPACE = [0, 1] 

class RosAgent:
    def __init__(self):
        self.observation_reward = rospy.Subscriber("/rl/environment_response", reward_observation, self.policy, queue_size=10)
        self.act_pub = rospy.Publisher("/rl/final_action", action_agent, queue_size=10)
        self.prev_state = None
        self.state = None
        self.reward  = None
        self.final_state = None
        self.agent = SAC()

    def get_state(self, obs):
        self.state = obs.observations[:-1]

    def get_human_act(self, obs):
        self.human_act = obs.observations[-1]


    def policy(self, obs_reward):
        h = std_msgs.msg.Header()
        h.stamp = rospy.Time.now() 

        self.prev_state = obs_reward.prev_state
        self.reward = obs_reward.reward
        self.get_state(obs_reward)
        self.get_human_act(obs_reward)
        self.final_state = obs_reward.final_state

        self.agent.update_rw_state(self.prev_state, self.reward, agent_act, self.state, self.final_state)


        agent_act = self.agent.next_action(self.state)
        # exec_time = self.game.play([0.0, agent_act.item()])   


        
        # agent_act = self.agent.next_action()
        # print("reward: %d" %(self.reward))

        act = action_agent()
        act.header = h
        act.action = [self.human_act, agent_act]
        self.act_pub.publish(act)
        # rate.sleep()


if __name__ == '__main__':
    rospy.init_node('ros_agent', anonymous=True)
    agent = RosAgent()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
