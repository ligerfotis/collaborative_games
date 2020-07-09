#!/usr/bin/env python
import sys
import random

import torch
from tqdm import tqdm

from hyperparams_ur10 import OFF_POLICY_BATCH_SIZE as BATCH_SIZE, DISCOUNT, ENTROPY_WEIGHT, HIDDEN_SIZE, LEARNING_RATE, \
    MAX_STEPS, POLYAK_FACTOR, REPLAY_SIZE, UPDATE_INTERVAL, UPDATE_START, SAVE_INTERVAL
from models_ur10_human import Critic, SoftActor, create_target_network, update_target_network
from hand_direction.msg import reward_observation, action_agent, action_human
# import rospy
from collections import deque
import numpy as np
from torch import optim
import pickle
import time 
import rospkg

resume = False
rospack = rospkg.RosPack()
package_path = rospack.get_path("hand_direction")

def next_action_random():
    rand = np.random.randint(low=1, high=11)
    if rand < 3:
        agent_act = -1
    elif 3 <= rand <= 6:
        agent_act = 1
    else:
        agent_act = 0

    return agent_act


class SAC:

    def __init__(self):
        self.next_state = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.action_space = 1
        self.state_space = 6
        self.actor = SoftActor(HIDDEN_SIZE, self.state_space).to(self.device)
        self.critic_1 = Critic(HIDDEN_SIZE, self.state_space, state_action=True).to(self.device)
        self.critic_2 = Critic(HIDDEN_SIZE, self.state_space, state_action=True).to(self.device)
        self.value_critic = Critic(HIDDEN_SIZE, self.state_space).to(self.device)

        self.reward_sparse = True
        self.reward_dense = False
        self.goal_x = 0.17
        self.goal_y = -0.14
        self.abs_max_theta = 0.1
        self.abs_max_phi = 0.1
        self.human_p = 5
        self.first_update = True
        self.reset_number = 0
        self.targets_reached_first500 = 0
        self.targets_reached = 0
        self.target_value_critic = create_target_network(self.value_critic).to(self.device)
        self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.critics_optimiser = optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
                                            lr=LEARNING_RATE)
        self.value_critic_optimiser = optim.Adam(self.value_critic.parameters(), lr=LEARNING_RATE)
        self.D = deque(maxlen=REPLAY_SIZE)
        # Automatic entropy tuning init
        self.target_entropy = -np.prod(self.action_space).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LEARNING_RATE)
        self.lr = None

        self.save_count = 0

        if resume:
            if os.path.isfile(package_path+"/src/scripts/checkpoints_human/agent.pth"):
                print("\nResuming training\n")
                checkpoint = torch.load(package_path+"/src/scripts/checkpoints_human/agent.pth")
                self.actor.load_state_dict(checkpoint['actor_state_dict'])
                self.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
                self.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
                self.value_critic.load_state_dict(checkpoint['value_critic_state_dict'])
                UPDATE_START = 1
                print(UPDATE_START)

                self.target_value_critic.load_state_dict(checkpoint['target_value_critic_state_dict'])
                self.actor_optimiser.load_state_dict(checkpoint['actor_optimiser_state_dict'])
                self.critics_optimiser.load_state_dict(checkpoint['critics_optimiser_state_dict'])
                self.value_critic_optimiser.load_state_dict(checkpoint['value_critic_optimiser_state_dict'])
                self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
                self.D = pickle.load( open( package_path+"/src/scripts/checkpoints_human/agent.p", "rb" ) )
            else:
                print("No checkpoint found at '{}'".format(package_path+"/src/scripts/checkpoints_human/agent.pth"))

        # print("Actor")
        # print(self.actor)
        # print("soft Q1")
        # print(self.critic_1)
        # print("soft Q2")
        # print(self.critic_2)
        # print("State Value V")
        # print(self.value_critic)



    def update_rw_state(self, state, reward, action, next_state, final_state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(next_state).to(self.device)
        self.D.append({'state': state.unsqueeze(0),
                       'action': action,
                       'reward': torch.tensor([reward], dtype=torch.float32, device=self.device),
                       'next_state': next_state.unsqueeze(0),
                       'done': torch.tensor([final_state], dtype=torch.float32, device=self.device)})

    def next_action(self, state, stochastic=True):
        start_time = time.time()
        try:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).to(self.device)
                if len(self.D) < UPDATE_START and not resume:
                    # To improve exploration take actions sampled
                    # from a uniform random distribution over actions at the start of training
                    # act = self.next_action_random()
                    # agent_act = torch.tensor([act], dtype=torch.float32, device=self.device).unsqueeze(0)
                    agent_act = torch.tensor([2 * random.random() - 1], device=self.device).unsqueeze(0)
                else:
                    # Observe state s and select action a ~ mu(a|s)
                    if stochastic:
                        agent_act = self.actor(state.unsqueeze(0)).sample()
                    else:
                        agent_act = self.actor(state.unsqueeze(0), stochastic)

                # print("Action Time: %f." % ( (time.time()-start_time)*1e3))
                return agent_act
        except KeyboardInterrupt:
            print("Exception in acting")


    def train(self, sample=None, verbose=True):
        try:
            if sample is None:
                # Randomly sample a batch of transitions B = {(s, a, r, s', d)} from D
                batch = random.sample(self.D, BATCH_SIZE)
                batch = dict((k, torch.cat([d[k] for d in batch], dim=0)) for k in batch[0].keys())
            else:
                state = torch.tensor(sample[0], dtype=torch.float32).to(self.device)
                next_state = torch.tensor(sample[3]).to(self.device)
                batch = {'state': state.unsqueeze(0),
                       'action': sample[2],
                       'reward': torch.tensor([sample[1]], dtype=torch.float32, device=self.device),
                       'next_state': next_state.unsqueeze(0),
                       'done': torch.tensor([sample[5]], dtype=torch.float32, device=self.device)}
                print("single batch")
                print(batch)

            # Compute targets for Q and V functions
            y_q = batch['reward'] + DISCOUNT * (1 - batch['done']) * self.target_value_critic(
                batch['next_state'])
            policy = self.actor(batch['state'])
            # a(s) is a sample from mu(:|s) which is differentiable wrt theta via the reparameterisation
            # trick
            action_update, log_prob = policy.rsample_log_prob()
            # Automatic entropy tuning
            alpha_loss = -(
                    self.log_alpha.float() * (log_prob + self.target_entropy).float().detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
            weighted_sample_entropy = (alpha.float() * log_prob).view(-1, 1)

            # Weighted_sample_entropy = ENTROPY_WEIGHT * policy.log_prob(action).sum(dim=1)  # Note: in practice it
            # is more numerically stable to calculate the log probability when sampling an action to avoid inverting
            # tanh
            y_v = torch.min(self.critic_1(batch['state'], action_update.detach()),
                            self.critic_2(batch['state'],
                                          action_update.detach())) - weighted_sample_entropy.detach()

            # Update Q-functions by one step of gradient descent
            value_loss = (self.critic_1(batch['state'], batch['action']) - y_q).pow(2).mean() + (
                    self.critic_2(batch['state'], batch['action']) - y_q).pow(2).mean()
            self.critics_optimiser.zero_grad()
            value_loss.backward()
            self.critics_optimiser.step()

            # Update V-function by one step of gradient descent
            value_loss = (self.value_critic(batch['state']) - y_v).pow(2).mean()
            self.value_critic_optimiser.zero_grad()
            value_loss.backward()
            self.value_critic_optimiser.step()

            # Update policy by one step of gradient ascent
            policy_loss = (weighted_sample_entropy - self.critic_1(batch['state'], action_update)).mean()
            self.actor_optimiser.zero_grad()
            policy_loss.backward()
            self.actor_optimiser.step()
            self.lr = self.critics_optimiser.state_dict()["param_groups"][0]["lr"]


            # Update target value network
            update_target_network(self.value_critic, self.target_value_critic, POLYAK_FACTOR)

            # Saving policy
            if self.save_count == SAVE_INTERVAL:
                self.save_count = 0
                # Reset
                if verbose:
                    print("Saving")

                torch.save({
                    'actor_state_dict': self.actor.state_dict(),
                    'critic_1_state_dict': self.critic_1.state_dict(),
                    'critic_2_state_dict': self.critic_1.state_dict(),
                    'value_critic_state_dict': self.value_critic.state_dict(),
                    'target_value_critic_state_dict': self.target_value_critic.state_dict(),
                    'value_critic_optimiser_state_dict': self.value_critic_optimiser.state_dict(),
                    'actor_optimiser_state_dict': self.actor_optimiser.state_dict(),
                    'critics_optimiser_state_dict': self.critics_optimiser.state_dict(),
                    'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
                }, "/home/liger/catkin_ws/src/hand_direction/src/scripts/checkpoints_human/agent.pth")
                # print("Saving replay buffer")
                pickle.dump(self.D, open("/home/liger/catkin_ws/src/hand_direction/src/scripts/checkpoints_human/agent.p", "wb"))
            self.save_count += 1

            torch.cuda.empty_cache()

        except KeyboardInterrupt:
            print("Exception in training")

            exit(1)


