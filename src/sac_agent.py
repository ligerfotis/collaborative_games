#!/usr/bin/env python
import torch
from tqdm import tqdm
from hyperparams_ur10 import OFF_POLICY_BATCH_SIZE as BATCH_SIZE, DISCOUNT, ENTROPY_WEIGHT, HIDDEN_SIZE, LEARNING_RATE, MAX_STEPS, POLYAK_FACTOR, REPLAY_SIZE, UPDATE_INTERVAL, UPDATE_START, SAVE_INTERVAL
from models_ur10_human import Critic, SoftActor, create_target_network, update_target_network
from hand_direction.msg import reward_observation, action_agent, action_human
import rospy
from collections import deque
import numpy as np
from torch import optim

class SAC:

	def __init__(self):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
		self.state = None
		self.done = False
		self.reward = 0

		self.action_space = 1
		self.state_space = 3
		self.actor = SoftActor(HIDDEN_SIZE).to(self.device)
		self.critic_1 = Critic(HIDDEN_SIZE, state_action=True).to(self.device)
		self.critic_2 = Critic(HIDDEN_SIZE, state_action=True).to(self.device)
		self.value_critic = Critic(HIDDEN_SIZE).to(self.device)

		self.reward_sparse = True
		self.reward_dense = False
		self.goal_x = 0.17
		self.goal_y = -0.14
		self.abs_max_theta = 0.1
		self.abs_max_phi = 0.1
		self.human_p = 5
		self.pbar = tqdm(xrange(1, MAX_STEPS + 1), unit_scale=1, smoothing=0)
		self.first_update = True
		self.reset_number = 0
		self.targets_reached_first500 = 0
		self.targets_reached = 0
		self.target_value_critic = create_target_network(self.value_critic).to(self.device)
		self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
		self.critics_optimiser = optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=LEARNING_RATE)
		self.value_critic_optimiser = optim.Adam(self.value_critic.parameters(), lr=LEARNING_RATE)
		self.D = deque(maxlen=REPLAY_SIZE)
		# Automatic entropy tuning init
		self.target_entropy = -np.prod(self.action_space).item()
		self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
		self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LEARNING_RATE)



	def update_rw_state(self, state, reward, final_state):
		self.state = state
		self.reward = reward
		self.final_state = final_state
	
	def next_action(self):
		agent_act = float(np.random.randint(low=0, high=2))
		return agent_act

	def train(self):
		# Training loop
		for step in pbar:
			try:
				with torch.no_grad():
					if step < UPDATE_START and sys.argv[-1] != "resuming":
						# To improve exploration take actions sampled from a uniform random distribution over actions at the start of training
						action = torch.tensor([2 * random.random() - 1, 2 * random.random() - 1], device=device).unsqueeze(0)
					else:
						# Observe state s and select action a ~ mu(a|s)
						action = self.actor(state.unsqueeze(0)).sample()
					# Execute a in the environment and observe next state s', reward r, and done signal d to indicate whether s' is terminal
					# Must compensate for translation
					current_pose = move_group.get_current_pose().pose
					current_rpy = move_group.get_current_rpy()
					total_action = [action[0][0], action[0][1]]
					
					# Setting linear commands to be published
			        # Setting angular commands, enforcing constraints
	                # Setting frame to base_link for planning
			        # Need to transform to base_link frame for planning
			        # Publish action
					T1 = rospy.get_rostime()
					self.cmd.publish(training_msg)
					rospy.sleep(0.05)

				# Starting updates of weights
				if step > UPDATE_START and step % UPDATE_INTERVAL == 0:
					if first_update:
						print("\nStarting updates")
						first_update = False
					# Randomly sample a batch of transitions B = {(s, a, r, s', d)} from D
					batch = random.sample(D, BATCH_SIZE)
					batch = dict((k, torch.cat([d[k] for d in batch], dim=0)) for k in batch[0].keys())

					# Compute targets for Q and V functions
					y_q = batch['reward'] + DISCOUNT * (1 - batch['done']) * target_value_critic(batch['next_state'])
					policy = self.actor(batch['state'])
					action_update, log_prob = policy.rsample_log_prob()  # a(s) is a sample from mu(:|s) which is differentiable wrt theta via the reparameterisation trick
					# Automatic entropy tuning
					alpha_loss = -(self.log_alpha.float() * (log_prob + self.target_entropy).float().detach()).mean()
					self.alpha_optimizer.zero_grad()
					alpha_loss.backward()
					self.alpha_optimizer.step()
					alpha = self.log_alpha.exp()
					weighted_sample_entropy = (alpha.float() * log_prob).view(-1,1)

					# Weighted_sample_entropy = ENTROPY_WEIGHT * policy.log_prob(action).sum(dim=1)  # Note: in practice it is more numerically stable to calculate the log probability when sampling an action to avoid inverting tanh
					y_v = torch.min(self.critic_1(batch['state'], action_update.detach()), self.critic_2(batch['state'], action_update.detach())) - weighted_sample_entropy.detach()

					# Update Q-functions by one step of gradient descent
					value_loss = (self.critic_1(batch['state'], batch['action']) - y_q).pow(2).mean() + (self.critic_2(batch['state'], batch['action']) - y_q).pow(2).mean()
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

					# Update target value network
					update_target_network(value_critic, target_value_critic, POLYAK_FACTOR)

				# Check if action has been executed for enough time
				T2 = rospy.get_rostime()
				print((T2.nsecs-T1.nsecs)*10e-7)
				while (T2-T1) < rospy.Duration.from_sec(0.2):
					try:
						T2 = rospy.get_rostime()
					continue:
						except KeyboardInterrupt:
						break
				# Action executed now calculate reward
				# Reward offered by the envirnment

				# Check if Done
				if self.final_state:
					done = True
					print("\nReached target")
					self.targets_reached += 1
					if step < UPDATE_START:
						self.targets_reached_first500 += 1

				# Store (s, a, r, s', d) in replay buffer D
				D.append({'state': state.unsqueeze(0), 'action': action, 'reward': torch.tensor([reward], dtype=torch.float32, device=device), 'next_state': next_state.unsqueeze(0), 'done': torch.tensor([done], dtype=torch.float32, device=device)})
				#T_append = rospy.get_rostime()
				#print("Append", (T_append.nsecs - T_check_done.nsecs)*10e-7)
				# If s' is terminal, reset environment state
				if done and step % SAVE_INTERVAL != 0:
					print("\nResetting")
				if reset_number == 0:
					reset(start_rpy1, move_group, cmd, reset_msg)
					reset_number += 1
				elif reset_number == 1:
					reset(start_rpy2, move_group, cmd, reset_msg)
					reset_number += 1
				else:
					reset(start_rpy3, move_group, cmd, reset_msg)
					reset_number = 0
				done = False

				# Saving policy
				if step % SAVE_INTERVAL == 0:
					# Reset
					print("Saving")
				if reset_number == 0:
					reset(start_rpy1, move_group, cmd, reset_msg)
					reset_number += 1
				elif reset_number == 1:
					reset(start_rpy2, move_group, cmd, reset_msg)
					reset_number += 1
				else:
					reset(start_rpy3, move_group, cmd, reset_msg)
					reset_number = 0
				done = False
				torch.save({
				'actor_state_dict': actor.state_dict(),
				'critic_1_state_dict': critic_1.state_dict(),
				'critic_2_state_dict': critic_1.state_dict(),
				'value_critic_state_dict': value_critic.state_dict(),
				'target_value_critic_state_dict': target_value_critic.state_dict(),
				'value_critic_optimiser_state_dict': value_critic_optimiser.state_dict(),
				'actor_optimiser_state_dict': actor_optimiser.state_dict(),
				'critics_optimiser_state_dict': critics_optimiser.state_dict(),
				'alpha_optimizer_state_dict': alpha_optimizer.state_dict(),
				},package_path+"/scripts/checkpoints_human/agent.pth")
				print("Saving replay buffer")
				pickle.dump( D, open( package_path+"/scripts/checkpoints_human/agent.p", "wb" ) )

			torch.cuda.empty_cache()
			#train_rate.sleep()

		except KeyboardInterrupt:
			break
print("Finished training")
print("Targets reached with random policy:", targets_reached_first500)
print("Targets reached with overall:", targets_reached)


	def evaluate(self):
		pass
