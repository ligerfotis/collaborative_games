#!/usr/bin/env python
import rospy
import gym
import std_msgs
from std_msgs.msg import Int16, Float32
from environment import Game 
from keypoint_3d_matching_msgs.msg import Keypoint3d_list
from collaborative_games.msg import observation, action_agent, reward_observation, action_msg
from std_srvs.srv import Empty,EmptyResponse, Trigger
import time
from statistics import mean 

class controller:

	def __init__(self):
		print("init")
		self.experiments_num = 10
		# self.human_sub = rospy.Subscriber("/RW_x_direction", Int16, self.simulate)
		self.game = Game()
		self.action_human = 0.0
		self.action_agent = 0.0
		# # self.human_sub = rospy.Subscriber("/rl/hand_action_x", action_msg, self.set_action_human)
		# self.human_sub = rospy.Subscriber("/rl/action_x", action_msg, self.set_action_human)
		self.agent_sub = rospy.Subscriber("/rl/total_action", action_agent, 
			self.set_action_agent)

		# self.reward_pub = rospy.Publisher('/rl/reward', Int16, queue_size = 10)
		self.obs_robot_pub = rospy.Publisher('/rl/game_response', reward_observation, 
			queue_size = 10, latch=True)
		
		self.transmit_time_list = []

		self.rewards_list = []
		self.turn_list = []
		self.interaction_time_list = []
		self.interaction_training_time_list = []
		self.mean_list = []
		self.stdev_list = []
		self.alpha_values = []
		self.policy_loss_list = []
		self.value_loss_list = []
		self.critics_lr_list = []
		self.value_critic_lr_list = []
		self.actor_lr_list = []

		self.plot_directory = package_path + "/src/plots/"
		if not os.path.exists(self.plot_directory):
			print("Dir %s was not found. Creating it..." %(self.plot_directory))
			os.makedirs(self.plot_directory)

	def set_action_agent(self, action_agent):
		self.prev_state = self.game.getState()
		self.action_agent = action_agent.action[1]
		self.action_human = action_agent.action[0]
		self.transmit_time_list.append(rospy.get_rostime().to_sec()  - action_agent.header.stamp.to_sec())

	def set_human_agent(self, action_agent):
		if action_agent.action != 0.0:
			self.action_human = action_agent.action
			self.transmit_time_list.append(rospy.get_rostime().to_sec()  - action_agent.header.stamp.to_sec())


	def publish_reward_and_observations(self):
		h = std_msgs.msg.Header()
		h.stamp = rospy.Time.now() 

		new_obs = reward_observation()
		new_obs.header = h
		new_obs.observations = self.game.getState()
		new_obs.prev_state = self.prev_state
		new_obs.final_state = self.game.finished
		new_obs.reward = self.game.getReward()

		self.obs_robot_pub.publish(new_obs)

	def game_loop(self):
		first_update = True
		global_time = rospy.get_rostime().to_sec()

		self.resetGame()

		for exp in range(MAX_STEPS+1):

			if self.game.running:
				start_interaction_time = time.time()

				self.game.experiment = exp
				self.turns += 1

				state = self.game.getState()
				agent_act = self.agent.next_action(state)

				tmp_time = time.time()
				act_human = self.action_human

				while time.time() - tmp_time < action_duration :
					exec_time = self.game.play([act_human, agent_act.item()])
					
				reward = self.game.getReward()
				next_state = self.game.getState()
				done = self.game.finished
				episode = [state, reward, agent_act, next_state, done]

				self.agent.update_rw_state(episode)

				self.total_reward_per_game += reward 

				self.interaction_time_list.append(time.time() - start_interaction_time)

				# when replay buffer has enough samples update gradient at every turn
				if len(self.agent.D) >= BATCH_SIZE:

					if first_update:
						print("\nStarting updates")
						first_update = False

					[alpha, policy_loss, value_loss, critics_lr, value_critic_lr, actor_lr] = self.agent.train(sample=episode)
					self.alpha_values.append(alpha.item())
					self.policy_loss_list.append(policy_loss.item())
					self.value_loss_list.append(value_loss.item())
					self.critics_lr_list.append(critics_lr)
					self.value_critic_lr_list.append(value_critic_lr)
					self.actor_lr_list.append(actor_lr)

					self.interaction_training_time_list.append(time.time() - start_interaction_time)

				# run "offline_updates_num" offline gradient updates every "UPDATE_INTERVAL" steps
				if len(self.agent.D) >= BATCH_SIZE and exp % UPDATE_INTERVAL == 0:

					print(str(offline_updates_num) + " Gradient upadates")
					self.game.waitScreen("Training... Please Wait.")

					pbar = tqdm(xrange(1, offline_updates_num + 1), unit_scale=1, smoothing=0)
					for _ in pbar:
						[alpha, policy_loss, value_loss, critics_lr, value_critic_lr, actor_lr] = self.agent.train(verbose=False)
						self.alpha_values.append(alpha.item())
						self.policy_loss_list.append(policy_loss.item())
						self.value_loss_list.append(value_loss.item())
						self.critics_lr_list.append(critics_lr)
						self.value_critic_lr_list.append(value_critic_lr)
						self.actor_lr_list.append(actor_lr)

					# run trials
					mean_score, stdev_score =  self.test()

					self.mean_list.append(mean_score)
					self.stdev_list.append(stdev_score)

					self.resetGame()
			else:
				self.turn_list.append(self.turns)
				self.rewards_list.append(self.total_reward_per_game)
				
				# reset game
				self.resetGame()

		self.save_and_plot_stats()
		
		
		print("Average Execution time per interaction: %f milliseconds(stdev: %f). \n" % (mean(self.interaction_time_list) * 1e3, stdev(self.interaction_time_list) * 1e3))
		print("Average Execution time per interaction and online update: %f milliseconds(stdev: %f). \n" % (mean(self.interaction_training_time_list) * 1e3, stdev(self.interaction_training_time_list) * 1e3))

		print("Total time of experiments is: %d minutes and %d seconds.\n" % ( ( rospy.get_rostime().to_sec() - global_time )/60, ( rospy.get_rostime().to_sec() - global_time )%60 )  )
		
		self.game.endGame()

	def test(self):

		score_list = []
		for game in range(test_num):
			score = 200

			self.resetGame("Testing Model. Trial %d of %d." % (game+1,test_num))

			while self.game.running:
				self.game.experiment = "Test: " + str(game+1)

				state = self.game.getState()
				agent_act = self.agent.next_action(state, stochastic=False) # take only the mean
				# print(agent_act)
				tmp_time = time.time()
				while time.time() - tmp_time < 0.2 :
					exec_time = self.game.play([self.action_human, agent_act.item()], total_games=test_num)

				score -= 1

			score_list.append(score)

		return [mean(score_list), stdev(score_list)]

	def resetGame(self, msg=None):
		wait_time = 3
		self.game.waitScreen(msg1="Put Right Wrist on starting point.", msg2=msg, duration=wait_time)
		self.game = Game()
		self.action_human = 0.0
		self.action_agent = 0.0
		self.game.start_time = time.time()
		self.total_reward_per_game = 0
		self.turns = 0

	def save_and_plot_stats(self):
		np.savetxt(self.plot_directory + 'alpha_values.csv', self.alpha_values, delimiter=',', fmt='%f')
		np.savetxt(self.plot_directory + 'policy_loss.csv', self.policy_loss_list, delimiter=',', fmt='%f')
		np.savetxt(self.plot_directory + 'value_loss.csv', self.value_loss_list, delimiter=',', fmt='%f')
		np.savetxt(self.plot_directory + 'rewards_list.csv', self.rewards_list, delimiter=',', fmt='%f')
		np.savetxt(self.plot_directory + 'turn_list.csv', self.turn_list, delimiter=',', fmt='%f')
		np.savetxt(self.plot_directory + 'means.csv', self.mean_list, delimiter=',', fmt='%f')		
		np.savetxt(self.plot_directory + 'stdev.csv', self.stdev_list, delimiter=',', fmt='%f')
		np.savetxt(self.plot_directory + 'critics_lr_list.csv', self.critics_lr_list, delimiter=',', fmt='%f')
		np.savetxt(self.plot_directory + 'value_critic_lr_list.csv', self.value_critic_lr_list, delimiter=',', fmt='%f')
		np.savetxt(self.plot_directory + 'actor_lr_list.csv', self.actor_lr_list, delimiter=',', fmt='%f')


		plot(range(len(self.alpha_values)), self.alpha_values, "alpha_values", 'Alpha Value', 'Number of Gradient Updates', self.plot_directory, save=True)
		plot(range(len(self.policy_loss_list)), self.policy_loss_list, "policy_loss", 'Policy loss', 'Number of Gradient Updates', self.plot_directory, save=True)
		plot(range(len(self.value_loss_list)), self.value_loss_list, "value_loss_list", 'Value loss', 'Number of Gradient Updates', self.plot_directory, save=True)
		plot(range(len(self.rewards_list)), self.rewards_list, "Rewards_per_game", 'Total Rewards per Game', 'Number of Games', self.plot_directory, save=True)
		plot(range(len(self.turn_list)), self.turn_list, "Steps_per_game", 'Turns per Game', 'Number of Games', self.plot_directory, save=True)	

		plot(range(len(self.critics_lr_list)), self.critics_lr_list, "critics_lr_list", 'Critic lr', 'Number of Gradient Updates', self.plot_directory, save=True)
		plot(range(len(self.value_critic_lr_list)), self.value_critic_lr_list, "value_critic_lr_list", 'Value lr', 'Number of Gradient Updates', self.plot_directory, save=True)
		plot(range(len(self.actor_lr_list)), self.actor_lr_list, "actor_lr_list", 'Actor lr', 'Number of Gradient Updates', self.plot_directory, save=True)		

		plot(range(UPDATE_INTERVAL, MAX_STEPS+UPDATE_INTERVAL, UPDATE_INTERVAL), self.mean_list, "trials", 'Tests Score', 'Number of Interactions', self.plot_directory, save=True, variance=True, stdev=self.stdev_list)		


if __name__ == '__main__':
	rospy.init_node('rl_wrapper', anonymous=True)
	ctrl = controller()
	ctrl.game.game_intro()

	ctrl.game_loop()
	# while ctrl.game.running:
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")
