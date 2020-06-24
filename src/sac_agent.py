from tqdm import tqdm
from hyperparams_ur10 import OFF_POLICY_BATCH_SIZE as BATCH_SIZE, DISCOUNT, ENTROPY_WEIGHT, HIDDEN_SIZE, LEARNING_RATE, MAX_STEPS, POLYAK_FACTOR, REPLAY_SIZE, UPDATE_INTERVAL, UPDATE_START, SAVE_INTERVAL
from models_ur10_human import Critic, SoftActor, create_target_network, update_target_network

class SAC:

	def __init__(self):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
		self.rospy.Subscriber("observations", observations, observation.get_state)
		self.state = torch.tensor(observation.state).to(device)
		self.done = False
		self.reward = 0

		self.action_space = 2
		self.state_space = 8
		self.actor = SoftActor(HIDDEN_SIZE).to(device)
		self.critic_1 = Critic(HIDDEN_SIZE, state_action=True).to(device)
		self.critic_2 = Critic(HIDDEN_SIZE, state_action=True).to(device)
		self.value_critic = Critic(HIDDEN_SIZE).to(device)

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
		self.target_value_critic = create_target_network(value_critic).to(device)
		self.actor_optimiser = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
		self.critics_optimiser = optim.Adam(list(critic_1.parameters()) + list(critic_2.parameters()), lr=LEARNING_RATE)
		self.value_critic_optimiser = optim.Adam(value_critic.parameters(), lr=LEARNING_RATE)
		self.D = deque(maxlen=REPLAY_SIZE)
		# Automatic entropy tuning init
		self.target_entropy = -np.prod(action_space).item()
		self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
		self.alpha_optimizer = optim.Adam([log_alpha], lr=LEARNING_RATE)

	def next_action(self):
		pass

	def train(self):
		pass

	def evaluate(self):
		pass
