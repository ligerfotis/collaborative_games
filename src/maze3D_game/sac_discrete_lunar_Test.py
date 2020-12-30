import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import gym
import torch.nn.functional as F

import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
https://github.com/EveLIn3/Discrete_SAC_LunarLander/blob/master/sac_discrete.py
"""
def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


# why retain graph? Do not auto free memory for one loss when computing multiple loss
# https://stackoverflow.com/questions/46774641/what-does-the-parameter-retain-graph-mean-in-the-variables-backward-method
def update_params(optim, loss):
    optim.zero_grad()
    loss.backward(retain_graph=True)
    optim.step()


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class ReplayBuffer:
    """
    Convert to numpy
    """
    def __init__(self, memory_size):
        self.storage = []
        self.memory_size = memory_size
        self.next_idx = 0

    # add the samples
    def add(self, obs, action, reward, obs_, done):
        data = (obs, action, reward, obs_, done)
        if self.next_idx >= len(self.storage):
            self.storage.append(data)
        else:
            self.storage[self.next_idx] = data
        # get the next idx
        self.next_idx = (self.next_idx + 1) % self.memory_size

    # encode samples
    def _encode_sample(self, idx):
        obses, actions, rewards, obses_, dones = [], [], [], [], []
        for i in idx:
            data = self.storage[i]
            obs, action, reward, obs_, done = data
            obses.append(np.array(obs, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_.append(np.array(obs_, copy=False))
            dones.append(done)
        return np.array(obses), np.array(actions), np.array(rewards), np.array(obses_), np.array(dones)

    # sample from the memory
    def sample(self, batch_size):
        idxes = [random.randint(0, len(self.storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, n_hidden_units):
        super(Actor, self).__init__()

        self.actor_mlp = nn.Sequential(
            nn.Linear(state_dim, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, action_dim)
        ).apply(init_weights)

    def forward(self, s):
        actions_logits = self.actor_mlp(s)
        return F.softmax(actions_logits, dim=-1)

    def greedy_act(self, s):  # no softmax more efficient
        s = torch.from_numpy(s).float().to(device)
        actions_logits = self.actor_mlp(s)
        greedy_actions = torch.argmax(actions_logits, dim=-1, keepdim=True)
        return greedy_actions.item()

    def sample_act(self, s):
        s = torch.from_numpy(s).float().to(device)
        actions_logits = self.actor_mlp(s)
        actions_probs = F.softmax(actions_logits, dim=-1)
        actions_distribution = Categorical(actions_probs)
        action = actions_distribution.sample()
        return action.item()


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, n_hidden_units):
        super(Critic, self).__init__()

        self.qnet1 = DuelQNet(state_dim, action_dim, n_hidden_units)
        self.qnet2 = DuelQNet(state_dim, action_dim, n_hidden_units)

    def forward(self, s):  # S: N x F(state_dim) -> Q: N x A(action_dim) Q(s,a)
        q1 = self.qnet1(s)
        q2 = self.qnet2(s)
        return q1, q2


class DuelQNet(nn.Module):
    def __init__(self, state_dim, action_dim, n_hidden_units):
        super(DuelQNet, self).__init__()

        self.shared_mlp = nn.Sequential(
            nn.Linear(state_dim, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU()
        ).apply(init_weights)

        # self.q_head = nn.Linear(n_hidden_units, action_dim)

        self.action_head = nn.Linear(n_hidden_units, action_dim).apply(init_weights)
        self.value_head = nn.Linear(n_hidden_units, 1).apply(init_weights)

    def forward(self, s):
        s = self.shared_mlp(s)
        a = self.action_head(s)
        v = self.value_head(s)
        return v + a - a.mean(1, keepdim=True)
        # return self.q_head(s)


class DiscreteSACAgent:
    def __init__(self, state_dim, action_dim, n_hidden_units, lr=0.001, memory_size=50000, gamma=0.99,
                 target_entropy_ratio=0.4,
                 update_interval=1, target_update_interval=300, num_eval_steps=1000, max_episode_steps=300,
                 eval_interval=10,
                 memory_batch_size=256, tau=0.005):
        self.lr = lr
        self.gamma = gamma
        self.tau = tau

        self.memory_size = memory_size
        self.target_entropy_ratio = target_entropy_ratio
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        self.num_eval_steps = num_eval_steps
        self.max_episode_steps = max_episode_steps
        self.eval_interval = eval_interval
        self.memory_batch_size = memory_batch_size

        self.actor = Actor(state_dim, action_dim, n_hidden_units).to(device)
        self.critic = Critic(state_dim, action_dim, n_hidden_units).to(device)
        self.target_critic = Critic(state_dim, action_dim, n_hidden_units).to(device)

        self.target_critic.load_state_dict(self.critic.state_dict())
        # self.soft_update_target()

        # disable gradient for target critic
        # for param in self.target_critic.parameters():
        #     param.requires_grad = False

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr, eps=1e-4)
        self.critic_q1_optim = torch.optim.Adam(self.critic.qnet1.parameters(), lr=lr, eps=1e-4)
        self.critic_q2_optim = torch.optim.Adam(self.critic.qnet2.parameters(), lr=lr, eps=1e-4)

        # target -> maximum entropy (same prob for each action)
        # - log ( 1 / A) = log A
        # self.target_entropy = -np.log(1.0 / action_dim) * self.target_entropy_ratio
        # self.target_entropy = np.log(action_dim) * self.target_entropy_ratio
        self.target_entropy = target_entropy_ratio

        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr, eps=1e-4)

        self.memory = ReplayBuffer(memory_size)

    def learn(self):
        states, actions, rewards, states_, dones = self.memory.sample(self.memory_batch_size)
        # states, actions, rewards, states_, dones = self.memory.sample(memory_batch_size)
        states = torch.from_numpy(states).float().to(device)
        states_ = torch.from_numpy(states_).float().to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device).unsqueeze(1)  # dim [Batch,] -> [Batch, 1]
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        rewards = torch.tensor(rewards).float().to(device)
        dones = torch.tensor(dones).float().to(device)

        batch_transitions = states, actions, rewards, states_, dones

        weights = 1.  # default
        q1_loss, q2_loss, errors, mean_q1, mean_q2 = self.calc_critic_loss(batch_transitions, weights)
        policy_loss, entropies = self.calc_policy_loss(batch_transitions, weights)
        entropy_loss = self.calc_entropy_loss(entropies, weights)

        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5)
        update_params(self.critic_q1_optim, q1_loss)
        update_params(self.critic_q2_optim, q2_loss)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5)
        update_params(self.actor_optim, policy_loss)
        update_params(self.alpha_optim, entropy_loss)

        return mean_q1, mean_q2, entropies

    def update_target(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def soft_update_target(self):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        curr_q1, curr_q2 = self.critic(states)
        curr_q1 = curr_q1.gather(1, actions)  # select the Q corresponding to chosen A
        curr_q2 = curr_q2.gather(1, actions)
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            action_probs = self.actor(next_states)
            z = (action_probs == 0.0).float() * 1e-8  # for numerical stability
            log_action_probs = torch.log(action_probs + z)

            next_q1, next_q2 = self.target_critic(next_states)
            # next_q = (action_probs * (
            #     torch.min(next_q1, next_q2) - self.alpha * log_action_probs
            # )).mean(dim=1).view(self.memory_batch_size, 1) # E = probs T . values

            alpha = self.log_alpha.exp()
            next_q = action_probs * (torch.min(next_q1, next_q2) - alpha * log_action_probs)
            next_q = next_q.sum(dim=1)

            target_q = rewards + (1 - dones) * self.gamma * (next_q)
            return target_q.unsqueeze(1)

        # assert rewards.shape == next_q.shape
        # return rewards + (1.0 - dones) * self.gamma * next_q

    def calc_critic_loss(self, batch, weights):
        target_q = self.calc_target_q(*batch)

        # TD errors for updating priority weights
        # errors = torch.abs(curr_q1.detach() - target_q)
        errors = None
        mean_q1, mean_q2 = None, None

        # We log means of Q to monitor training.
        # mean_q1 = curr_q1.detach().mean().item()
        # mean_q2 = curr_q2.detach().mean().item()

        # Critic loss is mean squared TD errors with priority weights.
        # q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        # q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        q1_loss = F.mse_loss(curr_q1, target_q)
        q2_loss = F.mse_loss(curr_q2, target_q)

        return q1_loss, q2_loss, errors, mean_q1, mean_q2

    def calc_policy_loss(self, batch, weights):
        states, actions, rewards, next_states, dones = batch

        # (Log of) probabilities to calculate expectations of Q and entropies.
        action_probs = self.actor(states)
        z = (action_probs == 0.0).float() * 1e-8  # for numerical stability
        log_action_probs = torch.log(action_probs + z)

        # with torch.no_grad():
        # Q for every actions to calculate expectations of Q.
        # q1, q2 = self.critic(states)
        # q = torch.min(q1, q2)

        q1, q2 = self.critic(states)

        alpha = self.log_alpha.exp()
        # minq = torch.min(q1, q2)
        # inside_term = alpha * log_action_probs - minq
        # policy_loss = (action_probs * inside_term).mean()

        # Expectations of entropies.
        entropies = - torch.sum(action_probs * log_action_probs, dim=1)
        # Expectations of Q.
        q = torch.sum(torch.min(q1, q2) * action_probs, dim=1, keepdim=True)

        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        policy_loss = (weights * (- q - alpha * entropies)).mean()  # avg over Batch

        return policy_loss, entropies

    def calc_entropy_loss2(self, pi_s, log_pi_s):
        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        alpha = self.log_alpha.exp()
        inside_term = - alpha * (log_pi_s + self.target_entropy).detach()
        entropy_loss = (pi_s * inside_term).mean()
        return entropy_loss

    def calc_entropy_loss(self, entropies, weights):
        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropies).detach()
            * weights)
        return entropy_loss


def main():
    ############## Hyperparameters ##############
    env_name = "LunarLander-v2"
    action_dim = 4
    n_hidden_units = 64  # number of variables in hidden layer
    lr = 0.002
    memory_size = 15000
    learn_every_n_steps = 1
    n_epoch_every_update = 1
    target_entropy = 0.4  # torch.prod(env.action_space.n) # KEY!
    max_timesteps = 500  # max timesteps in one episode

    # env_name = "LunarLander-v2" # working in 200 episodes
    # action_dim = 4
    # n_hidden_units = 128         # number of variables in hidden layer
    # lr = 0.002
    # memory_size = 15000
    # learn_every_n_steps = 1
    # n_epoch_every_update = 1
    # target_entropy = 0.4 # torch.prod(env.action_space.n) # KEY!

    # env_name = 'CartPole-v0'
    # action_dim = 2
    # lr = 0.0003
    # memory_size = 50000
    # learn_every_n_steps = 1
    # n_epoch_every_update = 1
    # n_hidden_units = 64         # number of variables in hidden layer
    # max_timesteps = 300         # max timesteps in one episode
    # target_entropy = 0.98 # torch.prod(env.action_space.n) # KEY!

    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    print("Action Space: " + str(env.action_space))
    print("Reward Threshold: " + str(env.spec.reward_threshold))
    print("Action Space Size: " + str(env.action_space.n))

    render = True
    solved_reward = 230  # stop training if avg_reward > solved_reward
    log_interval = 20  # print avg reward in the interval
    max_episodes = 1600  # max training episodes

    update_timestep = 4  # update actor critic every n timesteps

    gamma = 0.99  # discount factor
    tau = 0.005
    random_seed = None
    #############################################

    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    sac = DiscreteSACAgent(state_dim, action_dim, n_hidden_units, lr=lr, gamma=gamma, tau=tau, memory_size=memory_size,
                           update_interval=learn_every_n_steps)

    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0
    total_steps = 0
    start_training_step = 400
    mean_q1, mean_q2, entropies = 0, 0, 0
    training_epochs_per_update = 128

    # training loop
    for i_episode in range(1, max_episodes + 1):
        state = env.reset()
        for t in range(max_timesteps):
            timestep += 1
            total_steps += 1

            # if total_steps < start_training_step:  # Pure exploration
            #     action = random.randint(0, action_dim - 1)
            # else:  # Explore with actions_prob
            #     action = sac.actor.sample_act(state)
            action = sac.actor.sample_act(state)
            """
            Add the human part here
            """


            # with torch.no_grad():
            #     action = sac.actor.sample_act(state)
            next_state, reward, done, _ = env.step(action)

            # clipped reward in [-1.0, 1.0]
            # clipped_reward = max(min(reward, 1.0), -1.0)
            clipped_reward = reward

            if render:
                env.render()

            sac.memory.add(state, action, clipped_reward, next_state, done)
            state = next_state

            if total_steps >= start_training_step and total_steps % sac.update_interval == 0:
                for e in range(n_epoch_every_update):
                    mean_q1, mean_q2, entropies = sac.learn()
                    sac.soft_update_target()

            # if total_steps >= start_training_step and total_steps % sac.target_update_interval == 0:
            #     sac.soft_update_target()

            running_reward += reward
            # if render:
            #     env.render()
            if done:
                break

        # for e in range(training_epochs_per_update):
        #     sac.learn()
        #     sac.soft_update_target()

        avg_length += t

        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval * solved_reward):
            print("########## Solved! ##########")
            torch.save(sac.actor.state_dict(), './SAC_ACTOR2{}.pth'.format(env_name))
            break

        # logging
        if i_episode % log_interval == 0:
            print("mean_q1 {}  mean_q2 {}  entropy {}".format(mean_q1, mean_q2, 0))
            avg_length = int(avg_length / log_interval)
            running_reward = int((running_reward / log_interval))

            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0


if __name__ == '__main__':
    main()
    exit(0)
