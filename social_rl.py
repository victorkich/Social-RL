# -*- coding: utf-8 -*-
from torch.distributions import Normal, MultivariateNormal
from collections import namedtuple, deque
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from PIL import Image
import numpy as np
import enlighten
import pygame
import random
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using: ", device)


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state):
        e = self.experience(state, action, reward, next_state)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        return states, actions, rewards, next_states

    def __len__(self):
        return len(self.memory)


class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden_size=32, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=5, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=5, stride=2, padding=0)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.mu = nn.Linear(hidden_size, action_size)
        self.log_std_linear = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        mu = self.mu(x)

        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def evaluate(self, state, epsilon=1e-6):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(0, 1)
        e = dist.sample().to(device)
        action = torch.tanh(mu + e * std)
        log_prob = Normal(mu, std).log_prob(mu + e * std) - torch.log(1 - action.pow(2) + epsilon)

        return action, log_prob

    def get_action(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(0, 1)
        e = dist.sample().to(device)
        action = torch.tanh(mu + e * std).cpu()
        return action[0]


class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden_size=32):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Environment:
    def __init__(self):
        pygame.init()
        self.x_boundary = 800
        self.y_boundary = 800

        self.x_jobs = [50, 50, 750, 750]
        self.y_jobs = [50, 750, 50, 750]
        self.job_radius = 50

        self.x_banks = [300, 300, 500, 500]
        self.y_banks = [300, 500, 300, 500]
        self.bank_radius = 40

        self.x_locks = [150, 400, 400, 650]
        self.y_locks = [400, 150, 650, 400]
        self.lock_radius = 30

        self.scr = pygame.display.set_mode((env.x_boundary, env.y_boundary))
        pygame.display.set_caption('Social RL')

    def render(self, agents, render=True):
        for agent in agents:
            if agent.x > self.x_boundary - agent.radius:
                agent.x = self.x_boundary - agent.radius
            elif agent.x < agent.radius:
                agent.x = agent.radius
            if agent.y > self.y_boundary - agent.radius:
                agent.y = self.y_boundary - agent.radius
            elif agent.y < agent.radius:
                agent.y = agent.radius

        if render:
            self.scr.fill((0, 0, 0))
            for i in range(4):
                pygame.draw.circle(self.scr, (0, 0, 200), (self.x_banks[i], self.y_banks[i]), self.bank_radius)
                pygame.draw.circle(self.scr, (0, 200, 0), (self.x_jobs[i], self.y_jobs[i]), self.job_radius)
                pygame.draw.circle(self.scr, (200, 200, 200), (self.x_locks[i], self.y_locks[i]), self.lock_radius)

            for agent in agents:
                pygame.draw.circle(self.scr, agent.color, (agent.x, agent.y), agent.radius)

            pygame.display.flip()

    def get_state(self):
        surface = pygame.Surface((100, 200))
        data = pygame.image.tostring(surface, 'RGBA')
        state = Image.fromstring('RGBA', (100, 200), data)
        return state

    def grab_event(self, agent):
        pass

    def release_event(self, agent):
        pass

    def finish(self):
        pygame.quit()


class Agent:
    def __init__(self, id, x_start, y_start, radius, color, random_seed, hidden_size=256, state_size=30, action_size=3, action_prior="uniform"):
        self.id = id
        self.x = x_start
        self.y = y_start
        self.radius = radius
        self.color = color
        self.batch_size = 256
        self.lr = 5e-4
        self.gamma = 0.99
        self.tau = 1e-2
        self.memory = int(2e5)

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        self.target_entropy = -action_size
        self.alpha = 1
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=self.lr)
        self._action_prior = action_prior

        # Actor Network
        self.actor_local = Actor(state_size, action_size, random_seed, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr)

        # Critic Network (w/ Target Network)
        self.critic1 = Critic(state_size, action_size, random_seed, hidden_size).to(device)
        self.critic2 = Critic(state_size, action_size, random_seed, hidden_size).to(device)

        self.critic1_target = Critic(state_size, action_size, random_seed, hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(state_size, action_size, random_seed, hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.lr, weight_decay=0)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.lr, weight_decay=0)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.memory, self.batch_size, random_seed)

    def get_action(self, state):
        state = torch.from_numpy(state).float().to(device)
        action = self.actor_local.get_action(state).detach()
        return action

    def step(self, action, env):
        reward = 0
        self.x += action[0]
        self.y += action[1]

        if action[2] >= 0.5:
            env.grab_event(self)
        elif action[2] <= -0.5:
            env.release_event(self)

        return reward

    def optimize(self, state, action, reward, next_state):
        self.memory.add(state, action, reward, next_state)
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma)

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        next_action, log_pis_next = self.actor_local.evaluate(next_states)

        Q_target1_next = self.critic1_target(next_states.to(device), next_action.squeeze(0).to(device))
        Q_target2_next = self.critic2_target(next_states.to(device), next_action.squeeze(0).to(device))

        # take the mean of both critics for updating
        Q_target_next = torch.min(Q_target1_next, Q_target2_next)
        Q_targets = rewards.cpu() + (gamma * (Q_target_next.cpu() - self.alpha * log_pis_next.squeeze(0).cpu()))

        # Compute critic loss
        Q_1 = self.critic1(states, actions).cpu()
        Q_2 = self.critic2(states, actions).cpu()
        critic1_loss = 0.5 * F.mse_loss(Q_1, Q_targets.detach())
        critic2_loss = 0.5 * F.mse_loss(Q_2, Q_targets.detach())
        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        alpha = torch.exp(self.log_alpha)
        # Compute alpha loss
        actions_pred, log_pis = self.actor_local.evaluate(states)
        alpha_loss = - (self.log_alpha.cpu() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = alpha
        # Compute actor loss
        if self._action_prior == "normal":
            policy_prior = MultivariateNormal(loc=torch.zeros(self.action_size), scale_tril=torch.ones(self.action_size).unsqueeze(0))
            policy_prior_log_probs = policy_prior.log_prob(actions_pred)
        elif self._action_prior == "uniform":
            policy_prior_log_probs = 0.0

        actor_loss = (alpha * log_pis.squeeze(0).cpu() - self.critic1(states, actions_pred.squeeze(0)).cpu() - policy_prior_log_probs).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic1, self.critic1_target, self.tau)
        self.soft_update(self.critic2, self.critic2_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


env = Environment()
agents = [Agent() for _ in range(4)]
running = True

frame_cap = 1.0 / 60
time_1 = time.perf_counter()
unprocessed = 0

state = env.get_state()
while running:
    can_render = False
    time_2 = time.perf_counter()
    passed = time_2 - time_1
    unprocessed += passed
    time_1 = time_2

    while unprocessed >= frame_cap:
        unprocessed -= frame_cap
        can_render = True

    if can_render:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        actions = []
        rewards = []
        for agent in agents:
            action = agent.get_action(state)
            reward = agent.step(action, env)
            actions.append(action)
            rewards.append(reward)
        env.render(agents)
        next_state = env.get_state()

        states = [state] * 4
        next_states = [next_state] * 4

        for agent, action, reward, next_state, done in zip(agents, actions, rewards, next_states):
            agent.optimize(state, action, reward, next_state)

        state = next_state
