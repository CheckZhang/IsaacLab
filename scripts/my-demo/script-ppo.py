import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import gym
import numpy as np

# Hyperparameters
env_name = "Pendulum-v1"
gamma = 0.99  # Discount factor
lambda_gae = 0.95  # GAE parameter
clip_epsilon = 0.2  # PPO clipping parameter
epochs = 10  # Number of updates after each data collection
batch_size = 64  # Batch size for each update
lr_actor = 3e-4  # Learning rate for the actor network
lr_critic = 1e-3  # Learning rate for the critic network
max_episodes = 1000  # Maximum number of training episodes
max_steps = 200  # Maximum steps per episode
update_interval = 2000  # Number of steps collected before each update

# Actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.mu(x))  
        log_std = self.log_std(x)  
        return mu, log_std

# Critic network
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.value(x)

# PPO algorithm
class PPO:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
    
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        mu, log_std = self.actor(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        #return action.numpy()[0], log_prob.numpy()[0]
        return action.detach().numpy()[0], log_prob.detach().numpy()[0]

    def update(self, states, actions, log_probs_old, returns, advantages):
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        log_probs_old = torch.FloatTensor(log_probs_old)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)

        for _ in range(epochs):
            for idx in range(0, len(states), batch_size):
                batch_states = states[idx:idx + batch_size]
                batch_actions = actions[idx:idx + batch_size]
                batch_log_probs_old = log_probs_old[idx:idx + batch_size]
                batch_returns = returns[idx:idx + batch_size]
                batch_advantages = advantages[idx:idx + batch_size]

                # Calculate new action probabilities
                mu, log_std = self.actor(batch_states)
                std = log_std.exp()
                dist = Normal(mu, std)
                log_probs_new = dist.log_prob(batch_actions).sum(-1, keepdim=True)

                # Calculate importance weights
                ratio = (log_probs_new - batch_log_probs_old).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Update actor network
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update critic network
                values = self.critic(batch_states)
                critic_loss = F.mse_loss(values, batch_returns)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

# Compute GAE and returns
def compute_gae(rewards, values, dones, next_value):
    gae = 0
    returns = []
    advantages = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_value * (1 - dones[step]) - values[step]
        gae = delta + gamma * lambda_gae * (1 - dones[step]) * gae
        next_value = values[step]
        returns.insert(0, gae + values[step])
        advantages.insert(0, gae)
    return returns, advantages

# Main training loop
def train():
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    ppo = PPO(state_dim, action_dim)

    episode_rewards = []
    state = env.reset()
    states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

    for episode in range(max_episodes):
        episode_reward = 0
        for step in range(max_steps):
            action, log_prob = ppo.get_action(state)
            next_state, reward, done, _ = env.step(action)
            value = ppo.critic(torch.FloatTensor(state).unsqueeze(0)).item()

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            values.append(value)

            state = next_state
            episode_reward += reward

            if len(states) == update_interval or done:
                next_value = ppo.critic(torch.FloatTensor(next_state).unsqueeze(0)).item()
                returns, advantages = compute_gae(rewards, values, dones, next_value)
                ppo.update(states, actions, log_probs, returns, advantages)
                states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

            if done:
                break

        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}, Reward: {episode_reward}")

    env.close()

if __name__ == "__main__":
    train()