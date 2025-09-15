import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from typing import Dict

@dataclass
class PPOConfig:
    """Configuration for the PPO agent."""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    epochs: int = 10
    batch_size: int = 64
    hidden_size: int = 128
    num_lstm_layers: int = 1

class RolloutBuffer:
    """A buffer for storing trajectories for PPO."""
    def __init__(self):
        self.actions = []
        self.states = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        """Clears all stored trajectories."""
        del self.actions[:]
        del self.states[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.dones[:]
        del self.values[:]

class ActorCritic(nn.Module):
    """Actor-Critic network with an LSTM layer for PPO."""
    
    def __init__(self, state_size: int, action_size: int, config: PPOConfig):
        super(ActorCritic, self).__init__()
        self.config = config

        # LSTM layer to process sequences of states
        self.lstm = nn.LSTM(
            input_size=state_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_lstm_layers,
            batch_first=True
        )

        # Actor head: outputs a probability distribution over actions
        self.actor = nn.Sequential(
            nn.Linear(config.hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_size),
            nn.Softmax(dim=-1)
        )

        # Critic head: outputs the value of a state
        self.critic = nn.Sequential(
            nn.Linear(config.hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, state: torch.Tensor, hidden: tuple = None) -> (torch.Tensor, torch.Tensor, tuple):
        """
        Forward pass through the network.
        state shape: (batch_size, sequence_length, state_size)
        """
        lstm_out, hidden = self.lstm(state, hidden)
        
        # We only need the last output of the LSTM
        last_lstm_out = lstm_out[:, -1, :]
        
        action_probs = self.actor(last_lstm_out)
        state_value = self.critic(last_lstm_out)
        
        return action_probs, state_value, hidden

class LSTMPPOAgent:
    """PPO Agent with LSTM for thermal control."""

    def __init__(self, state_size: int, action_size: int, config: PPOConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy = ActorCritic(state_size, action_size, config).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        
        self.buffer = RolloutBuffer()
        self.hidden_state = None

    def select_action(self, state: np.ndarray) -> (int, float, float):
        """Selects an action based on the current policy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
            action_probs, state_value, self.hidden_state = self.policy(state_tensor, self.hidden_state)
            
            dist = Categorical(action_probs)
            action = dist.sample()
            
            action_log_prob = dist.log_prob(action)
        
        return action.item(), state_value.item(), action_log_prob.item()

    def update(self):
        """Updates the policy using the collected rollout."""
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.config.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalize rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Convert list to tensor
        old_states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        old_actions = torch.LongTensor(self.buffer.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        old_values = torch.FloatTensor(self.buffer.values).to(self.device)
        
        # Calculate advantages using GAE
        advantages = rewards - old_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.config.epochs):
            # Evaluating old actions and values
            # We need to process the whole sequence to get the final values
            action_probs, state_values, _ = self.policy(old_states.unsqueeze(1)) # Process as sequence of 1
            dist = Categorical(action_probs)
            
            log_probs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(log_probs - old_log_probs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.config.ppo_clip, 1 + self.config.ppo_clip) * advantages

            # Final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) \
                   + self.config.value_loss_coef * nn.MSELoss()(state_values.squeeze(), rewards) \
                   - self.config.entropy_coef * dist_entropy
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Clear buffer
        self.buffer.clear()
    
    def reset_hidden_state(self):
        self.hidden_state = None
