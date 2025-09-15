import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import project modules
from thermal_environment import ThermalBuildingEnv, ThermalEnvManager
from lstm_ppo_agent import LSTMPPOAgent, PPOConfig # <-- CHANGED
from baseline_controllers import PIDController, SimpleThermostat

def main():
    """Main script to train and evaluate the agent"""
    
    print("=== INTELLIGENT THERMAL LEARNING SYSTEM (PPO) ===\n")
    
    # --- 1. Configuration ---
    with open("config_dashboard.json", "r") as f:
        config_data = json.load(f)
    
    # PPO specific config
    ppo_config = PPOConfig(
        learning_rate=config_data['agent'].get('learning_rate', 3e-4),
        gamma=config_data['agent'].get('gamma', 0.99),
        epochs=10,
        ppo_clip=0.2,
        hidden_size=config_data['agent'].get('hidden_size', 128)
    )
    
    # Training settings
    training_episodes = config_data['training'].get('episodes', 500)
    max_steps_per_episode = config_data['training'].get('max_steps_per_episode', 168)
    update_timestep = 2048 # Update policy after this many steps

    # --- 2. Initialization ---
    env = ThermalBuildingEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = LSTMPPOAgent(state_size, action_size, ppo_config)
    
    # --- 3. Training Loop ---
    logging.info("Starting PPO training...")
    start_time = datetime.now()
    
    time_step = 0
    episode_rewards = []

    for episode in range(1, training_episodes + 1):
        state = env.reset()
        agent.reset_hidden_state() # Reset LSTM state at the start of each episode
        current_episode_reward = 0

        for t in range(1, max_steps_per_episode + 1):
            time_step += 1
            
            # Select action
            action, value, log_prob = agent.select_action(state)
            
            # Store data in buffer
            agent.buffer.states.append(state)
            agent.buffer.actions.append(action)
            agent.buffer.values.append(value)
            agent.buffer.log_probs.append(log_prob)
            
            # Step the environment
            state, reward, done, _ = env.step(action)
            
            agent.buffer.rewards.append(reward)
            agent.buffer.dones.append(done)
            
            current_episode_reward += reward

            # Update if buffer is full
            if time_step % update_timestep == 0:
                agent.update()

            if done:
                break
        
        episode_rewards.append(current_episode_reward)

        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            logging.info(f"Episode {episode} | Avg Reward (last 10): {avg_reward:.2f}")

    training_duration = (datetime.now() - start_time).total_seconds()
    logging.info(f"Training finished in {training_duration:.1f}s")
    
    # --- 4. Save Model & Results ---
    os.makedirs("./models", exist_ok=True)
    torch.save(agent.policy.state_dict(), './models/ppo_lstm_policy.pth')
    
    # --- Optional: Evaluation ---
    # The BenchmarkEvaluator can be adapted to work with the PPO agent's select_action method.

if __name__ == "__main__":
    main()
