"""
Deep Learning Agent for playing the game through websocket connection.
"""

import json
import numpy as np
import time
import random
import os
import pickle
from collections import deque
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ReplayBuffer:
    """A replay buffer for experience replay in reinforcement learning."""
    
    def __init__(self, capacity: int):
        """Initialize replay buffer with a fixed capacity."""
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add a new experience to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample a batch of experiences from the buffer."""
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)


class DQNNetwork(nn.Module):
    """Deep Q-Network for action selection."""
    
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 128):
        """Initialize network architecture."""
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class GameAgent:
    """Deep Learning Agent for playing the game."""
    
    # Define actions
    ACTION_NONE = 0      # Don't move
    ACTION_UP = 1        # Move up
    ACTION_DOWN = 2      # Move down
    ACTION_LEFT = 3      # Move left
    ACTION_RIGHT = 4     # Move right
    ACTION_UP_LEFT = 5   # Move diagonally up-left
    ACTION_UP_RIGHT = 6  # Move diagonally up-right
    ACTION_DOWN_LEFT = 7 # Move diagonally down-left
    ACTION_DOWN_RIGHT = 8 # Move diagonally down-right
    
    ACTIONS = {
        ACTION_NONE: {"x": 0, "y": 0},
        ACTION_UP: {"x": 0, "y": -1},
        ACTION_DOWN: {"x": 0, "y": 1},
        ACTION_LEFT: {"x": -1, "y": 0},
        ACTION_RIGHT: {"x": 1, "y": 0},
        ACTION_UP_LEFT: {"x": -1, "y": -1},
        ACTION_UP_RIGHT: {"x": 1, "y": -1},
        ACTION_DOWN_LEFT: {"x": -1, "y": 1},
        ACTION_DOWN_RIGHT: {"x": 1, "y": 1}
    }
    
    def __init__(self, model_dir: str = "models"):
        """Initialize the agent with given parameters."""
        try:
            print("Initializing GameAgent")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")
            
            # State and action variables
            self.state_size = 17  # Player stats + game stats + nearest enemy info
            self.action_size = len(self.ACTIONS)
            
            # Learning parameters
            self.gamma = 0.99  # Discount factor
            self.epsilon = 1.0  # Exploration rate
            self.epsilon_min = 0.1  # Minimum exploration rate
            self.epsilon_decay = 0.995  # Exploration decay rate
            self.learning_rate = 0.001  # Learning rate
            self.batch_size = 32  # Batch size for training
            self.update_target_every = 100  # Update target network every N episodes
            
            # Create model directory if it doesn't exist
            self.model_dir = model_dir
            os.makedirs(self.model_dir, exist_ok=True)
            print(f"Model directory: {self.model_dir}")
            
            print("Building neural networks...")
            # Build models
            self.policy_net = DQNNetwork(self.state_size, self.action_size).to(self.device)
            self.target_net = DQNNetwork(self.state_size, self.action_size).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()  # Target network is only used for inference
            print("Neural networks built successfully")
            
            # Optimizer
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
            
            # Experience replay buffer
            self.memory = ReplayBuffer(10000)
            
            # Training variables
            self.episode_count = 0
            self.step_count = 0
            self.current_state = None
            self.last_action = None
            self.cumulative_reward = 0
            
            # Game state variables
            self.player_pos = {"x": 0, "y": 0}
            self.player_health = {"current": 0, "max": 0}
            self.game_wave = 0
            self.last_game_wave = 0
            self.nearest_enemy_distance = float('inf')
            self.last_nearest_enemy_distance = float('inf')
            self.enemies_killed_count = 0
            self.last_health = 0
            self.last_game_state = None
            self.game_over = False
            
            # Try to load existing model
            print("Attempting to load existing model")
            self.load_model()
            print("GameAgent initialization complete")
        except Exception as e:
            import traceback
            print(f"Error initializing GameAgent: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            # Initialize with defaults
            self._initialize_defaults()
    
    def reset(self):
        """Reset agent state at the beginning of a new episode."""
        self.current_state = None
        self.last_action = None
        self.cumulative_reward = 0
        self.game_over = False
        self.last_health = 0
        self.last_nearest_enemy_distance = float('inf')
        self.enemies_killed_count = 0
        self.last_game_wave = 0
        self.last_game_state = None
    
    def _initialize_defaults(self):
        """Initialize with safe defaults in case of error"""
        self.device = torch.device("cpu")
        self.state_size = 17
        self.action_size = len(self.ACTIONS)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.update_target_every = 100
        self.model_dir = "models"
        
        # Create minimal models to prevent crashes
        self.policy_net = DQNNetwork(self.state_size, self.action_size).to(self.device)
        self.target_net = DQNNetwork(self.state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(1000) # Smaller buffer size
        
        # Training variables
        self.episode_count = 0
        self.step_count = 0
        self.current_state = None
        self.last_action = None
        self.cumulative_reward = 0
        
        # Game state variables
        self.player_pos = {"x": 0, "y": 0}
        self.player_health = {"current": 0, "max": 0}
        self.game_wave = 0
        self.last_game_wave = 0
        self.nearest_enemy_distance = float('inf')
        self.last_nearest_enemy_distance = float('inf')
        self.enemies_killed_count = 0
        self.last_health = 0
        self.last_game_state = None
        self.game_over = False
        
    def process_game_data(self, game_data: str) -> Dict:
        """Process game data received from WebSocket."""
        try:
            data = json.loads(game_data)
            
            # Update internal game state
            self.player_pos = data["player"]["position"]
            self.player_health = data["player"]["health"]
            self.game_wave = data["game"]["wave"]
            
            # Track wave changes (for rewards)
            if self.game_wave > self.last_game_wave:
                self.last_game_wave = self.game_wave
            
            # Process nearby enemies
            self.nearest_enemy_distance = float('inf')
            if "nearby" in data and "enemies" in data["nearby"] and len(data["nearby"]["enemies"]) > 0:
                for enemy in data["nearby"]["enemies"]:
                    if enemy["distance"] < self.nearest_enemy_distance:
                        self.nearest_enemy_distance = enemy["distance"]
            
            # Update state and calculate rewards
            current_state = self._get_state_representation(data)
            reward = self._calculate_reward(data)
            
            # Handle state transitions
            if self.current_state is not None and self.last_action is not None:
                # Store experience in replay buffer
                self.memory.push(
                    self.current_state, 
                    self.last_action, 
                    reward, 
                    current_state, 
                    self.game_over
                )
                
                # Learn from experiences
                self.learn()
            
            # Update current state
            self.current_state = current_state
            self.last_game_state = data
            
            # Training statistics
            self.step_count += 1
            self.cumulative_reward += reward
            
            # Get next action
            action = self.get_action()
            self.last_action = action
            
            return self.ACTIONS[action]
            
        except Exception as e:
            import traceback
            print(f"Error processing game data: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return {"x": 0, "y": 0}  # Default - don't move
    
    def _get_state_representation(self, data: Dict) -> np.ndarray:
        """Convert game data to a state representation for the agent."""
        state = []
        
        # Player position and normalization factors (based on assumed game area)
        x_norm = data["player"]["position"]["x"] / 2000.0
        y_norm = data["player"]["position"]["y"] / 2000.0
        state.extend([x_norm, y_norm])
        
        # Player health (current and max, normalized)
        health_current = data["player"]["health"]["current"]
        health_max = data["player"]["health"]["max"]
        state.extend([health_current / max(1, health_max), health_max / 100.0])
        
        # Player speed (normalized)
        state.append(data["player"]["speed"] / 1000.0)
        
        # Player gold and level (normalized)
        state.extend([data["player"]["gold"] / 1000.0, data["player"]["level"] / 10.0])
        
        # Game state
        state.extend([
            data["game"]["wave"] / 50.0,
            data["game"]["wave_timer"] / data["game"]["wave_duration"],
            data["game"]["enemies_count"] / 50.0
        ])
        
        # Nearby enemies
        nearest_enemy_pos = {"x": 0, "y": 0}
        nearest_enemy_health = {"current": 0, "max": 0}
        nearest_enemy_distance = float('inf')
        
        if "nearby" in data and "enemies" in data["nearby"] and len(data["nearby"]["enemies"]) > 0:
            # Find the nearest enemy
            for enemy in data["nearby"]["enemies"]:
                if enemy["distance"] < nearest_enemy_distance:
                    nearest_enemy_distance = enemy["distance"]
                    nearest_enemy_pos = enemy["position"]
                    nearest_enemy_health = enemy["health"]
        
        # Position relative to player
        if nearest_enemy_distance < float('inf'):
            rel_x = (nearest_enemy_pos["x"] - data["player"]["position"]["x"]) / 1000.0
            rel_y = (nearest_enemy_pos["y"] - data["player"]["position"]["y"]) / 1000.0
            distance = nearest_enemy_distance / 1000.0
            enemy_health_pct = nearest_enemy_health["current"] / max(1, nearest_enemy_health["max"])
        else:
            rel_x, rel_y, distance, enemy_health_pct = 0, 0, 1.0, 0
        
        state.extend([rel_x, rel_y, distance, enemy_health_pct])
        
        # Weapons info
        weapons_equipped = min(len(data["player"]["equipped_weapons"]), 2) / 2.0
        state.append(weapons_equipped)
        
        # Add cooldown of weapons
        cooldown1 = 0.0
        cooldown2 = 0.0
        if len(data["player"]["equipped_weapons"]) > 0:
            cooldown1 = data["player"]["equipped_weapons"][0]["cooldown"] / 100.0
        if len(data["player"]["equipped_weapons"]) > 1:
            cooldown2 = data["player"]["equipped_weapons"][1]["cooldown"] / 100.0
        state.extend([cooldown1, cooldown2])
        
        return np.array(state, dtype=np.float32)
    
    def _calculate_reward(self, current_data: Dict) -> float:
        """Calculate reward based on game state changes."""
        reward = 0.0
        
        # Basic survival reward
        reward += 0.01
        
        # Reward for health changes
        current_health = current_data["player"]["health"]["current"]
        if self.last_health > 0:
            health_change = current_health - self.last_health
            if health_change < 0:
                # Penalize taking damage
                reward -= abs(health_change) * 0.5
            elif health_change > 0:
                # Reward healing
                reward += health_change * 0.2
        self.last_health = current_health
        
        # Reward for getting closer to enemies (encourages engagement)
        if self.nearest_enemy_distance < self.last_nearest_enemy_distance:
            # Getting closer to enemy
            if self.nearest_enemy_distance < 200:  # Close combat range
                reward += 0.05
        self.last_nearest_enemy_distance = self.nearest_enemy_distance
        
        # Reward for completing waves
        if current_data["game"]["wave"] > self.last_game_wave:
            reward += 5.0  # Big reward for completing a wave
        
        # Check if game over (0 health)
        if current_health <= 0:
            reward -= 10.0  # Large negative reward for dying
            self.game_over = True
        
        return reward
    
    def get_action(self) -> int:
        """Get the next action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            # Exploration: random action
            return random.randint(0, self.action_size - 1)
        else:
            # Exploitation: use the policy network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(self.current_state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
    
    def learn(self):
        """Update the model based on stored experiences."""
        # Only learn if we have enough samples
        if len(self.memory) < self.batch_size:
            return
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        state_batch = torch.FloatTensor(states).to(self.device)
        action_batch = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(next_states).to(self.device)
        done_batch = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states.
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
        
        # Compute the expected Q values
        expected_state_action_values = reward_batch + (self.gamma * next_state_values * (1 - done_batch))
        
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to help stabilize training
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # Update target network
        if self.step_count % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def on_episode_end(self):
        """Called at the end of an episode for cleanup and statistics."""
        self.episode_count += 1
        print(f"Episode {self.episode_count} ended with reward: {self.cumulative_reward}")
        print(f"Current epsilon: {self.epsilon}")
        
        # Save model periodically
        if self.episode_count % 10 == 0:
            self.save_model()
    
    def save_model(self):
        """Save the trained model."""
        model_path = os.path.join(self.model_dir, "dqn_model.pth")
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'step_count': self.step_count
        }, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self):
        """Load a previously trained model if available."""
        model_path = os.path.join(self.model_dir, "dqn_model.pth")
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path)
                self.policy_net.load_state_dict(checkpoint['policy_net'])
                self.target_net.load_state_dict(checkpoint['target_net'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.epsilon = checkpoint['epsilon']
                self.episode_count = checkpoint['episode_count']
                self.step_count = checkpoint['step_count']
                print(f"Model loaded from {model_path}")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
        return False
