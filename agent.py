import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import os

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # CNN layers to process the 4x4 board
        self.conv1 = nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=0)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)  # 4 actions: up, down, left, right
        
        # Regularization
        self.dropout = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        
    def forward(self, x):
        # Reshape input to 4x4 grid with 1 channel
        x = x.view(-1, 1, 4, 4)
        
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class DQNAgent:
    def __init__(self, state_size=16, action_size=4, memory_size=100000, 
                 gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9995, 
                 learning_rate=0.0005, batch_size=256, device=None):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Explicitly check for GPU
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Q-Network and target network
        self.model = DQN().to(self.device)
        self.target_model = DQN().to(self.device)
        self.update_target_model()
        
        # Use Adam with weight decay for regularization
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        
        self.steps = 0
        self.update_target_every = 1000  # update target network every 1000 steps
        
        # For statistics
        self.rewards_history = []
        self.loss_history = []
        self.q_values_history = []
    
    # Add new method for episode-based epsilon decay
    def decay_epsilon_once(self):
        """Decay epsilon once per episode instead of every step"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def preprocess_state(self, state):
        # Convert board state to the proper input format
        normalized_state = np.zeros(16, dtype=np.float32)
        for i, val in enumerate(state):
            if val > 0:
                normalized_state[i] = np.log2(val)
            else:
                normalized_state[i] = 0
        
        # Scale to range [0, 1]
        if np.max(normalized_state) > 0:
            normalized_state = normalized_state / 16.0
        
        # Reshape to 4x4 for CNN
        board_tensor = torch.FloatTensor(normalized_state).to(self.device)
        board_tensor = board_tensor.view(-1, 4, 4)
        return board_tensor

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_moves=None):
        if valid_moves is None or len(valid_moves) == 0:
            valid_moves = [0, 1, 2, 3]  # All moves: up, down, left, right
            
        if np.random.rand() <= self.epsilon:
            # Explore: choose a random valid move
            return random.choice(valid_moves)
        
        # Exploit: choose best action based on Q values
        state_tensor = self.preprocess_state(state)
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy().flatten()
        
        # Filter out invalid moves by setting their Q-values very low
        for action in range(self.action_size):
            if action not in valid_moves:
                q_values[action] = -np.inf
                
        return np.argmax(q_values)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = torch.stack([self.preprocess_state(s) for s, _, _, _, _ in minibatch])
        actions = torch.tensor([a for _, a, _, _, _ in minibatch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([r for _, _, r, _, _ in minibatch], dtype=torch.float32).to(self.device)
        next_states = torch.stack([self.preprocess_state(ns) for _, _, _, ns, _ in minibatch])
        dones = torch.tensor([d for _, _, _, _, d in minibatch], dtype=torch.float32).to(self.device)
        
        self.model.train()
        self.target_model.eval()
        
        # Current Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values
        with torch.no_grad():
            max_next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
        
        # Compute loss and update weights
        if hasattr(self, 'use_amp') and self.use_amp:
            with torch.cuda.amp.autocast():
                current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    max_next_q_values = self.target_model(next_states).max(1)[0]
                    target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
                loss = self.criterion(current_q_values, target_q_values)
                
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Regular training (existing code)
            loss = self.criterion(current_q_values, target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.update_target_model()
        
        # Decay epsilon
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
        
        return loss.item()


    def load(self, filename):
        if os.path.exists(filename):
            self.model.load_state_dict(torch.load(filename, map_location=self.device))
            self.update_target_model()
            print(f"Model loaded from {filename}")
        else:
            print(f"No model found at {filename}")
        
    def save(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(self.model.state_dict(), filename)
        print(f"Model saved to {filename}")