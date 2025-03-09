import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(16, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # 4 actions: up, down, left, right
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_size=16, action_size=4, memory_size=100000, 
                 gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, 
                 learning_rate=0.001, batch_size=64, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device
        
        # Q-Network and target network
        self.model = DQN().to(device)
        self.target_model = DQN().to(device)
        self.update_target_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        self.steps = 0
        self.update_target_every = 1000  # update target network every 1000 steps
        
        # For statistics
        self.rewards_history = []
        self.loss_history = []

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def preprocess_state(self, state):
        # Convert board state to the proper input format
        # Apply log2 to the non-zero values to handle the exponential growth of tile values
        normalized_state = np.zeros(16, dtype=np.float32)
        for i, val in enumerate(state):
            if val > 0:
                normalized_state[i] = np.log2(val)
            else:
                normalized_state[i] = 0
        # Scale to range [0, 1]
        if np.max(normalized_state) > 0:
            normalized_state = normalized_state / 16.0  # Max possible value in 2048 is 2^16
        return torch.FloatTensor(normalized_state).to(self.device)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_moves=None):
        if valid_moves is None:
            valid_moves = [0, 1, 2, 3]  # All moves: up, down, left, right
            
        if np.random.rand() <= self.epsilon:
            # Explore: choose a random valid move
            return random.choice(valid_moves)
        
        # Exploit: choose best action based on Q values
        state_tensor = self.preprocess_state(state)
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy()
        
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
        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.update_target_model()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))
        self.update_target_model()
        
    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def decay_epsilon(self):
        """Manually decay epsilon when called"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay