import numpy as np
import torch
import gc
import pygame
import sys
import os
from agent import DQNAgent
from board import Board
from game import Game
import matplotlib.pyplot as plt
from tqdm import tqdm

# Map action indices to directions
ACTIONS = ['up', 'down', 'left', 'right']

def print_gpu_info():
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    else:
        print("No GPU available")

def get_valid_moves(board):
    """Return indices of valid moves"""
    valid_moves = []
    # Test each move and see if it changes the board
    for action_idx, action in enumerate(ACTIONS):
        board_copy = np.copy(board.board)
        if action == 'up':
            board.move_up()
        elif action == 'down':
            board.move_down()
        elif action == 'left':
            board.move_left()
        elif action == 'right':
            board.move_right()

        # If the board changed, it's a valid move
        if not np.array_equal(board_copy, board.board):
            valid_moves.append(action_idx)
        
        # Restore the board
        board.board = np.copy(board_copy)
    
    return valid_moves

def train_agent(episodes=10000, render_every=500, save_every=1000, render=False):
    """Train the DQN agent to play 2048"""
    print_gpu_info()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(state_size=16, action_size=4, batch_size=256, device=device)
    
    if torch.cuda.is_available():
        # Use mixed precision if available (speeds up training on modern GPUs)
        scaler = torch.cuda.amp.GradScaler()
        use_amp = True
        print("Using mixed precision training")
    else:
        use_amp = False
    
    scores = []
    max_tiles = []
    
    # Create directory for saving models
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Initialize PyGame if rendering
    if render:
        pygame.init()
    
    # Progress bar
    pbar = tqdm(range(episodes), desc="Training")
    
    for episode in pbar:
        # Initialize game environment
        game = Game()
        game.reset_game()
        board = game.board
        done = False
        total_reward = 0
        
        # Get initial state
        state = board.get_board_state()
        
        # For tracking the highest tile
        max_tile = np.max(board.board)
        
        while not done:
            # Get valid moves
            valid_moves = get_valid_moves(board)
            
            # If no valid moves, game is over
            if not valid_moves:
                done = True
                continue
                
            # Agent chooses action
            action_idx = agent.act(state, valid_moves)
            action = ACTIONS[action_idx]
            
            # Remember the score before the move
            old_score = board.get_current_score()
            
            # Execute action
            old_board = np.copy(board.board)
            board.move(action)
            
            # Calculate reward based on score improvement and highest tile
            new_score = board.get_current_score()
            score_reward = new_score - old_score

            # Additional reward for getting higher tiles
            new_max = np.max(board.board)
            if new_max > max_tile:
                tile_reward = np.log2(new_max) * 2.0  # Increased weight
                max_tile = new_max
            else:
                tile_reward = 0

            # Corner strategy reward
            corners = [(0, 0), (0, 3), (3, 0), (3, 3)]
            corner_bonus = 0
            for pos in corners:
                if board.board[pos] == new_max:
                    corner_bonus = np.log2(new_max) * 1.5
                    break

            # Extra bonus for reaching 2048 or higher
            extra_bonus = 0
            if new_max >= 2048:
                extra_bonus = (np.log2(new_max) - 10) * 10  # Logarithmic scaling

            # Merge-focused reward (count the number of merges)
            merge_reward = 0
            merge_count = board.get_merge_count() if hasattr(board, 'get_merge_count') else 0
            if merge_count > 0:
                merge_reward = merge_count * 2.0

            # Monotonicity reward: encourage tiles to be ordered by value
            monotonicity_reward = 0
            # Check rows for monotonicity (decreasing from left to right)
            for i in range(4):
                row = board.board[i]
                if row[0] >= row[1] >= row[2] >= row[3] and row[0] > 0:
                    monotonicity_reward += 1.0
                if row[3] >= row[2] >= row[1] >= row[0] and row[3] > 0:
                    monotonicity_reward += 1.0

            # Check columns for monotonicity (decreasing from top to bottom)
            for j in range(4):
                col = board.board[:, j]
                if col[0] >= col[1] >= col[2] >= col[3] and col[0] > 0:
                    monotonicity_reward += 1.0
                if col[3] >= col[2] >= col[1] >= col[0] and col[3] > 0:
                    monotonicity_reward += 1.0

            # Reward for empty tiles (maintaining space)
            empty_tiles = len(board.get_empty_tiles())
            empty_tiles_reward = empty_tiles * 0.2  # Increased weight

            # Small penalty per move to encourage efficiency
            move_penalty = -0.05

            # Combined reward (with scaling to prevent any single component from dominating)
            reward = (score_reward * 0.5 + 
                      tile_reward + 
                      corner_bonus + 
                      extra_bonus + 
                      merge_reward +
                      monotonicity_reward * 0.5 +
                      empty_tiles_reward +
                      move_penalty)

            # Scale the reward to a reasonable range to prevent exploding gradients
            reward = np.clip(reward, -20, 20)

            total_reward += reward
            
            # Get new state
            next_state = board.get_board_state()
            
            # Check if game is over
            done = board.game_over()
            
            # Remember experience
            agent.remember(state, action_idx, reward, next_state, done)
            
            # Train agent
            loss = agent.replay()
            
            # Move to next state
            state = next_state
            
            # Render the game occasionally
            if render and episode % render_every == 0:
                pygame.event.pump()  # Process PyGame events
                # You would need to adapt the display code to show the AI game
        
        # End of episode - add epsilon decay here
        agent.decay_epsilon_once()
        
        # End of episode
        scores.append(board.get_current_score())
        max_tiles.append(max_tile)
        
        # Update progress bar
        pbar.set_postfix({
            'Score': scores[-1], 
            'Max Tile': max_tiles[-1],
            'Epsilon': agent.epsilon
        })
        
        # Save model periodically
        if episode > 0 and episode % save_every == 0:
            agent.save(f'models/dqn_localv2_2048_episode_{episode}.h5')
    
    # Save final model
    agent.save('models/dqn_localv2_2048_final.h5')
    
    # Plot training results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title('Score History')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    plt.subplot(1, 2, 2)
    plt.plot(max_tiles)
    plt.title('Max Tile History')
    plt.xlabel('Episode')
    plt.ylabel('Max Tile Value')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    return agent

if __name__ == "__main__":
    # Command line arguments for training options
    import argparse
    parser = argparse.ArgumentParser(description='Train a DQN agent to play 2048')
    parser.add_argument('--episodes', type=int, default=10000, help='Number of episodes to train')
    parser.add_argument('--render', action='store_true', help='Render the game while training')
    parser.add_argument('--render_every', type=int, default=500, help='Render every N episodes')
    parser.add_argument('--save_every', type=int, default=1000, help='Save model every N episodes')
    
    args = parser.parse_args()
    
    # Train the agent
    trained_agent = train_agent(
        episodes=args.episodes, 
        render_every=args.render_every,
        save_every=args.save_every,
        render=args.render
    )