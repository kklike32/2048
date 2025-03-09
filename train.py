import numpy as np
import torch
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
    agent = DQNAgent(state_size=16, action_size=4)
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
                tile_reward = np.log2(new_max)
                max_tile = new_max
            else:
                tile_reward = 0
                
            # Penalty for not changing the board (should not happen with valid_moves)
            unchanged_penalty = 0
            if np.array_equal(old_board, board.board):
                unchanged_penalty = -10
                
            # Total reward for this step
            reward = score_reward + tile_reward + unchanged_penalty
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
            agent.save(f'models/dqn_2048_episode_{episode}.h5')
    
    # Save final model
    agent.save('models/dqn_2048_final.h5')
    
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