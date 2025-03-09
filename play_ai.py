import pygame
import sys
import numpy as np
import time
from agent import DQNAgent
from game import Game
from states import GameState
from display import Display

# Map action indices to directions
ACTIONS = ['up', 'down', 'left', 'right']

def get_valid_moves(board):
    """Return indices of valid moves"""
    valid_moves = []
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

        if not np.array_equal(board_copy, board.board):
            valid_moves.append(action_idx)
        
        # Restore the board
        board.board = np.copy(board_copy)
    
    return valid_moves

def main(model_path='models/dqn_2048_final.h5', delay=0.5, human_control=False):
    pygame.init()
    
    # Initialize the agent
    agent = DQNAgent(state_size=16, action_size=4)
    try:
        agent.load(model_path)
        print(f"Model loaded from {model_path}")
        agent.epsilon = 0.0  # No exploration in play mode
    except:
        print(f"Failed to load model from {model_path}. Starting with random agent.")
        
    game = Game()
    display = Display(game)

    # Start a new game
    game.reset_game()
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            # Allow exiting to menu
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                game.state = GameState.MainMenu
                
            # Allow manually controlling the game when in human_control mode
            if human_control and game.state == GameState.Playing:
                game.handle_event(event)
                
        if game.state == GameState.MainMenu:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    game.reset_game()
                    
        elif game.state == GameState.Playing:
            if not human_control:
                # AI makes a move
                state = game.board.get_board_state()
                valid_moves = get_valid_moves(game.board)
                
                if not valid_moves or game.board.game_over():
                    game.update_max_score()
                    game.state = GameState.GameOver
                    continue
                    
                action_idx = agent.act(state, valid_moves)
                action = ACTIONS[action_idx]
                
                # Execute the move
                game.board.move(action)
                game.update_max_score()
                
                # Small delay to watch the AI play
                time.sleep(delay)
                
        elif game.state == GameState.GameOver:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    game.reset_game()
        
        # Update the display
        display.draw(game.state)
        
        # Display AI info if playing
        if game.state == GameState.Playing and not human_control:
            # Could add more AI information display here if desired
            pygame.display.set_caption(f"2048 AI - Score: {game.board.get_current_score()} - Max Tile: {np.max(game.board.board)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Watch or control the 2048 AI')
    parser.add_argument('--model', type=str, default='models/dqn_2048_final.h5', help='Path to the trained model file')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay between AI moves (seconds)')
    parser.add_argument('--human', action='store_true', help='Enable human control instead of AI')
    
    args = parser.parse_args()
    main(model_path=args.model, delay=args.delay, human_control=args.human)