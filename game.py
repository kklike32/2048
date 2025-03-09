import pygame
from board import Board
from states import GameState

class Game:
    def __init__(self):
        self.state = GameState.MainMenu
        self.board = Board()
        self.max_score = 0

    def reset_game(self):
        self.board.reset()
        self.board.add_random_tile()
        self.board.add_random_tile()
        self.state = GameState.Playing

    # def start_game(self):
    #     # Add two random tiles at the start
    #     self.board.add_random_tile()
    #     self.board.add_random_tile()
    
    def get_max_score(self):
        return self.max_score
    
    def update_max_score(self):
        if self.board.current_score > self.max_score:
            self.max_score = self.board.current_score
    
    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.board.move('up')
            elif event.key == pygame.K_DOWN:
                self.board.move('down')
            elif event.key == pygame.K_LEFT:
                self.board.move('left')
            elif event.key == pygame.K_RIGHT:
                self.board.move('right')
        self.update_max_score()
