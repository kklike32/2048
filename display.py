import pygame
from tile import Tile
from states import GameState

# Constants
WIDTH = 800
HEIGHT = 900
TILE_SIZE = 150  
TILE_MARGIN = 20
SCORE_HEIGHT = 100 
SCORE_MARGIN = 30  


class Display:
    def __init__(self, game):
        self.game = game
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.font = pygame.font.SysFont(None, 55)

    def draw_score_area(self):
        # Calculate the total width taken up by the score boxes
        total_score_width = WIDTH // 2 - SCORE_MARGIN * 2 - 10
        # Calculate the X position to start drawing score boxes so they're centered
        score_x_position = (WIDTH - total_score_width * 2 - SCORE_MARGIN) // 2
        
        # Draw current score area
        pygame.draw.rect(
            self.screen, 
            (204, 192, 179),
            pygame.Rect(
                score_x_position,
                SCORE_MARGIN,
                total_score_width,
                SCORE_HEIGHT
            )
        )
        # Draw max score area
        pygame.draw.rect(
            self.screen, 
            (204, 192, 179),
            pygame.Rect(
                score_x_position + total_score_width + SCORE_MARGIN,
                SCORE_MARGIN,
                total_score_width,
                SCORE_HEIGHT
            )
        )
       
        score = self.game.board.get_current_score()
        # Calculate the center position for current score text
        current_score_text = self.font.render('Score: ' + str(score), True, (255, 255, 255))
        current_score_rect = current_score_text.get_rect()
        current_score_x = score_x_position + (total_score_width - current_score_rect.width) // 2
        current_score_y = SCORE_MARGIN + (SCORE_HEIGHT - current_score_rect.height) // 2

        max_score = self.game.get_max_score()
        # Calculate the center position for max score text
        max_score_text = self.font.render('Max Score: ' + str(max_score), True, (255, 255, 255))
        max_score_rect = max_score_text.get_rect()
        max_score_x = score_x_position + total_score_width + SCORE_MARGIN + (total_score_width - max_score_rect.width) // 2
        max_score_y = SCORE_MARGIN + (SCORE_HEIGHT - max_score_rect.height) // 2

        # Blit the score text onto the screen at the calculated positions
        self.screen.blit(current_score_text, (current_score_x, current_score_y))
        self.screen.blit(max_score_text, (max_score_x, max_score_y))

    
    def draw_tiles(self):
        board_width = 4 * TILE_SIZE + 3 * TILE_MARGIN
        board_height = board_width  # Square board
        
        # Calculate the total width and height of the board
        board_x_position = (WIDTH - 4 * TILE_SIZE - 3 * TILE_MARGIN) // 2
        board_y_position = (HEIGHT - board_height) // 2 + SCORE_HEIGHT - 40

        # Draw the board, tiles
        for i in range(4):
            for j in range(4):
                tile_value = self.game.board.get_tile_value(i, j)
                tile = Tile(tile_value, self.font)
                tile_x = board_x_position + j * (TILE_SIZE + TILE_MARGIN)
                tile_y = board_y_position + i * (TILE_SIZE + TILE_MARGIN)
                tile.draw(self.screen, tile_x, tile_y, TILE_SIZE)

    def draw_main_menu(self):
        self.screen.fill((0, 0, 0))  
        title_text = self.font.render('2048 Game', True, (255, 255, 255))
        self.screen.blit(title_text, (100, 100)) 
        play_text = self.font.render('Press SPACE to Play', True, (255, 255, 255))
        self.screen.blit(play_text, (100, 200)) 

    def draw_game_over(self):
        game_over_text = self.font.render('Game Over!', True, (255, 0, 0))
        self.screen.blit(game_over_text, (100, 100)) 
        play_again_text = self.font.render('Press R to Play Again', True, (255, 255, 255))
        self.screen.blit(play_again_text, (100, 200))
    def draw_playing(self):
        self.screen.fill((187, 173, 160))  # Background color
        self.draw_score_area()  # Draw the score areas
        self.draw_tiles()       # Draw the tiles
    def draw(self, state):
        if state == GameState.MainMenu:
            self.draw_main_menu()
        elif state == GameState.Playing:
            self.draw_playing()
            pass
        elif state == GameState.GameOver:
            self.draw_game_over()
        pygame.display.update()