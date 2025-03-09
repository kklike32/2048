import pygame
import sys
from game import Game
from states import GameState
from display import Display

def main():
    pygame.init()
    game = Game()
    display = Display(game)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif game.state == GameState.MainMenu:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    game.reset_game()
            elif game.state == GameState.Playing:
                game.handle_event(event)
            elif game.state == GameState.GameOver:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    game.reset_game()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                print(game.board.board)

        if game.state == GameState.Playing:
            if game.board.game_over():
                game.update_max_score()
                game.state = GameState.GameOver
        
        display.draw(game.state)

if __name__ == "__main__":
    main()
