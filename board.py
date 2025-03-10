import random
import numpy as np

class Board:
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.current_score = 0
        
    def reset(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.current_score = 0
    
    def get_board_state(self):
        return self.board.flatten()
    
    def normalize_board(self):
        normalized_board = np.where(self.board > 0, np.log2(self.board), 0)
        return normalized_board.flatten()
    
    def get_current_score(self):
        return self.current_score
    
    def get_empty_tiles(self):
        return [(i, j) for i in range(4) for j in range(4) if self.board[i][j] == 0]
    
    def get_tile_value(self, i, j):
        return self.board[i][j]
    
    def set_tile_value(self, i, j, value):
        self.board[i][j] = value
        
    def add_random_tile(self):
        empty_tiles = self.get_empty_tiles()
        if empty_tiles:
            i, j = random.choice(empty_tiles)
            if random.random() < 0.15:
                val = 4
            else:
                val = 2
            self.set_tile_value(i, j, val)
    
    def is_board_full(self):
        return all(self.board[i][j] != 0 for i in range(4) for j in range(4))
    
    def can_move(self):
        for i in range(4):
            for j in range(4):
                if j > 0 and self.board[i][j] == self.board[i][j - 1]:
                    return True
                if i > 0 and self.board[i][j] == self.board[i - 1][j]:
                    return True
        return False
    
    def game_over(self):
        return self.is_board_full() and not self.can_move()
    
    def compress(self, board):
        """Compress the board after moving tiles, no merge involved here."""
        if isinstance(board, list):
            board = np.array(board)
            
        new_board = np.zeros((4, 4), dtype=int)
        for i in range(4):
            pos = 0
            for j in range(4):
                if board[i][j] != 0:
                    new_board[i][pos] = board[i][j]
                    pos += 1
        return new_board

    def merge(self, board):
        """Merge the tiles with same value."""
        if isinstance(board, list):
            board = np.array(board)
            
        board_copy = np.copy(board)
        self._merge_count = 0  # Reset merge counter
        
        for i in range(4):
            for j in range(3):
                if board_copy[i][j] == board_copy[i][j + 1] and board_copy[i][j] != 0:
                    board_copy[i][j] *= 2
                    board_copy[i][j + 1] = 0
                    self.current_score += board_copy[i][j]
                    self._merge_count += 1  # Count each merge
        return board_copy

    def get_merge_count(self):
        """Return the number of merges from the last move"""
        return getattr(self, '_merge_count', 0)

    def reverse(self, board):
        """Reverse the board rows."""
        if isinstance(board, list):
            board = np.array(board)
        
        return np.array([row[::-1] for row in board])

    def transpose(self, board):
        """Transpose the board to easily compress and merge columns."""
        if isinstance(board, list):
            board = np.array(board)
            
        return board.T.copy()

    def move_left(self):
        # Compress the grid
        self.board = self.compress(self.board)
        
        # Merge the tiles
        self.board = self.merge(self.board)
        
        # Compress after merging
        self.board = self.compress(self.board)

    def move_right(self):
        # Reverse the board to move non-zero values to the rightmost
        self.board = self.reverse(self.board)

        # Compress the grid
        self.board = self.compress(self.board)
        
        # Merge the tiles
        self.board = self.merge(self.board)
        
        # Compress after merging
        self.board = self.compress(self.board)

        # Reverse back to original orientation
        self.board = self.reverse(self.board)

    def move_down(self):
        # Transpose the board to work with columns
        self.board = self.transpose(self.board)

        # Reverse to prepare for compressing non-zero values down
        self.board = self.reverse(self.board)

        # Compress the grid
        self.board = self.compress(self.board)
        
        # Merge the tiles
        self.board = self.merge(self.board)
        
        # Compress after merging
        self.board = self.compress(self.board)

        # Reverse back after compression and merging
        self.board = self.reverse(self.board)

        # Transpose the board back to original
        self.board = self.transpose(self.board)
    
    def move_up(self):
        # Transpose the board to work with columns as rows
        self.board = self.transpose(self.board)

        # Compress the board
        self.board = self.compress(self.board)

        # Merge the tiles
        self.board = self.merge(self.board)

        # Compress after merging
        self.board = self.compress(self.board)

        # Transpose the board back to original
        self.board = self.transpose(self.board)
    
    
    def move(self, direction):
        original_board = np.copy(self.board)
        
        if direction == 'up':
            self.move_up()
        elif direction == 'down':
            self.move_down()
        elif direction == 'left':
            self.move_left()
        elif direction == 'right':
            self.move_right()
        
        if isinstance(self.board, list):
            self.board = np.array(self.board)
        
        if not np.array_equal(original_board, self.board):
            self.add_random_tile()

