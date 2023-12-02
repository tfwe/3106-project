import game
import numpy as np

board = game.Board(size=4)
print(board)
board.start_game()
print(board)
board.move("up")
print(board)
board.load_array(np.array([[-2,0,0],[4,0,0],[0,0,0]]))
board.move("up")
print(board)
print(board.get_available_moves())
print(board.is_game_over())
print(board.board_array)
print(board.prev_board)

