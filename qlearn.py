import random
import numpy as np
import math
import game

class td_qlearning:
  alpha = 0.1 # learning rate
  gamma = 0.9 # discount factor
  qvalues = {} # Q function is represented as a dictionary (key: (state, action), value: q value)
  
  def __init__(self, board):
    num_iter = 1 # needs a lot of iterations to fully converge
    for _ in range(num_iter):
      # Perform temporal reinforcement learning algorithm on the trial data
      board.start_game()
      while not board.is_game_over():
        curr_state = np.array2string(encode_board_state(board.board_array)) 
        print(curr_state)
        possible_actions = board.get_available_moves()
        for action in possible_actions: 
          self.qvalue(curr_state, action)
        # can't get previous state when at the first iteration
        # if np.max(board.prev_board) == 0:
        #   optimal_move = self.policy(curr_state)
        #   board.move(optimal_move)
        #   continue
        prev_state = np.array2string(encode_board_state(board.prev_board))
        prev_action = board.prev_move

        q = self.qvalue(prev_state, prev_action)
        sorted_actions = self.policy(curr_state, possible_actions)
        if len(sorted_actions) == 0:
          break
        max_action = sorted_actions[0]
        error_term = calc_reward(board.prev_board) + self.gamma * self.qvalues[curr_state][max_action] - q
        self.qvalues[prev_state][prev_action] = q + self.alpha * error_term

        optimal_move = self.policy(curr_state, possible_actions)
        board.move(optimal_move)
        print(board)
      # directory is the path to a directory containing trials through state space
      # Return nothing

  def qvalue(self, state, action):
    # state is a string representation of a state
    # action is a string representation of an action

    # Return the q-value for the state-action pair
    if state not in self.qvalues:
      self.qvalues[state] = {}
    if action not in self.qvalues[state]:
      self.qvalues[state][action] = calc_reward(state)
    q = self.qvalues[state][action]
    return q
  
  def get_qvalues(self):
    return self.qvalues 
  
  def policy(self, state, possible_actions):
    # state is a string representation of a state
    a = max(self.qvalues[state], key=self.qvalues[state].get)
    filtered_a = [move for move in a if move in possible_actions]
    return filtered_a

def calc_reward(board_array):
  # values_sum          = np.sum(board_array)
  num_open_tiles      = np.count_nonzero(board_array == 0)
  # edge_bonus          = calc_edge_bonus(board_array)
  # corner_bonus        = calc_corner_bonus(board_array)
  # monotonicity        = calc_monotonicity(board_array)
  # smoothness          = calc_smoothness(board_array)
  # uniqueness_bonus    = count_unique_tiles(board_array)
  # num_merges          = count_available_merges(board_array)
  score = 0
  # score += values_sum 
  score += num_open_tiles 
  # score += uniqueness_bonus 
  # score += corner_bonus 
  # score += edge_bonus
  # score += monotonicity 
  # score += smoothness 
  # score += num_merges
  return score


# representation of environment
# def encode_game_state(board_array):
#   # flatten board to 1D array
#   flattened_board = np.ravel(board_array)
#
#   # convert board to log2 of each tile in hexadecimal (want to preserve each digit representing a tile up to 2048) (2 becomes '1', 4 becomes '2', 8 becomes '3', 16 becomes '4', 2048 becomes 'a' etc.)
#   state = ''
#   for tile in flattened_board:
#     # cant take log of 0 (empty piece)
#     if tile == 0:
#       state += '0'
#       continue
#     log2_tile = int(math.log2(tile))
#
#     # hex(int) outputs a string of the form '0x{int in hexadecimal}', so we need to remove the 0x
#     hex_tile = hex(log2_tile).lstrip('0x')
#
#     # concatenate tiles together to get total board in 16 digits
#     state += hex_tile
#   return state
def encode_board_state(board_array):

    flattened_arr = board_array.flatten()

    encoded_arr = []
    for tile in flattened_arr:
      if tile < 0:
        tile_abs = math.fabs(tile)
        log2_tile = int(math.log(tile_abs, 2))*-1
      elif tile == 0:
        log2_tile = 0
      else:
        log2_tile = int(math.log(tile, 2))

      encoded_arr.append(log2_tile)
    
    return np.array(encoded_arr)
def decode_game_state(state):
  # convert each hexadecimal digit to an integer
  int_state = [int(digit, 16) for digit in state]
  
  # compute 2 to the power of each integer to get the original tile value
  tiles = [2**num if num != 0 else 0 for num in int_state]
  
  # reshape back to 4x4 array
  board_array = np.array(tiles).reshape((3, 3))
  
  return board_array

def main():
  board = game.Board(size=3)
  qlearn = td_qlearning(board)
  print(qlearn.qvalues)
  board_array = np.array([[-4, -2, 0], [2, 4, 8], [16, 32, 64], [128, 256, 512]])
  encoded_game_state = encode_board_state(board_array)
  print(encoded_game_state)
  # decoded_game_state = decode_game_state(encoded_game_state)
  # print(decoded_game_state)

if __name__ == "__main__":
  main()
