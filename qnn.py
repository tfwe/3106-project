import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import game
import math
from collections import namedtuple, deque
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)  # 4 outputs for up, down, left, right

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64*6*6)  # Flatten layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation
        return x

def select_action(board, epsilon):
    # if np.random.rand() < epsilon:
    action = np.random.choice(board.get_available_moves())
    # else:
    #     # Calculate Q-values for each action
    #     q_values = calculate_q_values(state)
    #     # Choose action with the highest Q-value
    #     action = np.argmax(q_values)
    return action

def update_q_net():
    # Sample a mini-batch from the replay memory
    transitions = random.sample(replay_memory, batch_size)
    batch = Transition(*zip(*transitions))

    # Compute the Q-learning loss
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = q_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_batch).max(1)[0].detach()
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in q_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def update_target_net():
    target_net.load_state_dict(q_net.state_dict())
    # Update the target network to match the Q-network

def encode_game_state(board_array):

    flattened_arr = np.ravel(board_array)

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
  int_state = [int(digit) for digit in state]
  
  # compute 2 to the power of each integer to get the original tile value
  tiles = []
  for num in int_state:
    if num < 0:
      tiles.append(-2**math.fabs(num))
    elif num == 0:
      tiles.append(0)
    else:
      tiles.append(2**num)
  
  # reshape back to NxN array
  board_array = np.array(tiles).reshape((3, 3))
  
  return board_array

def calc_reward(board_array):
  # values_sum          = np.sum(np.ravel(board_array))
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



def main():
    board = game.Board(size=3)
    board.load_array(np.array([[-2,0,0],[2,0,0],[0,0,0]]))
    state = encode_game_state(board.board_array)
    print(board.board_array)
    print(decode_game_state(state))

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the network
    q_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    # Define a Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(q_net.parameters(), lr=0.001, momentum=0.9)

    # define replay memory
    Transition = namedtuple('Transition', ('board', 'action', 'next_state', 'reward'))
    replay_memory = deque(maxlen=10000)
    print(replay_memory)

    for episode in range(2):  
        running_loss = 0.0
        while not board.is_game_over(): 
            epsilon = max(epsilon_end, epsilon_decay * episode)
            action = select_action(state, epsilon)
            next_state, reward, done,  = env.step(action)
            replay_memory.append(Transition(state, action, next_state, reward))
            state = next_state
            if len(replay_memory) > batch_size:
                update_q_net()
                if total_steps % target_update == 0:
                    update_target_net()
            if done:
                break

if __name__ == "__main__":
   main() 
