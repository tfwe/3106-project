import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import game
import math
from collections import namedtuple, deque
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(1, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)  # 4 outputs for up, down, left, right

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = x.view(-1, 64*6*6)  # Flatten layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation
        return x

def select_action(board, epsilon, q_net, device):
    if np.random.rand() < epsilon:
        action = np.random.choice(board.get_available_moves())
    else:
        # Calculate Q-values for each action
        state = encode_game_state(board.board_array)
        with torch.no_grad():
            float32_state = np.float32(state)
            state_tensor = torch.from_numpy(float32_state).view(1, 3, 3).to(device)
            action = torch.argmax(q_net(state_tensor)).item()
            print(q_net(state_tensor))
            # print(action)
    return action

def update_q_net(q_net, replay_memory, batch_size, Transition, optimizer, device):
    # Sample a mini-batch from the replay memory
    transitions = random.sample(replay_memory, batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None]).to(device)
    # Compute the Q-learning loss
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    state_action_values = q_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(batch_size)
    next_state_values[non_final_mask] = target_net(non_final_batch).max(1)[0].detach()
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in q_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def update_target_net(target_net, q_net):
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
    
    return np.array([encoded_arr])
def calc_reward(board_array):
  num_open_tiles = np.count_nonzero(board_array == 0)
  score = 0
  score += num_open_tiles 
  return score

def main():
    epsilon_end = 0.01
    epsilon_decay = 0.9999
    board = game.Board(size=3)
    board.start_game()
    state = encode_game_state(board.board_array)
    print(board.board_array)

    # Check if CUDA is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using device {torch.cuda.get_device_name(0)}")

    # Initialize the network
    q_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    # Define a Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(q_net.parameters(), lr=0.001, momentum=0.9)

    # define replay memory
    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
    replay_memory = deque(maxlen=10000)
    print(replay_memory)

    for episode in range(2):  
        board.start_game()
        running_loss = 0.0
        while not board.is_game_over(): 
            epsilon = max(epsilon_end, epsilon_decay * episode)
            action = select_action(board, epsilon, q_net, device)
            # action = np.random.choice(board.get_available_moves())
            board.move(action)
            
            next_state = encode_game_state(board.board_array)
            reward = calc_reward(board.board_array)
            replay_memory.append(Transition(state, action, next_state, reward))
            state = next_state
            batch_size = 64
            if len(replay_memory) > batch_size:
                update_q_net(q_net, replay_memory, batch_size, Transition, optimizer, device)
                if total_steps % target_update == 0:
                    update_target_net(target_net, q_net)
            print(board)

if __name__ == "__main__":
   main() 
