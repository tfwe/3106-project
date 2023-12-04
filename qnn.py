import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import game
import math
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=1)
        self.conv2 = nn.Conv2d(128, 512, kernel_size=1)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=1)
        self.fc1 = nn.Linear(512*9, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)  # 4 outputs for up, down, left, right

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(1, 512*9)  # Flatten layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation
        return x

def select_action(board, epsilon, q_net, device):
    total_possible_actions = ["up","down","left","right"]
    available_actions = board.get_available_moves
    if np.random.rand() < epsilon:
        action_index = random.randint(0,3)
        return torch.tensor(action_index).view(1,1)
    else:
        # Calculate Q-values for each action
        state = encode_game_state(board.board_array)
        with torch.no_grad():
            float32_state = np.float32(state)
            state_tensor = torch.from_numpy(float32_state).view(1, 3, 3).to(device)
            action_index = torch.argmax(q_net(state_tensor)).item()
            return torch.tensor(action_index).view(1, 1)

def update_q_net(q_net, target_net, replay_memory, batch_size, Transition, optimizer, device, criterion, gamma):
    # Sample a mini-batch from the replay memory
    transitions = random.sample(replay_memory, batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)
    # Compute the Q-learning loss
    state_batch = torch.cat(batch.state).view(1,1,3,3).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    state_action_values = q_net(state_batch.view(1,1,3,3)).gather(1, action_batch)

    next_state_values = torch.zeros(batch_size).to(device)
    next_state_values[non_final_mask] = target_net(state_batch).max(1)[0].detach()
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
    encoded_batch =[]
    for tile in flattened_arr:
      if tile < 0:
        tile_abs = math.fabs(tile)
        log2_tile = float(math.log(tile_abs, 2))*-1
      elif tile == 0:
        log2_tile = 0
      else:
        log2_tile = float(math.log(tile, 2))

      encoded_arr.append(log2_tile)
    # for i in range(64):
    #     encoded_batch.append(encoded_arr)
    return torch.tensor(encoded_arr)

def calc_reward(board_array):
  num_open_tiles = np.count_nonzero(board_array == 0)
  score = 0
  score += num_open_tiles 
  # score += np.max(np.ravel(board_array))
  return torch.tensor(score).view(1)

def sample_random_games(n):
    turns = []
    max_tiles = []
    min_tiles = []
    piece_distributions = []
    for i in range(n):
        board = game.Board(size=3)
        board.start_game()
        while not board.is_game_over():
            board.random_move()
        num_pieces = {}
        turns.append(board.turn)
        max_tiles.append(np.max(np.ravel(board.board_array)))
        min_tiles.append(np.min(np.ravel(board.board_array)))
        for tile in (np.ravel(board.board_array)):
            if num_pieces.get(tile) == None:
                num_pieces[tile] = 1
                continue
            num_pieces[tile] += 1
        piece_distributions.append(num_pieces)
    max_tiles = [int(math.log(max_tile, 2)) for max_tile in max_tiles]
    return turns, max_tiles, min_tiles, piece_distributions

def sample_network_games(n, epsilon_end, epsilon_decay, q_net, device):
    turns = []
    max_tiles = []
    min_tiles = []
    piece_distributions = []
    for i in range(n):
        board = game.Board(size=3)
        board.start_game()
        while not board.is_game_over():
            epsilon = max(epsilon_end, epsilon_decay * i)
            action = select_action(board, epsilon, q_net, device)
            # action = np.random.choice(board.get_available_moves())
            total_possible_actions = ["up","down","left","right"]
            board.move(total_possible_actions[action])
        num_pieces = {}
        turns.append(board.turn)
        max_tiles.append(np.max(np.ravel(board.board_array)))
        min_tiles.append(np.min(np.ravel(board.board_array)))
        for tile in (np.ravel(board.board_array)):
            if num_pieces.get(tile) == None:
                num_pieces[tile] = 1
                continue
            num_pieces[tile] += 1
        piece_distributions.append(num_pieces)
    max_tiles = [int(math.log(max_tile, 2)) for max_tile in max_tiles]
    return turns, max_tiles, min_tiles, piece_distributions

def visualize_data(turns, max_tiles, min_tiles, piece_distributions, plt):


    # Plot histogram of turns
    plt.subplot(2, 2, 1)
    plt.hist(turns)
    plt.title('Turns taken in each game')

    # Plot histogram of max tiles
    plt.subplot(2, 2, 2)
    plt.hist(max_tiles, bins=range(0, max(max_tiles)))
    plt.title('Log(Max tile, 2) value in each game')

    # Plot histogram of min tiles
    plt.subplot(2, 2, 3)
    plt.hist(min_tiles, bins=20)
    plt.title('Min tile value in each game')

    # Plot bar chart of piece distributions
    plt.subplot(2, 2, 4)
    all_tiles = [tile for dist in piece_distributions for tile in dist]
    all_counts = [count for dist in piece_distributions for count in dist.values()]
    plt.bar(all_counts, all_tiles)
    plt.title('Tile distributions across all games')

    # Show the plot



def main():
    epsilon_end = 0.01
    epsilon_decay = 0.9
    batch_size = 1
    gamma = 0.9
    target_update = 100
    board = game.Board(size=3)
    board.start_game()
    state = encode_game_state(board.board_array)

    # Check if CUDA is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using device {torch.cuda.get_device_name(0)}")

    # Initialize the network
    q_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()
    q_net.train()

    # Define a Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(q_net.parameters(), lr=0.1, momentum=0.9)

    # define replay memory
    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
    replay_memory = deque(maxlen=10000)

    for episode in range(1000):  
        board = game.Board(size=3)
        board.start_game()
        running_loss = 0.0
        total_steps = 0
        while not board.is_game_over(): 
            total_steps += 1
            epsilon = max(epsilon_end, epsilon_decay * episode)
            action = select_action(board, epsilon, q_net, device)
            # action = np.random.choice(board.get_available_moves())
            total_possible_actions = ["up","down","left","right"]
            board.move(total_possible_actions[action])
            
            next_state = encode_game_state(board.board_array)
            reward = calc_reward(board.board_array)
            replay_memory.append(Transition(state, action, next_state, reward))
            state = next_state

            if len(replay_memory) > batch_size:
                update_q_net(q_net, target_net, replay_memory, batch_size, Transition, optimizer, device, criterion, gamma)
                if total_steps % target_update == 0:
                    update_target_net(target_net, q_net)
                    q_net.train()
                    torch.save(q_net.state_dict(), './q_net.pth')
                    torch.save(target_net.state_dict(), './target_net.pth')
        print(f"episode: {episode}")

    n = 1000

    turns, max_tiles, min_tiles, piece_distributions = sample_random_games(n)
    plt.figure()
    visualize_data(turns, max_tiles, min_tiles, piece_distributions, plt)
    plt.show()
    

    # plt.figure()
    q_net.eval()
    turns, max_tiles, min_tiles, piece_distributions = sample_network_games(n, epsilon_end, epsilon_decay, q_net, device)
    visualize_data(turns, max_tiles, min_tiles, piece_distributions, plt)
    plt.show()



        # print(board)

if __name__ == "__main__":
   main() 
