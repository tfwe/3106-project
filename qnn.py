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
        self.conv1 = nn.Conv2d(1, 512, kernel_size=1)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=1)
        self.fc1 = nn.Linear(512*9, 512)
        # self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(512, 4)  # 4 outputs for up, down, left, right

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = nn.Flatten()(x)
        x = x.view(1, 512*9)  # Flatten layer
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation
        return x

def init_batch(batch_size, board_size=3, board_array=np.array([])):
    boards = []
    for i in range(batch_size):
        board = game.Board(size=board_size)
        board.start_game()
        if board_array.size != 0:
            board.board_array = np.copy(board_array)
        boards.append(encode_game_state(board.board_array))

    return torch.tensor(boards)

def select_action(states, epsilon, q_net, device, batch_size):
    total_possible_actions = ["up", "down", "left", "right"]
    actions = []
    sim_board = game.Board(3)
    sim_board.start_game()
    for state in states:
        sim_board.board_array = state.reshape(3,3)
        available_actions = sim_board.get_available_moves()
        if np.random.rand() < epsilon:
            legal_action_indices = [total_possible_actions.index(action) for action in available_actions]
            action_index = random.choice(legal_action_indices)
        else:
            # Calculate Q-values for each legal action
            state = init_batch(batch_size, board_array=state)
            with torch.no_grad():
                float32_state = np.float32(state)
                state_tensor = torch.from_numpy(float32_state).view(batch_size, 1, 3, 3).to(device)
                q_values = q_net(state_tensor).view(-1)
                legal_action_indices = [total_possible_actions.index(action) for action in available_actions]
                legal_q_values = q_values[legal_action_indices]
                action_index = torch.argmax(legal_q_values).item()
                action_index = legal_action_indices[action_index]
        actions.append(action_index)
    return torch.tensor(actions)

def update_q_net(q_net, target_net, replay_memory, batch_size, Transition, optimizer, device, criterion, gamma):
    # Sample a mini-batch from the replay memory
    transitions = random.sample(replay_memory, batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([torch.tensor(s) for s in batch.next_state if s is not None]).to(device)
    # Compute the Q-learning loss
    states_array = [torch.tensor(s) for s in batch.state]
    actions_array = torch.tensor([batch_action.item() for batch_action in batch.action])
    # print(actions_array[0])
    # print(states_array[0])
    state_batch = torch.cat(states_array).view(batch_size,1,3,3).to(device)
    # print(torch.tensor([batch_action.item() for batch_action in batch.action]).unsqueeze(1))
    actions_tensor = actions_array.unsqueeze(-2)
    action_batch = torch.cat((actions_tensor,)).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    # print(state_batch.size())
    # print(action_batch.size())
    state_action_values = q_net(state_batch).gather(1, action_batch)

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
    return encoded_arr

def calc_reward(board_array, prev_max_tile, prev_open_tiles, move_errors):
    current_max_tile = float(max(board_array))
    current_open_tiles = np.count_nonzero(board_array == 0)
    
    # Reward based on increase in max tile value
    max_tile_reward = 0
    if current_max_tile > prev_max_tile:
        max_tile_reward = current_max_tile - prev_max_tile
    
    # Reward based on increase in score
    score_reward = (current_max_tile - prev_max_tile) * 10
    
    # Penalty for decreasing the number of open tiles
    open_tiles_penalty = 0
    if current_open_tiles < prev_open_tiles:
        open_tiles_penalty = -10
    
    # Total reward
    reward = max_tile_reward + score_reward + open_tiles_penalty - move_errors
    
    return torch.tensor(reward).view(1)

def main():
    epsilon_end = 0.01
    epsilon_decay = 0.9
    batch_size = 1
    gamma = 0.9
    target_update = 10
    episode_interval = 10
    num_episodes = 1000
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
    criterion = nn.HuberLoss()
    optimizer = optim.SGD(q_net.parameters(), lr=0.01, momentum=0.9)

    # define replay memory
    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
    replay_memory = deque(maxlen=10000)

    loss_array = []
    for episode in range(num_episodes):  
        board = game.Board(size=3)
        board.start_game()
        running_loss = 0.0
        total_steps = 0
        prev_max_tile = 0
        prev_open_tiles = 0
        while not board.is_game_over():
            move_errors = 0
            total_steps += 1
            epsilon = max(epsilon_end, epsilon_decay * episode)
            states = init_batch(batch_size, board_array=board.board_array).numpy()
            action = max(select_action(states, epsilon, q_net, device, batch_size))
            # action = np.random.choice(board.get_available_moves())
            total_possible_actions = ["up","down","left","right"]
            prev_max_tile = max(state)
            prev_open_tiles = np.count_nonzero(board.board_array == 0)
            while not board.move(total_possible_actions[action]):
                move_errors += 1
            next_state = encode_game_state(board.board_array)
            reward = calc_reward(next_state, prev_max_tile, prev_open_tiles, move_errors)
            running_loss += 9 - reward.item()
            replay_memory.append(Transition(state, action, next_state, reward))
            state = next_state

            if len(replay_memory) >= batch_size:
                update_q_net(q_net, target_net, replay_memory, batch_size, Transition, optimizer, device, criterion, gamma)
                if total_steps % target_update == 0:
                    update_target_net(target_net, q_net)
                    q_net.train()
                    torch.save(q_net.state_dict(), './q_net.pth')
                    torch.save(target_net.state_dict(), './target_net.pth')
        if episode % episode_interval == 0:
            print(f"episode: {episode}")
            print(board)
        loss_array.append(running_loss)
    visualize_loss(loss_array, num_episodes)




    n = 1000

    turns, max_tiles, min_tiles, piece_distributions = sample_random_games(n)
    plt.figure()
    visualize_data(turns, max_tiles, min_tiles, piece_distributions, plt)
    plt.show()

    q_net.eval()
    turns, max_tiles, min_tiles, piece_distributions = sample_network_games(n, epsilon_end, epsilon_decay, q_net, device, batch_size)
    visualize_data(turns, max_tiles, min_tiles, piece_distributions, plt)
    plt.show()

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

def sample_network_games(n, epsilon_end, epsilon_decay, q_net, device, batch_size):
    turns = []
    max_tiles = []
    min_tiles = []
    piece_distributions = []
    for i in range(n):
        board = game.Board(size=3)
        board.start_game()
        while not board.is_game_over():
            epsilon = max(epsilon_end, epsilon_decay * i)
            action = select_action([board.board_array], epsilon, q_net, device, batch_size)
            # action = np.random.choice(board.get_available_moves())
            total_possible_actions = ["up","down","left","right"]
            board.move(total_possible_actions[action])
        print(board)
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
    plt.title(f'Number of turns taken in each game, mean: {np.mean(turns)}, std: {np.std(turns)}')

    # Plot histogram of max tiles
    plt.subplot(2, 2, 2)
    plt.hist(max_tiles, bins=range(0, max(max_tiles)))
    plt.title(f'Log(Max tile, 2) value in each game, mean: {np.mean(max_tiles)}, std: {np.std(max_tiles)}')

    # Plot histogram of min tiles
    plt.subplot(2, 2, 3)
    plt.hist(min_tiles, bins=20)
    plt.title(f'Min tile value in each game')

    # Plot bar chart of piece distributions
    plt.subplot(2, 2, 4)
    all_tiles = [tile for dist in piece_distributions for tile in dist]
    all_counts = [count for dist in piece_distributions for count in dist.values()]
    plt.bar(all_counts, all_tiles)
    plt.title(f'Tile distributions across all games')

    # Show the plot

def visualize_loss(loss_array, num_episodes):
    plt.figure()
    plt.plot(np.arange(num_episodes), loss_array)
    plt.title(f'Running loss during training across {num_episodes}, mean: {np.mean(loss_array)}')
    plt.show()

if __name__ == "__main__":
   main() 
