import game
import qnn
import mcts
import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque

def test_MCTS():
    return 0

def test_DQN(n):
    return 0

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

def sample_mcts_games(n):
    turns = []
    max_tiles = []
    min_tiles = []
    piece_distributions = []
    for i in range(n):
        board = game.Board(size=3)
        board.start_game()
        while not board.is_game_over():
            action = mcts.pick_best_move(board)
            # action = np.random.choice(board.get_available_moves())
            board.move(action)
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
            action = qnn.select_action([board.board_array], epsilon, q_net, device, batch_size)
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


def init_network():
    epsilon_end = 0.01
    epsilon_decay = 0.9
    batch_size = 1
    gamma = 0.9
    target_update = 10
    episode_interval = 10
    num_episodes = 0
    board = game.Board(size=3)
    board.start_game()
    state = qnn.encode_game_state(board.board_array)

    # Check if CUDA is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using device {torch.cuda.get_device_name(0)}...")

    # Initialize the network
    q_net = qnn.DQN().to(device)
    target_net = qnn.DQN().to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()
    q_net.train()

    # Define a Loss function and optimizer
    criterion = nn.HuberLoss()
    optimizer = optim.SGD(q_net.parameters(), lr=0.01, momentum=0.9)

    # define replay memory
    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
    replay_memory = deque(maxlen=10000)
    q_net.load_state_dict(torch.load('./q_net.pth'))
    target_net.load_state_dict(torch.load('./target_net.pth'))
    if num_episodes > 0:
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
                states = qnn.init_batch(batch_size, board_array=board.board_array).numpy()
                action = max(qnn.select_action(states, epsilon, q_net, device, batch_size))
                # action = np.random.choice(board.get_available_moves())
                total_possible_actions = ["up","down","left","right"]
                prev_max_tile = max(state)
                prev_open_tiles = np.count_nonzero(board.board_array == 0)
                while not board.move(total_possible_actions[action]):
                    move_errors += 1
                next_state = qnn.encode_game_state(board.board_array)
                reward = qnn.calc_reward(next_state, prev_max_tile, prev_open_tiles, move_errors)
                running_loss += 9 - reward.item()
                replay_memory.append(Transition(state, action, next_state, reward))
                state = next_state

                if len(replay_memory) >= batch_size:
                    qnn.update_q_net(q_net, target_net, replay_memory, batch_size, Transition, optimizer, device, criterion, gamma)
                    if total_steps % target_update == 0:
                        qnn.update_target_net(target_net, q_net)
                        q_net.train()
                        torch.save(q_net.state_dict(), './q_net.pth')
                        torch.save(target_net.state_dict(), './target_net.pth')
            if episode % episode_interval == 0:
                print(f"episode: {episode}")
                print(board)
            loss_array.append(running_loss)
        qnn.visualize_loss(loss_array, num_episodes)
    q_net.eval()
    return epsilon_end, epsilon_decay, q_net, device, batch_size

def visualize_data(turns, max_tiles, min_tiles, piece_distributions, plt):
    plt.figure()

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
    plt.show()

def main():

    n = 1

    turns, max_tiles, min_tiles, piece_distributions = sample_random_games(n)
    visualize_data(turns, max_tiles, min_tiles, piece_distributions, plt)
    
    epsilon_end, epsilon_decay, q_net, device, batch_size = init_network()
    turns, max_tiles, min_tiles, piece_distributions = sample_network_games(n, epsilon_end, epsilon_decay, q_net, device, batch_size)
    visualize_data(turns, max_tiles, min_tiles, piece_distributions, plt)

    turns, max_tiles, min_tiles, piece_distributions = sample_mcts_games(n)
    visualize_data(turns, max_tiles, min_tiles, piece_distributions, plt)




if __name__ == "__main__":
    main()
