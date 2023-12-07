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
import pandas as pd
import sys
import os
from collections import namedtuple, deque

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
    device = torch.device("cpu")
    if torch.cuda.is_available(): 
        print(f"using device {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda:0")
    else:
        print(f"CUDA not available, using device CPU...")

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
    if not os.path.exists('./q_net.pth') or not os.path.exists('./target_net.pth'):
        num_episodes = 100
        print(f"Pre-trained network not found, training for {num_episodes} episodes...")
    else:
        q_net.load_state_dict(torch.load('./q_net.pth'))
        target_net.load_state_dict(torch.load('./target_net.pth'))
        print("Using pre-trained network...")
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
                        # torch.save(q_net.state_dict(), './q_net.pth')
                        # torch.save(target_net.state_dict(), './target_net.pth')
            if episode % episode_interval == 0:
                print(f"episode: {episode}")
                print(board)
            loss_array.append(running_loss)
        qnn.visualize_loss(loss_array, num_episodes)
    q_net.eval()
    return epsilon_end, epsilon_decay, q_net, device, batch_size

def visualize_data(turns, max_tiles, min_tiles, piece_distributions, filename):
    plt.figure()

    # Plot histogram of turns
    plt.hist(turns)
    plt.title(f'Number of turns taken in each game, mean: {round(np.mean(turns), 4)}, std: {round(np.std(turns), 4)}')
    plt.savefig(filename + "_num_turns")

    # Plot histogram of max tiles
    plt.figure()
    plt.hist(max_tiles, bins=range(0, max(max_tiles)))
    plt.title(f'Log(Max tile, 2) value in each game, mean: {round(np.mean(max_tiles), 4)}, std: {round(np.std(max_tiles), 4)}')
    plt.savefig(filename + "_max_tile")

    # Plot histogram of min tiles
    plt.figure()
    plt.hist(min_tiles, bins=20)
    plt.title(f'Min tile value in each game')
    plt.savefig(filename + "_min_tile")

    # Plot bar chart of piece distributions
    plt.figure()
    all_tiles = [tile for dist in piece_distributions for tile in dist]
    all_counts = [count for dist in piece_distributions for count in dist.values()]
    plt.bar(all_counts, all_tiles)
    plt.title(f'Tile distributions across all games')
    plt.savefig(filename + "_tile_dist")

    # Show the plot
    # plt.show()

def main():
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    else:
        n = 1

    print(f"Running random, QNN, and MCTS policies on {n} games...")
    print(f"\nTesting random on {n} runs...")
    turns_random, max_tiles_random, min_tiles_random, piece_distributions_random = sample_random_games(n)
    print("Random test complete")
    visualize_data(turns_random, max_tiles_random, min_tiles_random, piece_distributions_random, f"./Results/random_{n}")
    
    print(f"\nTesting QNN...")
    print(f"Initializizing neural network...")
    epsilon_end, epsilon_decay, q_net, device, batch_size = init_network()
    print(f"Testing QNN on {n} runs...")
    turns_qnn, max_tiles_qnn, min_tiles_qnn, piece_distributions_qnn = sample_network_games(n, epsilon_end, epsilon_decay, q_net, device, batch_size)
    print("Qnn test complete")
    visualize_data(turns_qnn, max_tiles_qnn, min_tiles_qnn, piece_distributions_qnn, f"./Results/qnn_{n}")

    print(f"\nTesting MCTS...")
    print(f"Testing MCTS on {n} runs...")
    turns_mcts, max_tiles_mcts, min_tiles_mcts, piece_distributions_mcts = sample_mcts_games(n)
    print("MCTS test complete")
    visualize_data(turns_mcts, max_tiles_mcts, min_tiles_mcts, piece_distributions_mcts, f"./Results/mcts_{n}")

    print("Saving results...")
    df_random = pd.DataFrame({
        'turns': turns_random,
        'max_tiles': max_tiles_random,
        'min_tiles': min_tiles_random,
        'piece_distributions': piece_distributions_random
    })
    df_qnn = pd.DataFrame({
        'turns': turns_qnn,
        'max_tiles': max_tiles_qnn,
        'min_tiles': min_tiles_qnn,
        'piece_distributions': piece_distributions_qnn
    })
    df_mcts = pd.DataFrame({
        'turns': turns_mcts,
        'max_tiles': max_tiles_mcts,
        'min_tiles': min_tiles_mcts,
        'piece_distributions': piece_distributions_mcts
    })

    df_random.to_csv(f'./Results/random_{n}_data.csv', index=False)
    df_qnn.to_csv(f'./Results/qnn_{n}_data.csv', index=False)
    df_mcts.to_csv(f'./Results/mcts_{n}_data.csv', index=False)
    print("Results have been saved to ./Results/")

if __name__ == "__main__":
    main()
