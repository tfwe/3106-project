import game
import qnn
import mcts
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
import json
import time
from multiprocessing import Pool
from collections import namedtuple, deque

def sample_random_games(n):
    turns = []
    max_tiles = []
    min_tiles = []
    piece_distributions = []
    for i in range(n):
        board = game.Board(size=3)
        board.start_game()
        num_pieces = {}
        while not board.is_game_over():
            for tile in (np.ravel(board.board_array)):
                if num_pieces.get(tile) == None:
                    num_pieces[tile] = 1
                    continue
                num_pieces[tile] += 1
            board.random_move()
        print(f"(Random Run {i+1} / {n})\n{board}")
        turns.append(board.turn)
        max_tiles.append(np.max(np.ravel(board.board_array)))
        min_tiles.append(np.min(np.ravel(board.board_array)))

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
        num_pieces = {}
        while not board.is_game_over():
            for tile in (np.ravel(board.board_array)):
                if num_pieces.get(tile) == None:
                    num_pieces[tile] = 1
                    continue
                num_pieces[tile] += 1
            action = pick_best_move(board)
            # action = np.random.choice(board.get_available_moves())
            board.move(action)
        print(f"(MCTS Run {i+1} / {n})\n{board}")
        turns.append(board.turn)
        max_tiles.append(np.max(np.ravel(board.board_array)))
        min_tiles.append(np.min(np.ravel(board.board_array)))
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
        num_pieces = {}
        while not board.is_game_over():
            for tile in (np.ravel(board.board_array)):
                if num_pieces.get(tile) == None:
                    num_pieces[tile] = 1
                    continue
                num_pieces[tile] += 1
            epsilon = max(epsilon_end, epsilon_decay * i)
            action = qnn.select_action([board.board_array], epsilon, q_net, device, batch_size)
            total_possible_actions = ["up","down","left","right"]
            board.move(total_possible_actions[action])
        print(f"(QNN Run {i+1} / {n})\n{board}")
        turns.append(board.turn)
        max_tiles.append(np.max(np.ravel(board.board_array)))
        min_tiles.append(np.min(np.ravel(board.board_array)))
        piece_distributions.append(num_pieces)
        max_tiles = [int(log_2_neg(max_tile)) for max_tile in max_tiles]
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
        num_episodes = 250
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
                        torch.save(q_net.state_dict(), './q_net.pth')
                        torch.save(target_net.state_dict(), './target_net.pth')
            if episode % episode_interval == 0:
                print(f"episode: {episode}")
                print(board)
            loss_array.append(running_loss)
        qnn.visualize_loss(loss_array, num_episodes)
    q_net.eval()
    return epsilon_end, epsilon_decay, q_net, device, batch_size

def pick_best_move(board, weights=np.ones(7)):
    MAX_DEPTH = 1
    MAX_ITERATIONS = 1
    MAX_EXPANDED_NODES = 4


    # print(f'finding best move from position {board}...')
    start_time = time.time()
    node = None 
    # print(f'initializing tree with root node {node}...')
    leaf_nodes = mcts.expansion(node, board)
    selected_node = None
    # count = 0
    # while len(leaf_nodes) >= 1:
    #     count += 1
    #     if count >= MAX_ITERATIONS:
    #         break
    for _ in range(MAX_ITERATIONS): # num iterations per turn 
        # print(f'exploring {len(leaf_nodes)} nodes...')
        simulated_nodes = [] 
        with Pool() as pool:
            # for result in pool.imap_unordered(simulation, leaf_nodes):
            #     simulated_nodes.append(result)
            args = []
            for leaf_node in leaf_nodes:
                args.append((leaf_node, weights)) 
            simulated_nodes = pool.starmap(mcts.simulation, args)
            leaf_nodes = list(map(mcts.update, simulated_nodes))

        new_node = mcts.selection(leaf_nodes)
        if not new_node:
            # print('could not select new leaf node, stopping search...')
            break
        selected_node = new_node
        # print(f'expanding tree at depth {selected_node.depth}...')
        new_nodes = mcts.expansion(selected_node, board)
        leaf_nodes.remove(selected_node)
        if len(new_nodes) > 0:
            leaf_nodes.extend(new_nodes)
        # print(f'expanded tree with {len(new_nodes)} new nodes (total leafs: {len(leaf_nodes)})')
    # print('search stopped')
    if selected_node:
        current_node = selected_node
        # print(f'selected: {selected_node}')
        while current_node.parent:
            current_node = current_node.parent
        # print(f'best move: {current_node.move} from depth {selected_node.depth},\t times played: {current_node.times_played},\ttotal score: {current_node.score}')
        end_time = time.time()
        # print(f'time elapsed: {end_time - start_time}')
        return current_node.move
    return
def log_2_neg(tile):
    if tile > 0:
        return int(math.log(tile, 2))
    if tile == 0:
        return 0
    return (-1)*(int(math.log(math.fabs(tile), 2)))

def visualize_data(turns, max_tiles, min_tiles, piece_distributions, filename, policy="Random"):
    plt.figure()

    # Plot histogram of turns
    plt.hist(turns)
    plt.title(f'{policy} - Number of Turns Taken Across All Games\nmax: {max(turns)}, min: {min(turns)}, mean: {round(np.mean(turns), 4)}, std: {round(np.std(turns), 4)}')
    plt.xlabel("Number of Turns Taken")
    plt.ylabel("Number of Games Played")
    plt.savefig(filename + "_num_turns")

    # Plot histogram of max tiles
    plt.figure()
    plt.hist(max_tiles)
    plt.title(f'Max Value Tile Across All Games\nmax: {max(max_tiles)}, min: {min(max_tiles)}, mean: {round(np.mean(max_tiles), 4)}, std: {round(np.std(max_tiles), 4)}')
    plt.xlabel("Maximum Value Tile (log2(max_tile))")
    plt.ylabel("Number of Games")
    plt.savefig(filename + "_max_tile")

    # Plot histogram of min tiles
    plt.figure()
    min_tiles = [int(log_2_neg(min_tile)) for min_tile in min_tiles]
    plt.hist(min_tiles)
    plt.title(f'Min Value Tile Across All Games\nmax: {max(min_tiles)}, min: {min(min_tiles)}, mean: {round(np.mean(min_tiles), 4)}, std: {round(np.std(min_tiles), 4)}')
    plt.xlabel("Minimum Value Tile (log2 of value with 0s and negatives representation")
    plt.ylabel("Number of Games Played")
    plt.savefig(filename + "_min_tile")

    # Plot bar chart of piece distributions
    plt.figure()
    all_tiles = [log_2_neg(tile) for dist in piece_distributions for tile in dist]
    all_counts = [count for dist in piece_distributions for count in dist.values()]
    plt.bar(all_tiles, all_counts)
    plt.xlabel("Tile Value (log2 of value with 0s and negatives representation)")
    plt.ylabel("Number of Times Counted")
    plt.title(f'Tile Distributions Across All Games')
    plt.savefig(filename + "_tile_dist")

    # Show the plot
    plt.show()

def csv_to_arrays(csv_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Extract the columns into separate arrays
    turns_random = df['turns'].values
    max_tiles_random = df['max_tiles'].values
    min_tiles_random = df['min_tiles'].values
    piece_distributions_random = df['piece_distributions'].values
    piece_distributions_random = [eval(x) for x in piece_distributions_random]

    return turns_random, max_tiles_random, min_tiles_random, piece_distributions_random

def main():
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    else:
        n = 1

    print(f"Running random, QNN, and MCTS policies on {n} games...")
    print(f"\nTesting random on {n} runs...")

    # turns_random, max_tiles_random, min_tiles_random, piece_distributions_random = sample_random_games(n)
    # turns_random, max_tiles_random, min_tiles_random, piece_distributions_random = csv_to_arrays("./Data/random_1000_data.csv")
    df_random = pd.DataFrame({
        'turns': turns_random,
        'max_tiles': max_tiles_random,
        'min_tiles': min_tiles_random,
        'piece_distributions': piece_distributions_random
    })
    visualize_data(turns_random, max_tiles_random, min_tiles_random, piece_distributions_random, f"./Results/random_{n}", policy="Random")
    df_random.to_csv(f'./Results/random_{n}_data.csv', index=False)
    print("Random test complete")

    print(f"\nTesting QNN...")
    print(f"Initializizing neural network...")
    epsilon_end, epsilon_decay, q_net, device, batch_size = init_network()
    print(f"Testing QNN on {n} runs...")
    # turns_qnn, max_tiles_qnn, min_tiles_qnn, piece_distributions_qnn = csv_to_arrays("./Data/qnn_1000_data.csv")
    turns_qnn, max_tiles_qnn, min_tiles_qnn, piece_distributions_qnn = sample_network_games(n, epsilon_end, epsilon_decay, q_net, device, batch_size)
    df_qnn = pd.DataFrame({
        'turns': turns_qnn,
        'max_tiles': max_tiles_qnn,
        'min_tiles': min_tiles_qnn,
        'piece_distributions': piece_distributions_qnn
    })
    visualize_data(turns_qnn, max_tiles_qnn, min_tiles_qnn, piece_distributions_qnn, f"./Results/qnn_{n}", policy="QNN")
    df_qnn.to_csv(f'./Results/qnn_{n}_data.csv', index=False)
    print("QNN test complete")

    print(f"\nTesting MCTS...")
    print(f"Testing MCTS on {n} runs...")
    # turns_mcts, max_tiles_mcts, min_tiles_mcts, piece_distributions_mcts = csv_to_arrays("./Data/mcts_1000_data.csv")
    turns_mcts, max_tiles_mcts, min_tiles_mcts, piece_distributions_mcts = sample_mcts_games(n)
    df_mcts = pd.DataFrame({
        'turns': turns_mcts,
        'max_tiles': max_tiles_mcts,
        'min_tiles': min_tiles_mcts,
        'piece_distributions': piece_distributions_mcts
    })
    visualize_data(turns_mcts, max_tiles_mcts, min_tiles_mcts, piece_distributions_mcts, f"./Results/mcts_{n}", policy="MCTS")
    df_mcts.to_csv(f'./Results/mcts_{n}_data.csv', index=False)
    print("MCTS test complete")

    print("Results have been saved to ./Results/")

if __name__ == "__main__":
    main()
