import random
import numpy as np
import math
from scipy.signal import fftconvolve
import game
import time
import sys
from multiprocessing import Pool
MAX_DEPTH = 7
MAX_ITERATIONS = 3
MAX_EXPANDED_NODES = 35

class Node(object):
    def __init__(self, move, board_array, parent=None):
        self.move = move
        self.times_played = 1
        self.score = 0
        self.parent = parent
        self.board_array = np.copy(board_array)
        self.depth = 1 if parent is None else parent.depth + 1

    def __str__(self):
        return f"Node (Move: {self.move}, depth: {self.depth},\t times played: {self.times_played},\ttotal score: {self.score})"

def apply_moves(board, moves):
    for i in moves:
        board.move(i)


def count_available_merges(board):
    merges = 0
    N = len(board)
    # Check horizontal merges
    for i in range(N):
        for j in range(N-1):
            if board[i][j] != 0 and board[i][j] == board[i][j+1]:
                merges += 1
    # Check vertical merges
    for j in range(N):
        for i in range(N-1):
            if board[i][j] != 0 and board[i][j] == board[i+1][j]:
                merges += 1
    return merges

def calc_edge_bonus(board):
    # The bonus is the sum of the values of unique tiles on the edges and corners.
    edges = np.concatenate((board[0, 1:-1], board[1:-1, -1], board[-1, 1:-1:1], board[1:-1, 0]))
    unique_values, counts = np.unique(board, return_counts=True)
    top_values = unique_values[:-board.shape[0]-1:-1]
    top_counts = counts[:-board.shape[0]-1:-1]
    bonus = 0
    # print(f"{top_values}, {top_counts}")
    # print(edges)
    for edge in edges:
        if edge in top_values and not edge == 0:
            bonus += math.log(edge, 2)
            # bonus += 1
    return bonus

def calc_corner_bonus(board):
    corners = np.array((board[0, 0], board[0, board.shape[1]-1], board[board.shape[0]-1, 0], board[board.shape[0]-1, board.shape[1]-1]))
    max_tile = np.max(board)
    max_tile_position = np.unravel_index(np.argmax(board), board.shape)
    # max_tile_position = [max_tile_position[0], max_tile_position[1]]
    bonus = 0
    corner_positions = np.array([(0,0), (board.shape[0]-1,board.shape[0]-1), (0,board.shape[0]-1), (board.shape[0]-1,0)])
    if max_tile in corners:
        # print(np.where(corner_positions == max_tile_position))
        # corner_positions = np.delete(corner_positions, np.where(corner_positions == max_tile_position))
        # print(corners)
        # print(corner_positions)
        # corners = np.delete(corners, )
        bonus += math.log(board[max_tile_position], 2)
        # for i in corner_positions:
        #     # value = board[i[0]][i[1]]
        #     print(i)
        #     # if board[i[0]][i[1]] > 0:
        #         # print(board[i[0]][i[1]])
        #         # bonus -= math.log(board[i[0]][i[1]], 2)
    return bonus

def calc_monotonicity(board):
    # Monotonicity measures how the values of the tiles change along the rows and columns of the board.
    # A board is considered more monotonic if the values of the tiles are either increasing or decreasing along each row and column.
    # The bonus is the sum of the signs of the differences between adjacent tiles.
    # A positive sign indicates that the adjacent tiles are either increasing or decreasing, while a negative sign indicates that they are changing in the opposite direction.
    # The bonus is higher if more adjacent tiles have the same sign, indicating a higher degree of monotonicity.
    differences = np.diff(board, axis=1)
    monotonicity = np.sum(np.sign(differences))
    return monotonicity

def calc_smoothness(board):
    # Smoothness measures how similar the values of adjacent tiles are.
    # A board is considered more smooth if the differences between adjacent tiles are smaller.
    # The bonus is the sum of the absolute differences between adjacent tiles.
    # A lower absolute difference indicates a smoother transition between tiles, resulting in a higher bonus.
    # The bonus is higher if the differences between adjacent tiles are consistently small, indicating a higher degree of smoothness.
    calc_board = [i + 1 for i in board]
    # calc_board = np.log2(calc_board)
    conv = fftconvolve(calc_board, calc_board[::-1], mode='same')
    # print(conv.astype(int))
    diff = np.diff(conv)
    # print(diff)
    smoothness = math.log(math.fabs(np.sum(diff)) + 1)
    return smoothness

def count_unique_tiles(board):
    # The bonus is the sum of the counts of each unique tile multiplied by its value.
    unique_values, counts = np.unique(board, return_counts=True)
    # print(f"{unique_values}, {counts}")
    bonus = math.log(sum(counts[i] * unique_values[i] for i in range(len(unique_values))), 2)
    return bonus

def eval_position(board, weights):
    values_sum          = np.sum(board.board_array)
    num_open_tiles      = np.count_nonzero(board.board_array == 0)
    # edge_bonus          = calc_edge_bonus(board.board_array)
    # corner_bonus        = calc_corner_bonus(board.board_array)
    # monotonicity        = calc_monotonicity(board.board_array)
    # smoothness          = calc_smoothness(board.board_array)
    # uniqueness_bonus    = count_unique_tiles(board.board_array)
    # num_merges          = count_available_merges(board.board_array)
    score = 0
    score += values_sum 
    # score += weights[1]*uniqueness_bonus 
    # score += weights[2]*corner_bonus 
    # score += weights[5]*edge_bonus
    # score *= weights[4]*monotonicity 
    # score *= weights[3]*smoothness 
    # score *= weights[6]*num_merges
    score *= weights[1]*num_open_tiles 
    return score

def expansion(node, board):
    depth = 0 if node is None else node.depth
    leaf_nodes = []
    if depth >= MAX_DEPTH:
        return leaf_nodes
    available_moves = board.get_available_moves()
    for i in available_moves:
        for _ in range(MAX_EXPANDED_NODES):
            simulated_board = game.Board(-1, board)
            simulated_board.move(i)
            new_node = None
            if node:
                new_node = Node(i, simulated_board.board_array, node)
            else:
                new_node = Node(i, simulated_board.board_array)
            leaf_nodes.append(new_node)
    return leaf_nodes

def simulation(selected_node, weights, num_moves=1, num_runs=1):
    moves = []
    simulated_board = game.Board()
    simulated_board.load_array(selected_node.board_array)
    total_score = 0
    for _ in range(num_runs):
        run_score = 0
        for __ in range(num_moves):
            if simulated_board.random_move():
                run_score += eval_position(simulated_board, weights)
            simulated_board.load_array(selected_node.board_array)
        total_score += run_score
    average_score = total_score / num_runs
    selected_node.score += average_score
    return selected_node

def update(selected_node):
    leaf_node = selected_node
    score = leaf_node.score
    selected_node = selected_node.parent
    while selected_node:
        selected_node.times_played += 1
        selected_node.score += score
        selected_node = selected_node.parent
    return leaf_node

def selection(leaf_nodes):
    selected_node = None
    if len(leaf_nodes) > 1:
        sum_leafs = sum(node.times_played for node in leaf_nodes)
        total_plays = 1 if sum_leafs >= 0 else sum_leafs
        score = lambda node: -1 if node.score == 0 else node.score
        # leaf_nodes.sort(key=lambda node: score(node))
        # leaf_nodes.sort(key=lambda node: math.sqrt((2 * math.log(total_plays)) / score(node)), reverse=True)
        leaf_nodes.sort(key=lambda node: score(node)/node.times_played + math.sqrt((2 * math.log(total_plays)) / node.times_played), reverse=True)
        # leaf_nodes.sort(key=lambda node: node.depth * node.score, reverse=True)
        selected_node = leaf_nodes[0]
    # sleep(random.random())
    return selected_node

def pick_best_move(board, weights=np.ones(7)):
    # print(f'finding best move from position {board}...')
    start_time = time.time()
    node = None 
    # print(f'initializing tree with root node {node}...')
    leaf_nodes = expansion(node, board)
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
            simulated_nodes = pool.starmap(simulation, args)
            leaf_nodes = list(map(update, simulated_nodes))

        # simulated_nodes = list(map(simulation, leaf_nodes))
        # #     # print(f'simulating game from leaf {i}...')
        # #     simulated_board = game.Board(-1, board)
        #     simulation(i)
        # #     # print(f'updating leaf \t{i.move} \tat depth \t{i.depth} \twith # children: \t{len(moves)}')
            # update(i)
        # print(f'selecting new node ({len(leaf_nodes)} leafs, {total_sims} total sims)...')
        # print(updated_nodes)
        # leaf_nodes = simulated_nodes
        new_node = selection(leaf_nodes)
        if not new_node:
            # print('could not select new leaf node, stopping search...')
            break
        selected_node = new_node
        # print(f'expanding tree at depth {selected_node.depth}...')
        new_nodes = expansion(selected_node, board)
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

def main():
    board = game.Board(3)
    # weights = [0.88895399, 0.24157116, 0.74744118, 1.01911034, 0.15860703, 0.80674962, 0.04410941]# Replace 3 with the number of heuristics
    weights = np.ones(8)
    for _ in range(1000):
    # while board.board_array.max() <= 2048:
        board = game.Board(3)
        board.start_game()
        # print(board)
        while not board.is_game_over():
            move = pick_best_move(board, weights)
            if move:
                board.move(move)
            # print(move)
            print(board)
        print(board)
    print("win!!!!")

if __name__ == "__main__":
    main()
