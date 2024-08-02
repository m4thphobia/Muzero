from network import *
from tqdm import trange
import numpy as np
import ray
from game import *
import torch
import random
from mcts import *


def print_board(state, i):

    board = state[0] - state[1]

    char_board = np.vectorize(index_to_char)(board)
    with open(f'../out/board/turn{i}.txt', 'w') as f:
        f.write("  0  1  2  3  4  5  6  7\n")
        for i in range(8):
            f.write(f'{i}')
            for j in range(8):
                f.write(f' {char_board[i][j]} ')
            f.write('\n')


def print_board(state):

    board = state[0] - state[1]

    char_board = np.vectorize(index_to_char)(board)
    with open(f'../out/board/turn{i}.txt', 'w') as f:
        f.write("  0  1  2  3  4  5  6  7\n")
        for i in range(8):
            f.write(f'{i}')
            for j in range(8):
                f.write(f' {char_board[i][j]} ')
            f.write('\n')


def index_to_char(index):
    if index == 1:
        return '●'
    elif index == 0:
        return '-'
    elif index == -1:
        return '○'
    else:
        print(index)
        raise Exception("これは例外です")


if __name__ == '__main__':

    network = Network()
    network.load_state_dict(torch.load('./var/muzero_vae.pth', map_location=torch.device('cpu')))

    board = np.array([[0,0,1,1,1,1,1,0],
                    [0,0,1,-1,-1,1,0,0],
                    [0,0,-1,-1,1,1,-1,-1],
                    [0,0,-1,-1,1,-1,1,0],
                    [0,0,-1,-1,-1,-1,1,1],
                    [0,0,-1,-1,-1,1,1,1],
                    [0,0,-1,-1,-1,1,1,1],
                    [0,0,0,0,0,0,0,0]])

    print(board.shape)
    print('^^')