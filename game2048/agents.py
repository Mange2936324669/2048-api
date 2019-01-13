import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms as transforms
from torch import nn
import pandas as pd
import torch
from PythonApplication2 import My_dataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
from game2048.planning import nextMove


class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction

class Rnn_Agent(Agent):
    def __init__(self,game,display = None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        self.model = torch.load('game2048.rnn.pkl',map_location=lambda storage, loc:storage)
        #self.model2 = torch.load('/home/noone/桌面/2048-api-master/game2048/rnn+512.pkl',map_location=lambda storage, loc:storage)
    def step(self):
            #flag = 0
            board = self.game.board
            board[board == 0] = 1
            board = np.log2(board).flatten()
            #if(board.max()>=11):
                #flag = 0
            #else:
                #flag = 1
            board = np.int32(board)
            #if(flag == 1):
            board = board/11
            #else: 
                #board = board/12
            board = torch.from_numpy(board).type(torch.FloatTensor)
            board = board.type(torch.float)
            board = board.view(-1,4,4)
            #if(flag == 1):
            test_output = self.model(board)
            #else:
                #test_output = self.model2(board)
            pred_y = torch.max(test_output,1)[1]
            direction = pred_y
            return int(direction)

class plan(Agent):
    def __init__(self,game,display = None):
            if game.size != 4:
                raise ValueError(
                    "`%s` can only work with game of `size` 4." % self.__class__.__name__)
            super().__init__(game, display)

            self.search_func = nextMove

    def step(self):
        direction = self.search_func(self.game.board)
        return direction