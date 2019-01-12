from __future__ import print_function, division
import numpy as np
from torch.utils.data import Dataset, DataLoader
from game2048.game import Game
from game2048.displays import Display


class myDataset(Dataset):

    def __init__(self):
        self.board = []
        self.direction = []

    def add(self, getboard, getdirection):
        self.board.append(getboard)
        self.direction.append(getdirection)

    def __len__(self):
        return len(self.board)

    def __getitem__(self, idx):
        myBoard = self.board[idx]
        myDirection = self.direction[idx]
        return myBoard, myDirection


def single_run(size, ds, AgentClass, **kwargs):
    game = Game(size, 2048)
    agent = AgentClass(game, display=Display(), **kwargs)
    agent.play(dataset=ds, verbose=False, train=1)


if __name__ == '__main__':
    GAME_SIZE = 4
    N_TESTS = 1000

    '''====================
    Use your own agent here.'''
    from game2048.agents import ExpectiMaxAgent as TestAgent
    '''===================='''

    scores = []
    dataset = myDataset()
    for _ in range(N_TESTS):

        single_run(GAME_SIZE, ds=dataset, AgentClass=TestAgent)
    np.save("b16", dataset.board)
    np.save("d16", dataset.direction)
    print(len(dataset))

