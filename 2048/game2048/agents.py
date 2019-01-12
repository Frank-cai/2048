import numpy as np
from game2048.data import myDataset
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch


def make_one_hot(data1, length):
    return (np.arange(length) == data1[:, None]).astype(np.integer)


class Agent:
    # Agent base

    def __init__(self, game, display=None):
        self.game = game
        self.display = display
        self.n_iter = 0

    def play(self, max_iter=np.inf, verbose=False, train=0, dataset=myDataset()):
        self.n_iter = 0
        while (self.n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            if train:
                onehotb = []
                tmpd = []
                tmp = self.game.board
                tmpd.append(direction)
                for i in range(4):
                    for j in range(4):
                        if tmp[i][j]== 0:
                            tmp[i][j] = 0
                        else:
                            tmp[i][j] = np.log2(tmp[i][j])
                    tmp[i] = np.array(tmp[i])
                    onehotb.append(make_one_hot(tmp[i], 12))
                tmpd = np.array(tmpd)
                tmpd = make_one_hot(tmpd, 4)
                dataset.add(onehotb, tmpd)
            self.game.move(direction)
            self.n_iter += 1
            if verbose:
                print("Iter: {}".format(self.n_iter))
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


class MyAgent(Agent):

    def __init__(self, game, display=None):
        super().__init__(game, display)
        from game2048.myAgent import Net
        from game2048.myAgent import Net2
        from game2048.myAgent import Net3

        self.net = Net()
        self.net1 = Net3()
        self.net2 = Net3()
        self.net3 = Net3()
        self.net4 = Net3()
        self.net5 = Net3()
        self.net6 = Net3()
        self.net7 = Net3()
        self.net8 = Net3()
        self.net_2 = Net2()
        self.net_3 = Net3()

        if torch.cuda.is_available():
            self.net = self.net.cuda()
        if torch.cuda.is_available():
            self.net_2 = self.net_2.cuda()
        if torch.cuda.is_available():
            self.net_3 = self.net_3.cuda()

        self.train_dataset = []

    def train3(self, BATCH_SIZE, NUM_EPOCHS):

        from game2048.myAgent import myDataset
        # ################################################################
        b = np.load("allb.npy")
        d = np.load("alld.npy")
        print("train3")
        print(len(b))
        print(len(d))
        # #################################################################
        self.train_dataset = myDataset(b, d)
        train_loader = DataLoader(self.train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        criterion = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            criterion = criterion.cuda()
        optimizer = optim.Adam(self.net_3.parameters(), lr=0.001)

        for epoch in range(NUM_EPOCHS):
            t = 0
            print("EPOCH", epoch+1)
            running_loss = 0.
            for board, direction in train_loader:
                t = t + 1
                board, direction = board.float(), direction.float()
                board = np.transpose(board, (0, 2, 1, 3))
                # print(board.size())
                board, direction = Variable(board), Variable(direction)
                if torch.cuda.is_available():
                    board = board.cuda()
                    direction = direction.cuda()
                optimizer.zero_grad()
                # board = torch.Tensor.reshape(board, BATCH_SIZE, 4, 4, 12)
                outputs = self.net_3(board)
                outputs = torch.Tensor.reshape(outputs, BATCH_SIZE, 4)
                direction = torch.Tensor.reshape(direction, BATCH_SIZE, 4)
                direction = direction.cpu().numpy()
                direction = np.array(direction)
                maxd = []
                for j in range(BATCH_SIZE):
                    max = 0
                    for i in range(4):
                        if direction[j][i] > direction[j][max]:
                            max = i
                    maxd.append(max)
                direction = torch.LongTensor(maxd)

                if torch.cuda.is_available():
                    direction = direction.cuda()
                loss = criterion(outputs, direction)
                # print(loss)
                running_loss = running_loss+float(loss)
                loss.backward()
                optimizer.step()
            print("runningloss=", running_loss/t)

    def step(self):
        tmp = self.game.board
        one_hot = []
        for i in range(4):
            for j in range(4):
                if tmp[i][j] == 0:
                    tmp[i][j] = 0
                else:
                    tmp[i][j] = np.log2(tmp[i][j])
            one_hot.append(make_one_hot(tmp[i], 12))

        # train model 1
        one_hot2 = np.transpose(one_hot, (2, 0, 1))

        # train model 2,3
        one_hot3 = np.transpose(one_hot, (1, 0, 2))

        # model 3 train
        '''
        one_hot3 = torch.Tensor(one_hot3)
        if torch.cuda.is_available():
            one_hot3 = torch.Tensor.cuda(one_hot3)
        one_hot3 = torch.Tensor.reshape(one_hot3, 1, 4, 4, 12)
        direction = self.net_3(one_hot3)
        direction = torch.Tensor.reshape(direction, 1, 4)
        direction = list(direction)
        max = 0
        for i in range(4):
            if direction[0][i] > direction[0][max]:
                max = i
        '''

        # model 2 train
        '''
        one_hot3 = torch.Tensor(one_hot3)
        if torch.cuda.is_available():
            one_hot3 = torch.Tensor.cuda(one_hot3)
        one_hot3 = torch.Tensor.reshape(one_hot3, 1, 4, 4, 12)
        direction = self.net_2(one_hot3)
        direction = torch.Tensor.reshape(direction, 1, 4)
        direction = list(direction)
        max = 0
        for i in range(4):
            if direction[0][i] > direction[0][max]:
                max = i
        '''

        # model 1 train
        '''
        one_hot2 = torch.Tensor(one_hot2)
        if torch.cuda.is_available():
            one_hot2 = torch.Tensor.cuda(one_hot2)
        one_hot2 = torch.Tensor.reshape(one_hot2, 1, 12, 4, 4)
        direction2 = self.net(one_hot2)
        # print (direction)
        # print(direction)
        direction2 = torch.Tensor.reshape(direction2, 1, 4)
        direction2 = list(direction2)
        # print(direction)
        max2 = 0
        for i in range(4):
            if direction2[0][i] > direction2[0][max2]:
                max2 = i
        '''

        # model 3 step
    
        max_cnt = [0, 0, 0, 0]
        one_hot3 = torch.Tensor(one_hot3)
        if torch.cuda.is_available():
            one_hot3 = torch.Tensor.cuda(one_hot3)
        one_hot3 = torch.Tensor.reshape(one_hot3, 1, 4, 4, 12)
        '''
        direction_test1 = self.net1(one_hot3)
        direction_test1 = torch.Tensor.reshape(direction_test1, 1, 4)
        direction_test1 = list(direction_test1)
        max_test1 = 0
        for i in range(4):
            if direction_test1[0][i] > direction_test1[0][max_test1]:
                max_test1 = i
        max_cnt[max_test1] = max_cnt[max_test1]+1
        '''

        direction_test2 = self.net2(one_hot3)
        direction_test2 = torch.Tensor.reshape(direction_test2, 1, 4)
        direction_test2 = list(direction_test2)
        max_test2 = 0
        for i in range(4):
            if direction_test2[0][i] > direction_test2[0][max_test2]:
                max_test2 = i
        max_cnt[max_test2] = max_cnt[max_test2] + 1
        '''
        direction_test3 = self.net3(one_hot3)
        direction_test3 = torch.Tensor.reshape(direction_test3, 1, 4)
        direction_test3 = list(direction_test3)
        max_test3 = 0
        for i in range(4):
            if direction_test3[0][i] > direction_test3[0][max_test3]:
                max_test3 = i
        max_cnt[max_test3] = max_cnt[max_test3] + 1
        '''
        direction_test4 = self.net4(one_hot3)
        direction_test4 = torch.Tensor.reshape(direction_test4, 1, 4)
        direction_test4 = list(direction_test4)
        max_test4 = 0
        for i in range(4):
            if direction_test4[0][i] > direction_test4[0][max_test4]:
                max_test4 = i
        max_cnt[max_test4] = max_cnt[max_test4] + 1

        direction_test5 = self.net5(one_hot3)
        direction_test5 = torch.Tensor.reshape(direction_test5, 1, 4)
        direction_test5 = list(direction_test5)
        max_test5 = 0
        for i in range(4):
            if direction_test5[0][i] > direction_test5[0][max_test5]:
                max_test5 = i
        max_cnt[max_test5] = max_cnt[max_test5] + 1

        direction_test6 = self.net6(one_hot3)
        direction_test6 = torch.Tensor.reshape(direction_test6, 1, 4)
        direction_test6 = list(direction_test6)
        max_test6 = 0
        for i in range(4):
            if direction_test6[0][i] > direction_test6[0][max_test6]:
                max_test6 = i
        max_cnt[max_test6] = max_cnt[max_test6] + 1


        direction_test7 = self.net7(one_hot3)
        direction_test7 = torch.Tensor.reshape(direction_test7, 1, 4)
        direction_test7 = list(direction_test7)
        max_test7 = 0
        for i in range(4):
            if direction_test7[0][i] > direction_test7[0][max_test7]:
                max_test7 = i
        max_cnt[max_test7] = max_cnt[max_test7] + 1
        '''
        direction_test8 = self.net8(one_hot3)
        direction_test8 = torch.Tensor.reshape(direction_test8, 1, 4)
        direction_test8 = list(direction_test8)
        max_test8 = 0
        for i in range(4):
            if direction_test8[0][i] > direction_test8[0][max_test8]:
                max_test8 = i
        max_cnt[max_test8] = max_cnt[max_test8] + 1
        '''
        max_out = 0
        for i in range(4):
            if max_cnt[i] > max_out:
                max_out = i

        # print(max_cnt)
        # return int(max)
        return int(max_out)
