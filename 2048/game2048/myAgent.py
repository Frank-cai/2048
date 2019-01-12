import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class myDataset(Dataset):

    def __init__(self, getboard, getdirection):
        self.board = getboard
        self.direction = getdirection

    def add(self, getboard, getdirection):
        self.board.append(getboard)
        self.direction.append(getdirection)

    def __len__(self):
        return len(self.direction)

    def __getitem__(self, idx):
        myBoard = self.board[idx]
        myDirection = self.direction[idx]
        return myBoard, myDirection


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        '''input size (12,4,4)'''
        self.conv1 = nn.Conv2d(12, 100, kernel_size=(1, 4), stride=1)
        self.conv2 = nn.Conv2d(100, 400, kernel_size=(4, 1), stride=1)
        self.conv3 = nn.Conv2d(400, 600, kernel_size=2, stride=1, padding=1)
        self.conv4 = nn.Conv2d(600, 400, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(400, 4, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = self.conv5(out)
        out = F.softmax(out, dim=1)
        return out


class Net2(nn.Module):

    def __init__(self):
        super(Net2, self).__init__()

        '''input size (4,4,12)'''
        self.conv1 = nn.Conv2d(4, 100, kernel_size=(1, 4), stride=1)
        self.conv2 = nn.Conv2d(100, 400, kernel_size=(4, 1), stride=1, padding=1)
        self.conv3 = nn.Conv2d(400, 400, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(400, 200, kernel_size=(1, 6), stride=1)
        self.conv5 = nn.Conv2d(200, 4, kernel_size=(1, 4), stride=1)

    def forward(self, x):

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = self.conv4(out)
        out = self.conv5(out)
        out = F.softmax(out, dim=1)
        return out


class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()

        self.conv1 = nn.Conv2d(4, 64, kernel_size=(4, 1), padding=(2, 0))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(1, 4), padding=(0, 2))
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(2, 2))
        self.conv4 = nn.Conv2d(256, 128, kernel_size=(3, 3), padding=(1, 1))
        self.conv5 = nn.Conv2d(128, 128, kernel_size=(4, 4), padding=(2, 2))
        # self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(128 * 5 * 13, 2000)
        self.fc2 = nn.Linear(2000, 500)
        self.fc3 = nn.Linear(500, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(-1, 128 * 5 * 13)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x
