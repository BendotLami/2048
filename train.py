import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class NN_2048(nn.Module):
    def __init__(self):
        super(NN_2048, self).__init__()
        self.conv1_1 = nn.Conv2d(16, 128, kernel_size=(1, 2))
        self.conv1_2 = nn.Conv2d(16, 128, kernel_size=(2, 1))
        self.relu1_1 = nn.ReLU(inplace=True)
        self.relu1_2 = nn.ReLU(inplace=True)

        self.conv2_1 = nn.Conv2d(128, 128, kernel_size=(1, 2))
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=(2, 1))
        self.relu2_1 = nn.ReLU(inplace=True)
        self.relu2_2 = nn.ReLU(inplace=True)

        self.conv2_3 = nn.Conv2d(128, 128, kernel_size=(1, 2))
        self.conv2_4 = nn.Conv2d(128, 128, kernel_size=(2, 1))
        self.relu2_3 = nn.ReLU(inplace=True)
        self.relu2_4 = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(7424, 256)
        self.relu_3 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        conv_1 = self.conv1_1(x)
        conv_2 = self.conv1_2(x)
        self.relu1_1(conv_1)
        self.relu1_2(conv_2)

        conv_3 = self.conv2_1(conv_1)
        conv_4 = self.conv2_2(conv_1)
        self.relu2_1(conv_3)
        self.relu2_2(conv_4)
        conv_5 = self.conv2_3(conv_2)
        conv_6 = self.conv2_4(conv_2)
        self.relu2_3(conv_5)
        self.relu2_4(conv_6)

        flatten_1 = conv_1.flatten(1)
        flatten_2 = conv_2.flatten(1)
        flatten_3 = conv_3.flatten(1)
        flatten_4 = conv_4.flatten(1)
        flatten_5 = conv_5.flatten(1)
        flatten_6 = conv_6.flatten(1)

        hidden = torch.cat([flatten_1, flatten_2, flatten_3, flatten_4, flatten_5, flatten_6], 1)
        hidden = self.fc1(hidden)
        self.relu_3(hidden)
        x = self.fc2(hidden)

        return x


if __name__ == '__main__':
    arr = np.ones([4, 4, 16], dtype=float)
    arr = np.moveaxis(arr, 2, 0)
    arr = np.expand_dims(arr, axis=0)  # single batch
    tensor = torch.from_numpy(arr)

    model = NN_2048()
    model.double()

    print(model(tensor).shape)
