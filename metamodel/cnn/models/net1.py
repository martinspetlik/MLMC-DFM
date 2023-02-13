import torch
import torch.nn as nn
import torch.nn.functional as F


class Net1(nn.Module):

    def __init__(self):
        super(Net1, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(6, 6, 3)
        self.conv1_bn = nn.BatchNorm2d(6)

        self.conv2 = nn.Conv2d(6, 6, 3)
        self.conv2_bn = nn.BatchNorm2d(6)

        self.conv3 = nn.Conv2d(6, 6, 3)
        self.conv3_bn = nn.BatchNorm2d(6)

        self.conv4 = nn.Conv2d(6, 6, 3)
        self.conv4_bn = nn.BatchNorm2d(6)

        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(6 * 14 * 14, 520)
        self.fc1_bn = nn.BatchNorm1d(520)
        self.fc2 = nn.Linear(520, 128)
        self.fc2_bn = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 32)
        self.fc3_bn = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 6)

    def forward(self, x):
        # ==============
        # Layer 1
        # ==============
        x = self.conv1(x)
        x = F.relu(self.conv1_bn(x))
        x = F.avg_pool2d(x, 2)

        # sum_kernel = torch.ones(1, 6, 2, 2)
        # x = F.conv2d(x, sum_kernel, stride=1, padding=1)

        # ==============
        # Layer 2
        # ==============
        x = self.conv2(x)
        x = F.relu(self.conv2_bn(x))
        x = F.avg_pool2d(x, 2)

        x = self.dropout(x)

        # ==============
        # Layer 3
        # ==============
        x = self.conv3(x)
        x = F.relu(self.conv3_bn(x))
        x = F.avg_pool2d(x, 2)

        x = self.dropout(x)

        # ==============
        # Layer 4
        # ==============
        x = self.conv4(x)
        x = F.relu(self.conv4_bn(x))
        x = F.avg_pool2d(x, 2)

        x = self.dropout(x)

        x = torch.flatten(x, 1)

        # ==============
        # Layer 4
        # ==============
        x = self.fc1(x)
        x = F.relu(self.fc1_bn(x))

        # ==============
        # Layer 5
        # ==============
        x = self.fc2(x)
        x = F.relu(self.fc2_bn(x))

        # ==============
        # Layer 6
        # ==============
        x = self.fc3(x)
        x = F.relu(self.fc3_bn(x))

        # ==============
        # Output layer
        # ==============
        x = self.fc4(x)
        return x
