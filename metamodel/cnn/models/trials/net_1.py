import torch
import torch.nn as nn
import torch.nn.functional as F


class Net1(nn.Module):

    def __init__(self):
        super(Net1, self).__init__()

        self._name = "net_1"
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding="valid")
        self.conv1_bn = nn.BatchNorm2d(3)

        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding="valid")
        self.conv2_bn = nn.BatchNorm2d(3)

        self.conv3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding="valid")
        self.conv3_bn = nn.BatchNorm2d(3)

        self.fc1 = nn.Linear(3 * 30 * 30, 3)


    def forward(self, x):
        # ==============
        # Layer 1
        # ==============
        x = self.conv1(x)
        x = F.relu(self.conv1_bn(x))
        x = F.avg_pool2d(x, 2)

        # ==============
        # Layer 2
        # ==============
        x = self.conv2(x)

        x = F.relu(self.conv2_bn(x))
        x = F.avg_pool2d(x, 2)

        # ==============
        # Layer 3
        # ==============
        x = self.conv3(x)
        x = F.relu(self.conv3_bn(x))
        x = F.avg_pool2d(x, 2)

        x = torch.flatten(x, 1)

        # ==============
        # Layer 4
        # ==============
        x = self.fc1(x)

        return x
