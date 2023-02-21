import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, trial=None, pool=None, max_channel=3, kernel_size=3, stride=3, use_dropout=False, input_size=256, min_channel=3):

        super(Net, self).__init__()
        self._name = "pure_cnn_net"
        self._use_dropout = use_dropout
        self._pool = pool
        self._convs = nn.ModuleList()

        n_layers = 0
        while True:
            input_size = int(((input_size - kernel_size) / stride)) + 1
            n_layers += 1

            if input_size == 1:
                break
            elif input_size < 1:
                raise ValueError("Stride and kernel size result in inappropriate number of final pixels: {}".format(input_size))

        channels = np.linspace(start=min_channel, stop=max_channel, num=n_layers+1, dtype=int)

        for i in range(n_layers):
            self._convs.append(nn.Conv2d(in_channels=channels[i],
                                        out_channels=channels[i+1],
                                        kernel_size=kernel_size,
                                        stride=stride))

            #print("in_channels: {}, out_channels: {}".format(channels[i - 1], channels[i]))


    def forward(self, x):
        for i, conv_i in enumerate(self._convs):  # For each convolutional layer
            if self._pool == "max":
                pool = F.max_pool2d
            elif self._pool == "avg":
                pool = F.avg_pool2d

            if i == len(self._convs)-1:
                x = conv_i(x)
            else:
                if self._pool is not None:
                    if self._use_dropout and i == 2:
                        x = F.relu(pool(self.conv2_drop(conv_i(x)), 2))
                else:
                    x = F.relu(conv_i(x))

        return x
