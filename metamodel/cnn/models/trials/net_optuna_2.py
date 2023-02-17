import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, trial, num_conv_layers, pool, max_channel, kernel_size, stride):

        super(Net, self).__init__()
        input_size = 256
        min_channel = 3

        n_layers = 0
        while True:
            input_size = int(((input_size - kernel_size) / stride)) + 1
            n_layers += 1

            if input_size == 1:
                break


        channels = np.linspace(start=min_channel, stop=max_channel, num=n_layers, dtype=int)

        for i in range(n_layers):
            self._convs.append(nn.Conv2d(in_channels=channels[i],
                                             out_channels=channels[i+1],
                                             kernel_size=kernel_size,
                                             stride=stride)
                                   )



            print("in_channels: {}, out_channels: {}".format(channels[i - 1], channels[i]))

        self._pool = pool


        for i in range(1, num_conv_layers):
            if same_channels:
                pass
            else:

                out_size = out_size - kernel_size + 1                                       # Size of the output kernel
                out_size = int(out_size/2)                                                  # Size after pooling


        for i in range(1, num_conv_layers):
            nn.init.kaiming_normal_(self.convs[i].weight, nonlinearity='relu')
            if self._convs[i].bias is not None:
                nn.init.constant_(self.convs[i].bias, 0)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')

    def forward(self, x):
        for i, conv_i in enumerate(self._convs):  # For each convolutional layer
            if self._pool == "max":
                pool = F.max_pool2d
            elif self._pool == "avg":
                pool = F.avg_pool2d

            if self._pool is not None:
                #@TODO: flag for using dropout
                if i == 2:  # Add dropout if layer 2
                    x = F.relu(pool(self.conv2_drop(conv_i(x)), 2))
                else:
                    x = F.relu(pool(conv_i(x), 2))

        return x
