import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, trial=None, n_conv_layers=3, max_channel=3, pool=None, kernel_size=3, stride=1, pool_size=2,
                 pool_stride=2, use_batch_norm=True, n_hidden_layers=1, max_hidden_neurons=520,
                 hidden_activation=F.relu, input_size=256, min_channel=3, use_dropout=False, conv_layer_obj=[],
                 output_layer=True):
        super(Net, self).__init__()
        self._name = "cnn_net"
        self._use_dropout = use_dropout
        self._pool = pool
        self._pool_size = pool_size
        self._pool_stride = pool_stride
        self._convs = nn.ModuleList()
        self._hidden_layers = nn.ModuleList()
        self._use_batch_norm = use_batch_norm
        self._hidden_activation = hidden_activation
        self._batch_norms = nn.ModuleList()
        self._conv_layer_obj = conv_layer_obj
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)

        if self._pool == "None":
            self._pool = None

        if type(max_channel) == list and max_channel == len(n_conv_layers):
            channels = max_channel
        else:
            channels = np.linspace(start=min_channel, stop=max_channel, num=n_conv_layers+1, dtype=int)

        #print("conv layer obj ", conv_layer_obj)

        for i in range(n_conv_layers):
            if len(conv_layer_obj) > 0:
                if len(conv_layer_obj) > i:
                    self._convs.append(conv_layer_obj[i])
                else:
                    raise Exception("Not enough conv layer objects")
            else:
                self._convs.append(nn.Conv2d(in_channels=channels[i],
                                             out_channels=channels[i + 1],
                                             kernel_size=kernel_size,
                                             stride=stride))
            if self._use_batch_norm:
                self._batch_norms.append(nn.BatchNorm2d(channels[i + 1]))

            input_size = int(((input_size - self._convs[-1].kernel_size[0]) / self._convs[-1].stride[0])) + 1

            if self._pool is not None:
                input_size = int(((input_size - pool_size) / pool_stride)) + 1

        #print("input size ", input_size)

        if type(max_hidden_neurons) == list and len(max_hidden_neurons) == n_hidden_layers:
            hidden_neurons = max_hidden_neurons
        else:
            hidden_neurons = np.linspace(start=max_hidden_neurons, stop=min_channel, num=n_hidden_layers, dtype=int)

        # print("self._convs[-1] ", self._convs[-1])
        # print("self._convs[-1].out_channels ", self._convs[-1].out_channels)
        input_size = self._convs[-1].out_channels * input_size * input_size

        #print("input size to hidden layers ", input_size)

        for i in range(n_hidden_layers):
            self._hidden_layers.append(nn.Linear(input_size, hidden_neurons[i]))
            input_size = hidden_neurons[i]

        if output_layer:
            self._output_layer = nn.Linear(input_size, min_channel, bias=False)

        self.out_channels = min_channel

    def forward(self, x):
        for i, conv_i in enumerate(self._convs):  # For each convolutional layer
            if self._pool == "max":
                pool = F.max_pool2d
            elif self._pool == "avg":
                pool = F.avg_pool2d

            if self._use_batch_norm:
                if self._pool is not None:
                    if self._use_dropout and i == 2:
                        x = F.relu(pool(self.conv2_drop(self._batch_norms[i](conv_i(x))),
                                        kernel_size=self._pool_size, stride=self._pool_stride))
                    else:
                        x = F.relu(pool(self._batch_norms[i](conv_i(x)),
                                        kernel_size=self._pool_size, stride=self._pool_stride))
                else:
                    x = F.relu(self._batch_norms[i](conv_i(x)))
            else:
                if self._pool is not None:
                    if self._use_dropout and i == 2:
                        x = F.relu(pool(self.conv2_drop(conv_i(x)),
                                        kernel_size=self._pool_size, stride=self._pool_stride))
                    else:
                        x = F.relu(pool(conv_i(x), kernel_size=self._pool_size, stride=self._pool_stride))
                else:
                    x = F.relu(conv_i(x))

        x = torch.flatten(x, 1)

        for i, hidden_i in enumerate(self._hidden_layers):
            x = self._hidden_activation(hidden_i(x))

        x = self._output_layer(x)
        return x
