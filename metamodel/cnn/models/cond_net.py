import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from metamodel.cnn.models.cond_linear import CondLinear


class CondNet(nn.Module):

    def __init__(self, trial=None, n_conv_layers=3, max_channel=3, pool=None, kernel_size=3, stride=1, pool_size=2,
                 pool_stride=2, use_batch_norm=True, n_hidden_layers=1, max_hidden_neurons=520,
                 hidden_activation=F.relu, input_size=256, min_channel=3, use_dropout=False, conv_layer_obj=[],
                 output_layer=True):
        super(CondNet, self).__init__()
        self._name = "cond_net"
        self._use_dropout = use_dropout
        self._pool = pool
        self._pool_size = pool_size
        self._pool_stride = pool_stride
        self._convs = nn.ModuleList()
        self._fcns = nn.ModuleList()
        self._hidden_layers = nn.ModuleList()
        self._use_batch_norm = use_batch_norm
        self._hidden_activation = hidden_activation
        self._batch_norms = nn.ModuleList()
        self._conv_layer_obj = conv_layer_obj
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)


        self._use_single_nn = True

        n_layers = 0

        while True:
            input_size = int(((input_size - kernel_size) / stride)) + 1
            n_layers += 1

            if input_size == 1:
                break
            elif input_size < 1:
                raise ValueError(
                    "Stride and kernel size result in inappropriate number of final pixels: {}".format(input_size))

        self._n_layers = n_layers

        if self._use_single_nn:
            self._convs = nn.Conv2d(in_channels=min_channel,
                                    out_channels=max_channel,
                                    kernel_size=kernel_size,
                                    #padding="same",
                                    stride=stride)

            if self._use_batch_norm:
                self._batch_norms = nn.BatchNorm2d(max_channel)

            self._fcns = CondLinear(in_neurons=max_channel,
                                    out_neurons=min_channel,
                                    hidden_neurons=[max_hidden_neurons])

        else:
            for i in range(n_layers):
                self._convs.append(nn.Conv2d(in_channels=min_channel,
                                             out_channels=max_channel,
                                             kernel_size=kernel_size,
                                             stride=stride))
                if self._use_batch_norm:
                    self._batch_norms.append(nn.BatchNorm2d(max_channel))

                self._fcns.append(CondLinear(in_neurons=max_channel, out_neurons=min_channel,
                                             hidden_neurons=[max_hidden_neurons]))

    def forward(self, x):
        for i in range(self._n_layers):
            if self._use_single_nn:
                if self._use_batch_norm:
                    x = F.relu(self._batch_norms(self._convs(x)))
                else:
                    x = self._convs(x)
                    x = F.relu(x)

                n_pixels = x.shape[-1]
                batch_size = x.shape[0]
                x = x.permute(0, 2, 3, 1)  # batch size X pixels_x X pixels_y X out channels
                x = torch.flatten(x, start_dim=0, end_dim=2)
                x = self._fcns(x)

                x = torch.reshape(x, (batch_size, x.shape[-1], n_pixels, n_pixels))
            else:
                if self._use_batch_norm:
                    x = F.relu(self._batch_norms[i](self._convs[i](x)))
                else:
                    x = F.relu(self._convs[i](x))
                    # print("x.shape ", x.shape)

                n_pixels = x.shape[-1]
                batch_size = x.shape[0]
                x = x.permute(0, 2, 3, 1)  # batch size X pixels_x X pixels_y X out channels
                x = torch.flatten(x, start_dim=0, end_dim=2)
                #print("x.shape ", x.shape)

                x = self._fcns[i](x)

                x = torch.reshape(x, (batch_size, n_pixels, n_pixels, x.shape[-1]))




        # x = F.relu(self._conv_layer(x))
        # #print("x.shape ", x.shape)
        # n_pixels = x.shape[-1]
        # batch_size = x.shape[0]
        # x = x.permute(0,2,3,1)   # batch size X pixels_x X pixels_y X out channels
        # #print("x. shape ", x.shape)
        # x = torch.flatten(x, start_dim=0, end_dim=2)
        # print("x.shape ", x.shape)
        #
        # x = self._fcn_layer(x)
        #
        # x = torch.reshape(x, (batch_size, n_pixels, n_pixels, x.shape[-1]))
        #
        #
        return x
