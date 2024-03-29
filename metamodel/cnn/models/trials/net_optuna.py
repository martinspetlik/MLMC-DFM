import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, trial=None, n_conv_layers=3, max_channel=3, pool=None, kernel_size=3, stride=1, pool_size=2,
                 pool_stride=2, use_batch_norm=True, n_hidden_layers=1, max_hidden_neurons=520, cnn_activation=F.relu,
                 hidden_activation=F.relu, input_size=256, input_channel=3, conv_layer_obj=[], pool_indices={},
                 use_cnn_dropout=False, use_fc_dropout=False, cnn_dropout_indices=[], fc_dropout_indices=[],
                 cnn_dropout_ratios=[], fc_dropout_ratios=[], n_output_neurons=3,
                 output_layer=True, output_bias=False, global_pool=None):
        super(Net, self).__init__()
        self._name = "cnn_net"
        self._use_cnn_dropout = use_cnn_dropout
        self._use_fc_dropout = use_fc_dropout
        self._pool = pool
        self._pool_size = pool_size
        self._pool_stride = pool_stride
        self._n_output_neurons = n_output_neurons
        self._convs = nn.ModuleList()
        self._hidden_layers = nn.ModuleList()
        self._cnn_dropouts = {}
        self._fc_dropouts = {}
        self._use_batch_norm = use_batch_norm
        self._hidden_activation = hidden_activation
        self._cnn_activation = cnn_activation
        self._batch_norms = nn.ModuleList()
        self._conv_layer_obj = conv_layer_obj
        self._pool_indices = pool_indices
        self._cnn_dropout_indices = cnn_dropout_indices
        self._fc_dropout_indices = fc_dropout_indices
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self._global_pool = global_pool

        if self._pool == "None":
            self._pool = None

        ##########################
        ## Conovlutional layers ##
        ##########################
        if type(max_channel) == list and len(max_channel) == n_conv_layers:
            channels = [input_channel, *max_channel]
        else:
            channels = np.linspace(start=input_channel, stop=max_channel, num=n_conv_layers + 1, dtype=int)

        ## Pool indices
        if len(self._pool_indices) == 0:
            for i in range(n_conv_layers):
                self._pool_indices[i] = self._pool

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

            if self._pool is not None and i in list(self._pool_indices.keys()):
                input_size = int(((input_size - pool_size) / pool_stride)) + 1

        if self._global_pool is not None:
            input_size = 1

        ###################################
        ## Fully-connected hidden layers ##
        ###################################
        if type(max_hidden_neurons) == list and len(max_hidden_neurons) == n_hidden_layers:
            hidden_neurons = max_hidden_neurons
        else:
            hidden_neurons = np.linspace(start=max_hidden_neurons, stop=self._n_output_neurons, num=n_hidden_layers, dtype=int)

        input_size = self._convs[-1].out_channels * input_size * input_size
        for i in range(n_hidden_layers):
            self._hidden_layers.append(nn.Linear(input_size, hidden_neurons[i]))
            input_size = hidden_neurons[i]

        ##################
        ## Output layer ##
        ##################
        if output_layer:
            self._output_layer = nn.Linear(input_size, self._n_output_neurons, bias=output_bias)
        self.out_channels = self._n_output_neurons

        ############
        # Dropouts #
        ############
        if len(cnn_dropout_indices) == 0:
            cnn_dropout_indices = range(len(self._convs))
        if use_cnn_dropout:
            for i in cnn_dropout_indices:
                dropout_ratio = 0.5
                if i < len(cnn_dropout_ratios):
                    dropout_ratio = cnn_dropout_ratios[i]
                self._cnn_dropouts['{}'.format(i)] = nn.Dropout2d(p=dropout_ratio)
        self._cnn_dropouts = nn.ModuleDict(self._cnn_dropouts)

        if use_fc_dropout:
            if len(fc_dropout_indices) == 0:
                fc_dropout_indices = range(len(self._hidden_layers))
            for i in fc_dropout_indices:
                dropout_ratio = 0.5
                if i < len(fc_dropout_ratios):
                    dropout_ratio = fc_dropout_ratios[i]
                self._fc_dropouts['{}'.format(i)] = nn.Dropout(p=dropout_ratio)
        self._fc_dropouts = nn.ModuleDict(self._fc_dropouts)


    def forward(self, x):
        verbose= False

        for i, conv_i in enumerate(self._convs):  # For each convolutional layer
            if verbose:
                print("i: {}, x shape: {}".format(i, x.shape))
                #print("x.shape " ,x.shape)
                print("conv_i.weights ", np.prod(conv_i.weight.shape))
                print("conv_i.bias ", conv_i.bias.shape)


            if self._use_batch_norm:
                if self._pool is not None and i in list(self._pool_indices.keys()):
                    #pool = self._pool_indices[i]
                    if self._pool_indices[i] == "max":
                        pool = F.max_pool2d
                    elif self._pool_indices[i] == "avg":
                        pool = F.avg_pool2d
                    if self._use_cnn_dropout and '{}'.format(i) in self._cnn_dropouts:
                        conv_dropout = self._cnn_dropouts['{}'.format(i)]

                        x = self._cnn_activation(pool(conv_dropout(self._batch_norms[i](conv_i(x))),
                                        kernel_size=self._pool_size, stride=self._pool_stride))
                    else:
                        x = self._cnn_activation(pool(self._batch_norms[i](conv_i(x)),
                                        kernel_size=self._pool_size, stride=self._pool_stride))
                else:
                    x = self._cnn_activation(self._batch_norms[i](conv_i(x)))
            else:
                if self._pool is not None and i in list(self._pool_indices.keys()):
                    if self._pool_indices[i] == "max":
                        pool = F.max_pool2d
                    elif self._pool_indices[i] == "avg":
                        pool = F.avg_pool2d
                    if self._use_cnn_dropout and '{}'.format(i) in self._cnn_dropouts:
                        conv_dropout = self._cnn_dropouts['{}'.format(i)]
                        x = self._cnn_activation(pool(conv_dropout(conv_i(x)),
                                        kernel_size=self._pool_size, stride=self._pool_stride))
                    else:
                        x = self._cnn_activation(pool(conv_i(x), kernel_size=self._pool_size, stride=self._pool_stride))
                else:
                    x = self._cnn_activation(conv_i(x))

        ##
        # Global pooling
        ##
        if self._global_pool is not None:
            if self._global_pool == "max":
                global_pool = F.max_pool2d
            elif self._global_pool == "avg":
                global_pool = F.avg_pool2d
            x = global_pool(x, kernel_size=x.shape[-1])

        if verbose:
            print("ccn output shape ", x.shape)
        x = torch.flatten(x, 1)

        if verbose:
            print("x flatten shape ", x.shape)

        for i, hidden_i in enumerate(self._hidden_layers):
            x = self._hidden_activation(hidden_i(x))
            if self._use_fc_dropout and '{}'.format(i) in self._fc_dropouts:
                #print("fc droppout ", self._fc_dropouts['{}'.format(i)])
                x = self._fc_dropouts['{}'.format(i)](x)

            if verbose:
                print("hidden shape ", x.shape)
                print("hidden_i.weights ", np.prod(hidden_i.weight.shape))
                print("hidden_i.bias ", hidden_i.bias.shape)

        x = self._output_layer(x)
        return x
