import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ViTRegressor(nn.Module):
    def __init__(self, trial=None, n_conv_layers=3, max_channel=3, pool=None, kernel_size=3, stride=1, pool_size=2,
                 pool_stride=2, use_batch_norm=True, n_hidden_layers=1, max_hidden_neurons=520,
                 hidden_activation=F.relu, input_size=256, input_channel=3, conv_layer_obj=[], pool_indices=[],
                 use_cnn_dropout=False, use_fc_dropout=False, cnn_dropout_indices=[], fc_dropout_indices=[],
                 cnn_dropout_ratios=[], fc_dropout_ratios=[], n_output_neurons=3,
                 output_layer=True, output_bias=False, patch_size=16, vit_params={}):
        super().__init__()
        self._name = "ViTRegressor"

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
        self._batch_norms = nn.ModuleList()
        self._conv_layer_obj = conv_layer_obj
        self._pool_indices = pool_indices
        self._cnn_dropout_indices = cnn_dropout_indices
        self._fc_dropout_indices = fc_dropout_indices
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)

        if self._pool == "None":
            self._pool = None

        ####################################
        ## Vision transformer parameters  ##
        ####################################
        if len(vit_params) == 0:
            raise ValueError("vit params are empty")

        patch_size = vit_params["patch_size"] if "patch_size" in vit_params else 16
        vit_n_layers = vit_params["n_layers"] if "n_layers" in vit_params else 7
        n_heads = vit_params["n_heads"] if "n_heads" in vit_params else 12
        dim_feedforward = vit_params["dim_feedforward"] if "dim_feedforward" in vit_params else 2048
        in_features = vit_params["in_features"] if "in_features" in vit_params else max_channel

        vit_dropout = vit_params["vit_dropout"] if "vit_dropout" in vit_params else 0.1


        #assert input_size % patch_size == 0, 'Image size must be divisible by patch size!'
        #num_patches = (input_size // patch_size) ** 2
        #self.patch_size = patch_size

        #heads = kernel_size

        #d_model = max_channel

        ##########################
        ## Conovlutional layers ##
        ##########################
        if type(max_channel) == list and len(max_channel) == n_conv_layers:
            channels = [input_channel, *max_channel]
        else:
            channels = np.linspace(start=input_channel, stop=max_channel, num=n_conv_layers + 1, dtype=int)

        ## Pool indices
        if len(self._pool_indices) == 0:
            self._pool_indices = range(n_conv_layers)

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

            #print("initial input size ", input_size)

            input_size = int(((input_size - self._convs[-1].kernel_size[0]) / self._convs[-1].stride[0])) + 1

            if self._pool is not None and i in self._pool_indices:
                input_size = int(((input_size - pool_size) / pool_stride)) + 1


            #print("input size ", input_size)


        # print("patch size ", patch_size)
        # self.patch_embedding = nn.Conv2d(in_channels=input_channel, out_channels=max_channel,
        #                                  kernel_size=patch_size, stride=patch_size)

        ############################
        ###  Positional encoding ###
        ############################
        num_patches = input_size*input_size
        self.positional_encoding = nn.Parameter(torch.zeros(1, num_patches, channels[i + 1]))

        print("d_model: {} nheads: {}".format(channels[i + 1], n_heads))
        #print("max hidden  neurons ", max_hidden_neurons)
        max_neurons = np.max(max_hidden_neurons)

        ########################
        ## Transformer layers ##
        ########################
        transformer_layer = nn.TransformerEncoderLayer(d_model=channels[i + 1],
                                                       nhead=n_heads,
                                                       dim_feedforward=dim_feedforward,
                                                       dropout=vit_dropout)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=vit_n_layers)

        ###################################
        ## Fully-connected hidden layers ##
        ###################################
        if type(max_hidden_neurons) == list and len(max_hidden_neurons) == n_hidden_layers:
            hidden_neurons = max_hidden_neurons
        else:
            hidden_neurons = np.linspace(start=max_hidden_neurons, stop=self._n_output_neurons, num=n_hidden_layers,
                                         dtype=int)

        fc_input_size = channels[i + 1] * num_patches
        #print("fc input size ", fc_input_size)
        for i in range(n_hidden_layers):
            self._hidden_layers.append(nn.Linear(fc_input_size, hidden_neurons[i]))
            fc_input_size = hidden_neurons[i]

        ##################
        ## Output layer ##
        ##################
        if output_layer:
            self._output_layer = nn.Linear(fc_input_size, self._n_output_neurons, bias=output_bias)
        self.out_channels = self._n_output_neurons

        ############
        # Dropouts #
        ############
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
        #print("x.shape ", x.shape)

        #############################
        ## Feature extractor - CNN ##
        #############################
        for i, conv_i in enumerate(self._convs):  # For each convolutional layer
            if self._pool == "max":
                pool = F.max_pool2d
            elif self._pool == "avg":
                pool = F.avg_pool2d

            if self._use_batch_norm:
                if self._pool is not None and i in self._pool_indices:
                    if self._use_cnn_dropout and '{}'.format(i) in self._cnn_dropouts:
                        conv_dropout = self._cnn_dropouts['{}'.format(i)]

                        x = F.relu(pool(conv_dropout(self._batch_norms[i](conv_i(x))),
                                        kernel_size=self._pool_size, stride=self._pool_stride))
                    else:
                        x = F.relu(pool(self._batch_norms[i](conv_i(x)),
                                        kernel_size=self._pool_size, stride=self._pool_stride))
                else:
                    x = F.relu(self._batch_norms[i](conv_i(x)))
            else:
                if self._pool is not None and i in self._pool_indices:
                    if self._use_cnn_dropout and '{}'.format(i) in self._cnn_dropouts:
                        conv_dropout = self._cnn_dropouts['{}'.format(i)]
                        x = F.relu(pool(conv_dropout(conv_i(x)),
                                        kernel_size=self._pool_size, stride=self._pool_stride))
                    else:
                        x = F.relu(pool(conv_i(x), kernel_size=self._pool_size, stride=self._pool_stride))
                else:
                    x = F.relu(conv_i(x))

        #print("x cnn embeding shape ", x.shape)

        x = x.flatten(2).permute(0, 2, 1)
        #print("flatten x ", x.shape)
        #print("shape ", self.positional_encoding.shape)
        x = x + self.positional_encoding
        #print("transformer intpu shape ", x.shape)

        x = self.transformer(x)
        #print("transfromer output shape ", x.shape)
        x = x.flatten(1)
        #print("x. flatten shape ", x.shape)
        #x = self.fc(x)

        for i, hidden_i in enumerate(self._hidden_layers):
            x = self._hidden_activation(hidden_i(x))
            #print("hidden layer output shape ", x.shape)
            if self._use_fc_dropout and '{}'.format(i) in self._fc_dropouts:
                x = self._fc_dropouts['{}'.format(i)](x)
        x = self._output_layer(x)

        return x
