import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# The main CNN model class
class Net(nn.Module):
    def __init__(
        self,
        trial=None,
        n_conv_layers=3,
        max_channel=6,
        activation_before_pool=None,
        kernel_size=3,
        stride=1,
        pool_size=2,
        pool_stride=2,
        use_batch_norm=True,
        n_hidden_layers=1,
        max_hidden_neurons=520,
        cnn_activation=F.relu,
        hidden_activation=F.relu,
        input_size=256,
        input_channel=6,
        conv_layer_obj=[],
        pool_indices={},
        use_cnn_dropout=False,
        use_fc_dropout=False,
        cnn_dropout_indices=[],
        fc_dropout_indices=[],
        cnn_dropout_ratios=[],
        fc_dropout_ratios=[],
        n_output_neurons=6,
        output_layer=True,
        output_bias=False,
        global_pool=None,
        bias_reduction_layer_indices=[],
        padding=0,
        pool=None,
    ):
        super(Net, self).__init__()

        # --- Store parameters ---
        self._name = "cnn_net"
        self._use_cnn_dropout = use_cnn_dropout
        self._use_fc_dropout = use_fc_dropout
        self._activation_before_pool = activation_before_pool
        self._pool_size = pool_size
        self._pool_stride = pool_stride
        self._n_output_neurons = n_output_neurons
        self._use_batch_norm = use_batch_norm
        self._cnn_activation = cnn_activation
        self._hidden_activation = hidden_activation
        self._conv_layer_obj = conv_layer_obj
        self._pool_indices = pool_indices
        self._cnn_dropout_indices = cnn_dropout_indices
        self._fc_dropout_indices = fc_dropout_indices
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self._global_pool = global_pool
        self._bias_reduction_layer_indices = bias_reduction_layer_indices

        # --- Define layer containers ---
        self._convs = nn.ModuleList()
        self._batch_norms = nn.ModuleList()
        self._hidden_layers = nn.ModuleList()
        self._bias_reduction_layers = nn.ModuleList()
        self._cnn_dropouts = {}
        self._fc_dropouts = {}

        # --- Define channel sizes for convolutional layers ---
        if isinstance(max_channel, list) and len(max_channel) == n_conv_layers:
            channels = [input_channel] + max_channel
        else:
            # Evenly space channels from input_channel to max_channel
            channels = np.linspace(start=input_channel, stop=max_channel, num=n_conv_layers + 1, dtype=int)

        # --- Set default pooling behavior if not provided ---
        if len(self._pool_indices) == 0:
            self._pool_indices = {i: pool for i in range(n_conv_layers)}

        # --- Create convolutional layers (and optional batch norm + pool) ---
        for i in range(n_conv_layers):
            if conv_layer_obj:
                # Use externally provided layers if available
                if i < len(conv_layer_obj):
                    self._convs.append(conv_layer_obj[i])
                else:
                    raise Exception("Not enough conv layer objects")
            else:
                self._convs.append(nn.Conv3d(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                ))
            # Optional batch normalization
            if self._use_batch_norm:
                self._batch_norms.append(nn.BatchNorm3d(channels[i + 1]))

            # Update spatial dimensions after conv and pool
            input_size = int((input_size - self._convs[-1].kernel_size[0]) / self._convs[-1].stride[0]) + 1
            if i in self._pool_indices:
                input_size = int((input_size - pool_size) / pool_stride) + 1

        # --- Apply global pooling if enabled ---
        if self._global_pool is not None:
            input_size = 1  # Global pooling collapses spatial dimensions

        # --- Prepare fully connected layers ---
        if isinstance(max_hidden_neurons, list) and len(max_hidden_neurons) == n_hidden_layers:
            hidden_neurons = max_hidden_neurons
        else:
            # Linearly interpolate hidden layer sizes
            hidden_neurons = np.linspace(start=max_hidden_neurons, stop=self._n_output_neurons, num=n_hidden_layers, dtype=int)

        # Flatten input dimension after CNN
        input_size = self._convs[-1].out_channels * input_size ** 3

        for i in range(n_hidden_layers):
            if i - 1 in self._bias_reduction_layer_indices:
                self._hidden_layers.append(nn.Linear(self._n_output_neurons, hidden_neurons[i]))
            else:
                self._hidden_layers.append(nn.Linear(input_size, hidden_neurons[i]))

            if i in self._bias_reduction_layer_indices:
                self._bias_reduction_layers.append(nn.Linear(hidden_neurons[i], self._n_output_neurons))

            input_size = hidden_neurons[i]

        # --- Final output layer ---
        if output_layer:
            self._output_layer = nn.Linear(input_size, self._n_output_neurons, bias=output_bias)
        self.out_channels = self._n_output_neurons

        # --- CNN Dropout setup ---
        if self._use_cnn_dropout:
            if not cnn_dropout_indices:
                cnn_dropout_indices = range(len(self._convs))
            for i in cnn_dropout_indices:
                p = cnn_dropout_ratios[i] if i < len(cnn_dropout_ratios) else 0.5
                self._cnn_dropouts[str(i)] = nn.Dropout3d(p=p)
        self._cnn_dropouts = nn.ModuleDict(self._cnn_dropouts)

        # --- FC Dropout setup ---
        if self._use_fc_dropout:
            if not fc_dropout_indices:
                fc_dropout_indices = range(len(self._hidden_layers))
            for i in fc_dropout_indices:
                p = fc_dropout_ratios[i] if i < len(fc_dropout_ratios) else 0.5
                self._fc_dropouts[str(i)] = nn.Dropout(p=p)
        self._fc_dropouts = nn.ModuleDict(self._fc_dropouts)

    # --- Forward pass ---
    def forward(self, x):
        def count_conv_params_per_layer(module):
            return {
                name: sum(p.numel() for p in layer.parameters() if p.requires_grad)
                for name, layer in module.named_modules()
                if isinstance(layer, (nn.Conv2d, nn.Conv3d, nn.Linear))
            }

        # --- Convolutional layers ---
        for i, conv_i in enumerate(self._convs):
            if self._use_batch_norm:
                x = self._batch_norms[i](conv_i(x))
            else:
                x = conv_i(x)

            # Apply activation
            x = self._cnn_activation(x)

            # Optional dropout
            if self._use_cnn_dropout and str(i) in self._cnn_dropouts:
                x = self._cnn_dropouts[str(i)](x)

            # Optional pooling
            if i in self._pool_indices:
                pool_type = self._pool_indices[i]
                if pool_type == "max":
                    x = F.max_pool3d(x, kernel_size=self._pool_size, stride=self._pool_stride)
                elif pool_type == "avg":
                    x = F.avg_pool3d(x, kernel_size=self._pool_size, stride=self._pool_stride)

        # --- Global pooling ---
        if self._global_pool == "max":
            x = F.max_pool3d(x, kernel_size=x.shape[-1])
        elif self._global_pool == "avg":
            x = F.avg_pool3d(x, kernel_size=x.shape[-1])

        # --- Flatten before FC layers ---
        x = torch.flatten(x, 1)

        # --- Fully connected hidden layers ---
        for i, hidden_i in enumerate(self._hidden_layers):
            x = self._hidden_activation(hidden_i(x))
            if self._use_fc_dropout and str(i) in self._fc_dropouts:
                x = self._fc_dropouts[str(i)](x)

        # --- Output layer ---
        if hasattr(self, '_output_layer'):
            x = self._output_layer(x)

        return x
