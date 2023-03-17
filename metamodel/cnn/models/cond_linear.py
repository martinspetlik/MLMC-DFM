import torch.nn as nn
import torch.nn.functional as F


class CondLinear(nn.Module):
    def __init__(self, in_neurons, out_neurons=3, hidden_neurons=[], hidden_activation=F.relu):
        super().__init__()
        self._hidden_neurons = hidden_neurons
        self._in_neurons = in_neurons
        self._out_neurons = out_neurons

        self._hidden_layers = nn.ModuleList()
        self._output_layer = None
        self._hidden_activation = hidden_activation
        self._create_fcl_layers()
        self._weights = None

    def _create_fcl_layers(self):
        for h_neurons in self._hidden_neurons:
            self._hidden_layers.append(nn.Linear(self._in_neurons, h_neurons))
        self._output_layer = nn.Linear(h_neurons, self._out_neurons, bias=False)

    @property
    def weight(self):
        if self._weights is None:
            self.get_weights()
        return self._weights

    def get_weight(self):
        self._weights = []
        for hidden_layer in self._hidden_layers:
            self._weights.append(hidden_layer.weight)
        self._weights.append(self._output_layer.weight)


    def forward(self, x):
        for hidden_layer in self._hidden_layers:
            x = self._hidden_activation(hidden_layer(x))
        x = self._output_layer(x)
        return x
