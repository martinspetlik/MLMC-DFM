import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from metamodel.cnn.models.cond_linear import CondLinear


class CondNet(nn.Module):

    def __init__(self, trial=None, n_conv_layers=3, max_channel=3, pool=None, kernel_size=3, stride=1, pool_size=2,
                 pool_stride=2, use_batch_norm=True, n_hidden_layers=1, max_hidden_neurons=520,
                 hidden_activation=F.relu, input_size=256, min_channel=3, use_dropout=False, convs=None, fcls=None, batch_norms=None,
                 output_layer=True, layer_models=[]):
        super().__init__()
        self._name = "cond_net"
        self._use_dropout = use_dropout
        self._pool = pool
        self._pool_size = pool_size
        self._pool_stride = pool_stride
        self._convs=nn.ModuleList()
        self._fcls=nn.ModuleList()

        self._batch_norms = nn.ModuleList()
        self._batch_norms_after = nn.ModuleList()

        if convs is not None:
            self._convs = convs

        if fcls is not None:
            self._fcls = fcls

        if batch_norms is not None:
            self._batch_norms = batch_norms

        self._n_pretrained_layers = len(self._convs)

        self._hidden_layers = nn.ModuleList()
        self._use_batch_norm = use_batch_norm
        self._hidden_activation = hidden_activation

        self._layer_models = layer_models
        #self._conv_layer_obj = conv_layer_obj
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)

        #print("initial convs: {}, fcls: {}".format(convs, fcls))

        self._use_single_nn = True

        for cnv in self._convs:
            cnv.stride = stride
            #print("cnv final stride", cnv.stride)

        n_layers = 0
        #print("input size ", input_size)
        while True:
            input_size = int(((input_size - kernel_size) / stride)) + 1
            n_layers += 1

            #print("input_size ", input_size)

            if input_size == 1:
                break
            elif input_size < 1:
                raise ValueError(
                    "Stride and kernel size result in inappropriate number of final pixels: {}".format(input_size))

        self._n_layers = n_layers

        #self._n_layers -= len(self._layer_models)

        #print("self._n_layers ", self._n_layers)

        if self._use_single_nn:
            conv = nn.Conv2d(in_channels=min_channel,
                                    out_channels=max_channel,
                                    kernel_size=kernel_size,
                                    stride=stride)

            if self._use_batch_norm:
                batch_norms = nn.BatchNorm2d(max_channel)

            # if fcls is not None:
            #     #print("fcls ", fcls)
            #     fcls = fcls
            # else:
            fcls = CondLinear(in_neurons=max_channel,
                             out_neurons=min_channel,
                            hidden_neurons=[max_hidden_neurons])

        n_layers = self._n_layers - self._n_pretrained_layers

        for i in range(n_layers):
            if self._use_single_nn:
                self._convs.append(conv)
                if self._use_batch_norm:
                    self._batch_norms.append(batch_norms)
                self._fcls.append(fcls)

            else:
                # if len(self._convs) > 0:
                #     print("self._convs.shape ", self._convs[0].weight)
                # print("min channels: {}, max channel: {}".format(min_channel, max_channel))
                self._convs.append(nn.Conv2d(in_channels=min_channel,
                                             out_channels=max_channel,
                                             kernel_size=kernel_size,
                                             stride=stride))
                if self._use_batch_norm:
                    self._batch_norms.append(nn.BatchNorm2d(max_channel))
                    self._batch_norms_after.append(nn.BatchNorm2d(min_channel))

                self._fcls.append(CondLinear(in_neurons=max_channel, out_neurons=min_channel,
                                             hidden_neurons=[max_hidden_neurons]))

    def _plot_data(self, data, idx):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
        axes[0].matshow(data[0])
        axes[1].matshow(data[1])
        axes[2].matshow(data[2])
        # fig.colorbar(caxes)
        plt.savefig("layer_output_{}.pdf".format(idx))
        plt.show()
        # plt.imshow(data)
        # plt.show()

    def _process_data(self, samples):
        samples = samples.numpy()
        print("samples.shape ", samples.shape)

        batch_mean = np.mean(samples, axis=0)

        print("batch mean ", batch_mean)
        exit()


    def forward(self, x):
        # for trained_layer in self._layer_models:
        #     x = trained_layer(x)
        #print("x.shape ",x.shape)
        #print("self._convs ", self._convs)

        for i in range(self._n_layers):
            #print("layer: {}".format(i))
            #print("x.shape ", x.shape)

            # if i > 0:
            #     print("conv weights ", self._convs[i].weight)

            if self._use_batch_norm:
                x = F.relu(self._batch_norms[i](self._convs[i](x)))
            else:
                x = self._convs[i](x)
                x = F.relu(x)

            n_pixels = x.shape[-1]
            batch_size = x.shape[0]

            x = x.permute(0, 2, 3, 1)  # batch size X pixels_x X pixels_y X out channels
            x = torch.flatten(x, start_dim=0, end_dim=2)

            x = self._fcls[i](x)


            x = torch.reshape(x, (batch_size, x.shape[-1], n_pixels, n_pixels))

            # if i < len(self._batch_norms_after):
            #     x = self._batch_norms_after[i](x)


            #self._process_data(x)

            # for idx, channel in enumerate(x):
            #     if idx > 25:
            #         break
            #     self._plot_data(channel, idx)
            # exit()

            #print("layer {} i, n pixels {}".format(i, x.shape))
            # if self._use_single_nn:
            #     if self._use_batch_norm:
            #         x = F.relu(self._batch_norms(self._convs(x)))
            #     else:
            #         x = self._convs(x)
            #         x = F.relu(x)
            #
            #     n_pixels = x.shape[-1]
            #
            #     batch_size = x.shape[0]
            #     x = x.permute(0, 2, 3, 1)  # batch size X pixels_x X pixels_y X out channels
            #     x = torch.flatten(x, start_dim=0, end_dim=2)
            #     #print("fcls input shape ", x.shape)
            #     x = self._fcls(x)
            #
            #     x = torch.reshape(x, (batch_size, x.shape[-1], n_pixels, n_pixels))
            #
            #     # for idx, channel in enumerate(x):
            #     #     if idx > 25:
            #     #         break
            #     #     self._plot_data(channel, idx)
            #     # exit()
            # else:

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

        #print("conv 0 weight ", self._convs[0].weight[0, 0])

        return x