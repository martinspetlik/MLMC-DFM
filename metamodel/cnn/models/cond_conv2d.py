import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class CondConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1,
                 cnn_activation=F.relu, hidden_activation=F.relu):
        super(CondConv2d, self).__init__()

        self.kernel_size = (kernel_size, kernel_size)
        self.n_kernel_weights = kernel_size * kernel_size
        self.out_channels = out_channels
        self.padding = (padding, padding)
        self.stride = (stride, stride)
        self.in_channels = in_channels
        self.cnn_weights = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.n_kernel_weights))

        self._hidden_layers = []
        self._output_layer = None
        self._cnn_activation = cnn_activation
        self._hidden_activation = hidden_activation
        self._create_fcl_layers()

    def _create_fcl_layers(self):
        self._hidden_layers.append(nn.Linear(self.out_channels, 24))
        self._output_layer = nn.Linear(24, 3)

    def forward(self, x):
        width, height = self._get_new_shape(x)
        windows = self._get_windows(x)

        print("width: {}, height: {}, windows.shape: {}".format(width, height, windows.shape))

        #result = torch.zeros([x.shape[0], self.out_channels, width * height], dtype=torch.float32, device=device)
        result = torch.zeros([width * height, x.shape[0], 3], dtype=torch.float32, device=device)
        print("result.shape ", result.shape)

        #out_channel_res = []
        for channel in range(x.shape[1]):  # input channels
            for w_idx, window in enumerate(windows[channel]):  # conv windows
                window_out_channel_res = torch.zeros([self.out_channels, x.shape[0]], dtype=torch.float32, device=device)
                for i_conv in range(self.out_channels):  # out channels
                    print("window ", window)
                    print("window.shape ", window.shape)
                    print("self.cnn_weights[i_conv][channel] ", self.cnn_weights[i_conv][channel])

                    xx = torch.matmul(window, self.cnn_weights[i_conv][channel])
                    xx = self._cnn_activation(xx)
                    print("xx ", xx)

                    print("xx.shape ", xx.shape)
                    window_out_channel_res[i_conv] = xx
                    print("window_out_channel_res.shape ", window_out_channel_res.shape)
                    #exit()
                    #@TODO: add cnn activation

                window_out_channel_res = window_out_channel_res.T
                print("window_out_channel_res.shape ", window_out_channel_res.shape)
                # exit()

                for hidden_layer in self._hidden_layers:
                    xx = self._hidden_activation(hidden_layer(window_out_channel_res))
                xx = self._output_layer(xx)

                print("xx ", xx)
                #exit()

                print("x.shape ", xx.shape)

                #print("x.shape ", x.size(dim=0))
                #print("xx_view.shape ", xx.shape)
                #exit()
                #out_channel_res[i_conv * xx.shape[0]: (i_conv + 1) * xx.shape[0]] += xx
                print("result[w_idx] ", result[w_idx].shape)
                print("xx.shape ", xx.shape)
                result[w_idx] += xx


        print("result.shape ", result.shape)
        result = torch.reshape(result, (width, height, x.shape[0], 3))
        result = result.permute(2, 3, 0, 1)


        print("final result.shape ", result.shape)

        print("result ", result)
        return result


    def _get_new_shape(self, x):
        return int(((x.shape[2] - self.kernel_size[0]) / self.stride[0])) + 1,\
               int(((x.shape[3] - self.kernel_size[1]) / self.stride[1])) + 1

    def _get_windows(self, x):
        """
        Get convolutional window
        :param x: input data
        :return: windows, shape: (number of input channels, number of windows, batch size, pixels per window)
        """
        #print("x.shape ", x.shape)
        windows = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        #print("windows.shape ", windows.shape)
        #print(" windows.transpose(1, 2).shape ",  windows.transpose(1, 2).contiguous().shape)
        windows = windows.transpose(1, 2).contiguous().view(-1, x.shape[1], x.shape[0], self.n_kernel_weights)
        #print("win shape ", windows.shape)
        windows = windows.transpose(0, 1)
        #print("final window shape ", windows.shape)
        return windows


device = 'cpu'
conv = CondConv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1)
x = torch.randn(25, 3, 3, 3) # batch size x n channels x pixel size
out = conv(x)
print("out.shape",  out.shape)
out.mean().backward()
print(conv.cnn_weights.grad)
