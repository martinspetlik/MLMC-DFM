import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, trial, num_conv_layers, pool, num_filters, num_neurons, drop_conv2, drop_fc1, same_channels):

        super(Net, self).__init__()
        self._convs = []
        self._pool = pool

        # Define the convolutional layers
        # self.convs = nn.ModuleList([nn.Conv2d(1, num_filters[0], kernel_size=(3, 3))]) # List with the Conv layers
        # print("self.convs ", self.convs)
        #@TODO: add batch normalization
        #self.batch_norms = nn.ModuleList([nn.Conv2d(1, num_filters[0], kernel_size=(3, 3))])
        # out_size = in_size - kernel_size + 1                                            # Size of the output kernel
        # out_size = int(out_size / 2)                                                    # Size after pooling

        print("same channels ", same_channels)
        for i in range(1, num_conv_layers):
            if same_channels:
                pass
            else:
                print("in_channels: {}, out_channels: {}".format(num_filters[i-1], num_filters[i]))
                self._convs.append(nn.Conv2d(in_channels=num_filters[i-1], out_channels=num_filters[i], kernel_size=(3, 3)))
                out_size = out_size - kernel_size + 1                                       # Size of the output kernel
                out_size = int(out_size/2)                                                  # Size after pooling


        self.conv2_drop = nn.Dropout2d(p=drop_conv2)                                    # Dropout for conv2
        self.out_feature = num_filters[num_conv_layers-1] * out_size * out_size         # Size of flattened features
        self.fc1 = nn.Linear(self.out_feature, num_neurons)                             # Fully Connected layer 1
        self.fc2 = nn.Linear(num_neurons, 3)                                           # Fully Connected layer 2
        self.p1 = drop_fc1                                                              # Dropout ratio for FC1

        # Initialize weights with the He initialization
        for i in range(1, num_conv_layers):
            nn.init.kaiming_normal_(self.convs[i].weight, nonlinearity='relu')
            if self._convs[i].bias is not None:
                nn.init.constant_(self.convs[i].bias, 0)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')

    def forward(self, x):
        """Forward propagation.

        Parameters:
            - x (torch.Tensor): Input tensor of size [N,1,28,28]
        Returns:
            - (torch.Tensor): The output tensor after forward propagation [N,10]
        """
        for i, conv_i in enumerate(self._convs):  # For each convolutional layer
            pool = F.avg_pool2d
            if self._pool == "max":
                pool = F.max_pool2d

            if i == 2:  # Add dropout if layer 2
                x = F.relu(pool(self.conv2_drop(conv_i(x)), 2))  # Conv_i, dropout, max-pooling, RelU
            else:
                x = F.relu(pool(conv_i(x), 2))                   # Conv_i, max-pooling, RelU

        x = x.view(-1, self.out_feature)                     # Flatten tensor
        x = F.relu(self.fc1(x))                              # FC1, RelU
        x = F.dropout(x, p=self.p1, training=self.training)  # Apply dropout after FC1 only when training
        x = self.fc2(x)                                      # FC2

        return x
