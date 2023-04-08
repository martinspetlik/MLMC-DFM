import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ViTRegressor(nn.Module):
    def __init__(self, #image_size, patch_size, num_classes, dim, depth, heads, mlp_dim):
                 trial=None, n_conv_layers=3, max_channel=3, pool=None, kernel_size=3, stride=1, pool_size=2,
                 pool_stride=2, use_batch_norm=True, n_hidden_layers=1, max_hidden_neurons=520,
                 hidden_activation=F.relu, input_size=256, input_channel=3, conv_layer_obj=[], pool_indices=[],
                 use_cnn_dropout=False, use_fc_dropout=False, cnn_dropout_indices=[], fc_dropout_indices=[],
                 cnn_dropout_ratios=[], fc_dropout_ratios=[], n_output_neurons=3,
                 output_layer=True, output_bias=False, patch_size=16, heads=None):
        super().__init__()
        self._name = "ViTRegressor"
        assert input_size % patch_size == 0, 'Image size must be divisible by patch size!'
        num_patches = (input_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2
        self.patch_size = patch_size
        self.patch_embedding = nn.Conv2d(in_channels=3, out_channels=max_channel, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, max_channel))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, max_channel))
        if heads is None:
            heads = int(np.sqrt(max_channel))

        print("d_model: {} nheads: {}".format(max_channel, heads))

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=max_channel, nhead=heads, dim_feedforward=max_hidden_neurons),
            num_layers=n_hidden_layers
        )
        self.regressor = nn.Linear(max_channel, n_output_neurons)

    def forward(self, x):
        # split image into patches and flatten
        x = self.patch_embedding(x)
        b, c, h, w = x.shape
        x = x.view(b, c, h*w).transpose(1, 2)
        n = x.size(1)

        # add cls token and positional embeddings
        cls_token = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        # transformer encoder
        x = self.transformer_encoder(x)

        # mean pooling
        x = x.mean(dim=1)

        # regression layer
        x = self.regressor(x)
        return x