from __future__ import print_function, division
import os
import re
import torch
import pandas as pd
#from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class DFMDataset(Dataset):
    """DFM models dataset"""

    def __init__(self, data_dir, bulk_file_name="bulk_256.npz", fracture_file_name="fractures_256.npz",
                 output_file_name="output_tensor.npy", transform=None):

        self._data_dir = data_dir
        self._bulk_file_name = bulk_file_name
        self._fracture_file_name = fracture_file_name
        self._output_file_name = output_file_name
        self.transform = transform

        self._bulk_file_paths = []
        self._fracture_file_paths = []
        self._output_file_paths = []

        self._set_paths_to_samples()

    def _set_paths_to_samples(self):
        if self._data_dir is None:
            raise AttributeError

        for s_dir in os.listdir(self._data_dir):
            try:
                l = re.findall(r'sample_[0-9]*', s_dir)[0]
            except IndexError:
                continue

            if os.path.isdir(os.path.join(self._data_dir, s_dir)):
                sample_dir = os.path.join(self._data_dir, s_dir)

                bulk_file = os.path.join(sample_dir, self._bulk_file_name)
                fractures_file = os.path.join(sample_dir, self._fracture_file_name)
                output_file = os.path.join(sample_dir, self._output_file_name)

                if os.path.exists(bulk_file):
                    self._bulk_file_paths.append(bulk_file)
                else:
                    raise FileNotFoundError("File {} containing bulk values not found".format(bulk_file))

                if os.path.exists(fractures_file):
                    self._fracture_file_paths.append(fractures_file)
                else:
                    raise FileNotFoundError("File {} containing bulk values not found".format(fractures_file))

                if os.path.exists(output_file):
                    self._output_file_paths.append(output_file)
                else:
                    raise FileNotFoundError("File {} containing bulk values not found".format(output_file))

    def __len__(self):
        return len(self._bulk_file_paths)

    def __getitem__(self, idx):
        bulk_path, fractures_path = self._bulk_file_paths[idx], self._fracture_file_paths[idx]
        print("bulk_path ", bulk_path)
        output_path = self._output_file_paths[idx]

        bulk_features = np.load(bulk_path)["data"]
        fractures_features = np.load(fractures_path)["data"]
        output_features = np.load(output_path, allow_pickle=True)

        # @TODO: logarithm data to get normal distribution
        bulk_features_shape = bulk_features.shape

        flatten_bulk_features = bulk_features.reshape(-1)
        flatten_fracture_features = fractures_features.reshape(-1)

        not_nan_indices = np.argwhere(~np.isnan(flatten_fracture_features))
        print("not nan indices ", not_nan_indices)
        flatten_bulk_features[not_nan_indices] = flatten_fracture_features[not_nan_indices]
        final_features = flatten_bulk_features.reshape(bulk_features_shape)

        final_features = np.transpose(final_features, (1, 2, 0))

        return final_features, output_features



if __name__ == "__main__":
    dfm_dataset = DFMDataset(data_dir='/home/martin/Documents/MLMC-DFM/test/01_cond_field/homogenization_samples_fractures')

    for i in range(len(dfm_dataset)):
        sample = dfm_dataset[i]

        # Plot features
        # features = np.transpose(sample[0], (2,0,1))
        # for feature in features:
        #     plt.gray()
        #     plt.imshow(feature)
        #     plt.show()
