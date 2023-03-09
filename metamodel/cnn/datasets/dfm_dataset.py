import os
import re
import copy
import torch
import pandas as pd
#from skimage import io, transform
import numpy as np
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class DFMDataset(Dataset):
    """DFM models dataset"""

    def __init__(self, data_dir, bulk_file_name="bulk_256.npz", fracture_file_name="fractures_256.npz",
                 output_file_name="output_tensor.npy", input_transform=None, output_transform=None, two_dim=True):

        self._data_dir = data_dir
        self._bulk_file_name = bulk_file_name
        self._fracture_file_name = fracture_file_name
        self._output_file_name = output_file_name
        self.input_transform = input_transform
        self.output_transform = output_transform
        self._two_dim = two_dim

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

                if os.path.exists(bulk_file) and os.path.exists(output_file):
                    self._bulk_file_paths.append(bulk_file)
                    self._output_file_paths.append(output_file)
                    if os.path.exists(fractures_file):
                        self._fracture_file_paths.append(fractures_file)
                else:
                    continue

    def __len__(self):
        return len(self._bulk_file_paths)

    def __getitem__(self, idx):
        fractures_path = []
        bulk_path = self._bulk_file_paths[idx]

        if idx in self._fracture_file_paths:
            fractures_path = self._fracture_file_paths[idx]
        output_path = self._output_file_paths[idx]

        if isinstance(bulk_path, (list, np.ndarray)):
            new_dataset= copy.deepcopy(self)
            new_dataset._bulk_file_paths = bulk_path
            new_dataset._fracture_file_paths = fractures_path
            new_dataset._output_file_paths = output_path
            return new_dataset

        bulk_features = np.load(bulk_path)["data"]

        if len(fractures_path) > 0:
            fractures_features = np.load(fractures_path)["data"]
        else:
            fractures_features = None
        output_features = np.load(output_path, allow_pickle=True)

        if self._two_dim:
            indices = [0,1,3]
            bulk_features = bulk_features[indices, ...]
            if fractures_features is not None:
                fractures_features = fractures_features[indices, ...]
            output_features = output_features[indices, ...]

        bulk_features_shape = bulk_features.shape

        flatten_bulk_features = bulk_features.reshape(-1)
        if fractures_features is not None:
            flatten_fracture_features = fractures_features.reshape(-1)
            not_nan_indices = np.argwhere(~np.isnan(flatten_fracture_features))
            flatten_bulk_features[not_nan_indices] = flatten_fracture_features[not_nan_indices]

        final_features = flatten_bulk_features.reshape(bulk_features_shape)
        final_features = torch.from_numpy(final_features)
        output_features = torch.from_numpy(output_features)

        if self.input_transform is not None:
            final_features = self.input_transform(final_features)
        if self.output_transform is not None:
            reshaped_output = torch.reshape(output_features, (*output_features.shape, 1, 1))
            output_features = np.squeeze(self.output_transform(reshaped_output))

        return final_features, output_features
