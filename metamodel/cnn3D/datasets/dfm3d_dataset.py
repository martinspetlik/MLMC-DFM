import os
import re
import copy
import shutil

import torch
import sklearn
import pandas as pd
#from skimage import io, transform
import numpy as np
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class DFMDataset(Dataset):
    """DFM models dataset"""

    def __init__(self, zarr_file, input_transform=None, output_transform=None, init_transform=None,
                 input_channels=None, output_channels=None, fractures_sep=False, cross_section=False, plot=False, init_norm_use_all_features=False):
        self.zarr_file = zarr_file

        # Access the 'inputs' and 'outputs' datasets
        self.inputs = self.zarr_file['inputs']
        self.outputs = self.zarr_file['outputs']

        # Read the channel names (optional, for reference)
        self.input_channel_names = self.inputs.attrs['channel_names']
        self.output_channel_names = self.outputs.attrs['channel_names']

        self.init_transform = init_transform
        self.input_transform = input_transform
        self.output_transform = output_transform

        self._input_channels = input_channels
        self._output_channels = output_channels
        self._fractures_sep = fractures_sep
        self._cross_section = cross_section
        self._init_transform_use_all_features = init_norm_use_all_features


        self.plot = plot


    #@TODO: DataLoader should be responsible for shuffling
    # def shuffle(self, seed):
    #     np.random.seed(seed)
    #     perm = np.random.permutation(len(self._bulk_file_paths))
    #     self._bulk_file_paths = list(np.array(self._bulk_file_paths)[perm])
    #     self._output_file_paths = list(np.array(self._output_file_paths)[perm])
    #
    #     if len(self._fracture_file_paths) == len(self._bulk_file_paths):
    #         self._fracture_file_paths = list(np.array(self._fracture_file_paths)[perm])
    #     if len(self._cross_section_file_paths) == len(self._bulk_file_paths):
    #         self._cross_section_file_paths = list(np.array(self._cross_section_file_paths)[perm])

    def __len__(self):
        # Return the number of samples
        return self.inputs.shape[0]

    def __getitem__(self, idx):

        input_sample = self.inputs[idx]
        output_sample = self.outputs[idx]

        # Convert to PyTorch tensors
        input_tensor = torch.tensor(input_sample, dtype=torch.float32)
        output_tensor = torch.tensor(output_sample, dtype=torch.float32)

        #@TODO: refactor

        if self.init_transform is not None:
            bulk_features_avg = np.mean(bulk_features)
            self._bulk_features_avg = bulk_features_avg

        if cross_section_features is not None:
            flatten_cross_section_features = cross_section_features.reshape(-1)
            nan_indices = np.argwhere(np.isnan(flatten_cross_section_features))
            flatten_cross_section_features[nan_indices] = 1
            flatten_cross_section_features_channel = flatten_cross_section_features.reshape(bulk_features_shape[1:])

        flatten_bulk_features = bulk_features.reshape(-1)
        if fractures_features is not None:
            if self._fractures_sep is False:
                flatten_fracture_features = fractures_features.reshape(-1)
                not_nan_indices = np.argwhere(~np.isnan(flatten_fracture_features))
                #print("flatten_fracture_features[not_nan_indices] ", flatten_fracture_features[not_nan_indices])
                # if len(not_nan_indices) == 0:
                #     print("not nan indices ", not_nan_indices)
                flatten_bulk_features[not_nan_indices] = flatten_fracture_features[not_nan_indices]
            else:
                flatten_fracture_features = fractures_features.reshape(-1)
                nan_indices = np.argwhere(np.isnan(flatten_fracture_features))
                flatten_fracture_features[nan_indices] = 0
                fractures_channel = flatten_fracture_features[0, ...]
        else:
            print("fr Nan")

        final_features = flatten_bulk_features.reshape(bulk_features_shape)


        if self.init_transform is not None and self._init_transform_use_all_features:
            bulk_features_avg = np.mean(final_features)
            self._bulk_features_avg = bulk_features_avg

        if self._fractures_sep:
            final_features = np.concatenate((final_features, np.expand_dims(fractures_channel, axis=0)), axis=0)

        if self._cross_section:
            final_features = np.concatenate((final_features, np.expand_dims(flatten_cross_section_features_channel, axis=0)), axis=0)

        final_features = torch.from_numpy(final_features)
        output_features = torch.from_numpy(output_features)

        if self._input_channels is not None:
            final_features = final_features[self._input_channels]
        if self._output_channels is not None and len(output_features) > 0:
            output_features = output_features[self._output_channels]

        if self.init_transform is not None:
            reshaped_output = torch.reshape(output_features, (*output_features.shape, 1, 1))
            final_features, reshaped_output = self.init_transform((bulk_features_avg, final_features, reshaped_output, self._cross_section))
        else:
            reshaped_output = torch.reshape(output_features, (*output_features.shape, 1, 1))

        # else:
        #     print("else reshaped output ", reshaped_output)


            #print("reshaped output shape", reshaped_output.shape)

            #exit()

        if self.input_transform is not None:
            final_features = self.input_transform(final_features)
        if self.output_transform is not None and len(output_features) > 0:
            #print("self.output transform ", self.output_transform)
            output_features = np.squeeze(self.output_transform(reshaped_output))

        # if output_features[0] < -2:
        #     print("bulk path ", bulk_path)
        #     print("output features orig ", output_features_orig)
        #     print("output_features ", output_features)
        #     print("not_nan_indices ", not_nan_indices)
        # else:
        #     pass
        #     #print("else output_features orig", output_features_orig)


        DFMDataset._check_nans(final_features, str_err="Input features contains NaN values, {}".format(bulk_path), file=bulk_path)

        if len(output_features) > 0:
            DFMDataset._check_nans(output_features, str_err="Output features contains NaN values, {}".format(output_path), file=output_path)


        #print("final features shape ", final_features.shape)
        #print("final features [3, :] ", final_features[1, :])

        # if output_features is None:
        #     output_features = torch.empty(1, dtype=None)
        #     print("output features ", output_features)

        # if self.plot:
        #     #print("self. output transform ", len(self.output_transform))
        #     if output_features[0] == 1:
        #         print("output features ", output_features)
        #         print("bulk_path ", bulk_path)
        #     else:
        #         pass
        #         #print("else")

        return final_features, output_features

    @staticmethod
    def _check_nans(final_features, str_err="Data contains NaN values", file=None):
        has_nan = torch.any(torch.isnan(final_features))
        if has_nan:
            print("str err ", str_err)
            print("file ", os.path.dirname(file))
            shutil.rmtree(os.path.dirname(file))
            #raise ValueError(str_err)
