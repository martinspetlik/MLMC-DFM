import os
import re
import copy
import shutil
import zarr
import torch
import sklearn
import pandas as pd
#from skimage import io, transform
import numpy as np
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torchvision import transforms, utils


class DFM3DDataset(Dataset):
    """DFM models dataset"""

    def __init__(self, zarr_path, input_transform=None, output_transform=None, init_transform=None,
                 input_channels=None, output_channels=None, fractures_sep=False, cross_section=False, plot=False,
                 init_norm_use_all_features=False, mode="whole", train_size=None, val_size=None, test_size=None, chunk_size=128):
        super(DFM3DDataset, self).__init__()
        print("zarr path ", zarr_path)

        zarr_file = zarr.open(zarr_path, mode='r')

        self.data = zarr_file

        # Access the 'inputs' and 'outputs' datasets
        self.inputs = zarr_file['inputs']
        self.outputs = zarr_file['outputs']

        self.bulk_avg = None
        if "bulk_avg" in zarr_file:
            self.bulk_avg = zarr_file['bulk_avg']

        #self._remove_zero_rows()

        # Read the channel names (optional, for reference)
        # self.input_channel_names = self.inputs.attrs['channel_names']
        # self.output_channel_names = self.outputs.attrs['channel_names']

        self.init_transform = init_transform
        self.input_transform = input_transform
        self.output_transform = output_transform

        self._input_channels = input_channels
        self._output_channels = output_channels
        self._fractures_sep = fractures_sep
        self._cross_section = cross_section
        self._init_transform_use_all_features = init_norm_use_all_features

        self.plot = plot
        #self.chunk_size = chunk_size

        print("self.inputs shape ", self.inputs.shape)

        total_size = self.inputs.shape[0]
        if mode != "whole":
            train_size = train_size if train_size is not None else int(total_size * 0.64)
            val_size = val_size if val_size is not None else int(total_size * 0.16)
            test_size = test_size if test_size is not None else int(total_size * 0.2)

            print("total size: {}, train size: {}, val size: {}, test size: {}".format(total_size, train_size, val_size, test_size))

        if mode == "whole":
            self.start = 0
            self.end = total_size
        elif mode == "train":
            self.start = 0
            self.end = train_size
        elif mode == "val":
            self.start = train_size
            self.end = train_size + val_size
        elif mode == "test":
            self.start = train_size + val_size
            self.end = train_size + val_size + test_size

        self.end = np.min([self.end, total_size])

        print("start: {} end: {} ".format(self.start, self.end))

    def __len__(self):
        # Return the number of samples
        print("self.end: {}, self.start: {} ".format(self.end, self.start))
        return self.end - self.start #self.inputs.shape[0]

    # def __iter__(self):
    #     for i in range(self.start, self.end, self.chunk_size):
    #         inputs_chunk = self.inputs[i:min(i + self.chunk_size, self.end)]
    #         outputs_chunk = self.outputs[i:min(i + self.chunk_size, self.end)]
    #         for input_item, output_item in zip(inputs_chunk, outputs_chunk):
    #             yield torch.tensor(input_item), torch.tensor(output_item)

    def __getitem__(self, idx):
        idx = idx + self.start

        if idx > self.end:
            raise IndexError("Index has to be between start: {} and end: {}".format(self.start, self.end))

        #print("idx ", idx)

        input_sample = self.inputs[idx]
        output_sample = self.outputs[idx]

        if self.bulk_avg is not None:
            bulk_avg = self.bulk_avg[idx]

        if isinstance(idx, (slice, list)):
            raise NotImplementedError("Dataset index has to be int")
            # new_dataset = copy.deepcopy(self)
            # new_dataset.inputs = input_sample
            # new_dataset.outputs = output_sample
            # return new_dataset

        # Convert to PyTorch tensors
        input_tensor = torch.tensor(input_sample, dtype=torch.float32)
        output_tensor = torch.tensor(output_sample, dtype=torch.float32)

        #print("input shape: {}, output shape: {}".format(input_tensor.shape, output_tensor.shape))

        #input_tensor = input_tensor.permute(0, 4, 1, 2, 3)
        #output_tensor = output_tensor.permute(0, 4, 1, 2, 3)

        if self._input_channels is not None:
            input_tensor = input_tensor[self._input_channels]
        if self._output_channels is not None and len(output_tensor) > 0:
            output_tensor = output_tensor[self._output_channels]

        #@TODO: refactor
        if self.init_transform is not None:
            #print("input_sample[input_sample < np.max(np.abs(input_sample)) / 1e3] ", input_sample[input_sample < np.max(np.abs(input_sample)) / 1e2])
            #bulk_features_avg = np.mean(input_sample[input_sample < np.max(np.abs(input_sample)) / 1e2])
            #bulk_features_avg = np.abs(bulk_features_avg)
            bulk_features_avg = np.mean(bulk_avg)

            #bulk_features_avg = bulk_avg#[:, None, None, None]
            #print("bulk features avg ", bulk_features_avg)
            #
            #
            # print("avg input sample ", np.mean(input_sample))
            #
            # print("avg input sample ", np.mean(input_sample[input_sample<np.max(np.abs(input_sample))/1e3]))
            #
            # print("input sample min: {}, max: {}".format(np.min(np.abs(input_sample)), np.max(np.abs(input_sample))))
            #
            # print("order of magnitude min: {}, max: {}".format(min_order_of_magnitude, max_order_of_magnitude))
            # exit()
            self._bulk_features_avg = np.mean(bulk_avg) #bulk_features_avg
            #self._bulk_features_avg = bulk_avg#[:, None, None, None]

        # flatten_bulk_features = bulk_features.reshape(-1)
        # if fractures_features is not None:
        #     if self._fractures_sep is False:
        #         flatten_fracture_features = fractures_features.reshape(-1)
        #         not_nan_indices = np.argwhere(~np.isnan(flatten_fracture_features))
        #         #print("flatten_fracture_features[not_nan_indices] ", flatten_fracture_features[not_nan_indices])
        #         # if len(not_nan_indices) == 0:
        #         #     print("not nan indices ", not_nan_indices)
        #         flatten_bulk_features[not_nan_indices] = flatten_fracture_features[not_nan_indices]
        #     else:
        #         flatten_fracture_features = fractures_features.reshape(-1)
        #         nan_indices = np.argwhere(np.isnan(flatten_fracture_features))
        #         flatten_fracture_features[nan_indices] = 0
        #         fractures_channel = flatten_fracture_features[0, ...]
        # else:
        #     print("fr Nan")
        #
        # final_features = flatten_bulk_features.reshape(bulk_features_shape)


        # if self.init_transform is not None and self._init_transform_use_all_features:
        #     bulk_features_avg = np.mean(final_features)
        #     self._bulk_features_avg = bulk_features_avg

        # if self._fractures_sep:
        #     final_features = np.concatenate((final_features, np.expand_dims(fractures_channel, axis=0)), axis=0)
        #
        # if self._cross_section:
        #     final_features = np.concatenate((final_features, np.expand_dims(flatten_cross_section_features_channel, axis=0)), axis=0)
        #
        # final_features = torch.from_numpy(final_features)
        # output_features = torch.from_numpy(output_features)

        if self.init_transform is not None:
            #reshaped_output = torch.reshape(output_features, (*output_features.shape, 1, 1))
            #print("bulk features avg ", bulk_features_avg)
            input_tensor, output_tensor = self.init_transform((bulk_features_avg, input_tensor, output_tensor, self._cross_section))
            # print("input tensor shape ", input_tensor.shape)
            # print("reshaped output ", output_tensor.shape)

        reshaped_output = torch.reshape(output_tensor, (*output_tensor.shape, 1, 1))

        #print("reshape output shape ", reshaped_output.shape)

        if self.input_transform is not None:
            # print("input tensor shape ", input_tensor.shape)
            input_tensor = self.input_transform(input_tensor)
        if self.output_transform is not None and len(output_tensor) > 0:
            output_tensor = np.squeeze(self.output_transform(reshaped_output))

        #print("reshaped output ", reshaped_output)
        #print("reshaped output shape ", reshaped_output.shape)
        # else:
        #     print("else reshaped output ", reshaped_output)
        #print("reshaped output shape", reshaped_output.shape)
        #exit()
        #print("input transform ", self.input_transform)
        #print("output transform ", self.output_transform)
        # if output_features[0] < -2:
        #     print("bulk path ", bulk_path)
        #     print("output features orig ", output_features_orig)
        #     print("output_features ", output_features)
        #     print("not_nan_indices ", not_nan_indices)
        # else:
        #     pass
        #     #print("else output_features orig", output_features_orig)

        DFM3DDataset._check_nans(input_tensor, str_err="Input features contains NaN values, {}".format(idx))

        if len(output_tensor) > 0:
            print("output tensor ", output_tensor)
            DFM3DDataset._check_nans(output_tensor, str_err="Output features contains NaN values, idx: {}".format(idx))


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


        #print("output tensor ", output_tensor)

        return input_tensor, output_tensor

    @staticmethod
    def _check_nans(final_features, str_err="Data contains NaN values", file=None):
        has_nan = torch.any(torch.isnan(final_features))
        if has_nan:
            print("str err ", str_err)
            #print("file ", os.path.dirname(file))
            #shutil.rmtree(os.path.dirname(file))
            raise ValueError(str_err)
