import zarr
import torch
import numpy as np
from torch.utils.data import Dataset


class DFM3DDataset(Dataset):
    """DFM models dataset"""

    def __init__(self, zarr_path, input_transform=None, output_transform=None, init_transform=None,
                 input_channels=None, output_channels=None, fractures_sep=False, cross_section=False, plot=False,
                 init_norm_use_all_features=False, mode="whole", train_size=None, val_size=None, test_size=None, chunk_size=128, return_centers_bulk_avg=False):
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
        self.centers = None
        if "centers" in zarr_file:
            self.centers = zarr_file["centers"]

        self.init_transform = init_transform
        self.input_transform = input_transform
        self.output_transform = output_transform

        self._input_channels = input_channels
        self._output_channels = output_channels
        self._fractures_sep = fractures_sep
        self._cross_section = cross_section
        self._init_transform_use_all_features = init_norm_use_all_features

        self._return_centers_bulk_avg = return_centers_bulk_avg

        self.plot = plot

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

    def __len__(self):
        # Return the number of samples
        #print("self.end: {}, self.start: {} ".format(self.end, self.start))
        return self.end - self.start

    def __getitem__(self, idx):
        idx = idx + self.start

        if idx > self.end:
            raise IndexError("Index has to be between start: {} and end: {}".format(self.start, self.end))

        input_sample = self.inputs[idx]
        output_sample = self.outputs[idx]

        if self.bulk_avg is not None:
            bulk_avg = self.bulk_avg[idx]
        if self.centers is not None:
            centers = self.centers[idx]

        if isinstance(idx, (slice, list)):
            raise NotImplementedError("Dataset index has to be int")
            # new_dataset = copy.deepcopy(self)
            # new_dataset.inputs = input_sample
            # new_dataset.outputs = output_sample
            # return new_dataset

        # Convert to PyTorch tensors
        input_tensor = torch.tensor(input_sample, dtype=torch.float32)
        output_tensor = torch.tensor(output_sample, dtype=torch.float32)

        if self._input_channels is not None:
            input_tensor = input_tensor[self._input_channels]
        if self._output_channels is not None and len(output_tensor) > 0:
            output_tensor = output_tensor[self._output_channels]

        if self.init_transform is not None:
            bulk_features_avg = np.mean(bulk_avg)
            self._bulk_features_avg = bulk_features_avg
            input_tensor, output_tensor = self.init_transform((bulk_features_avg, input_tensor, output_tensor, self._cross_section))

        reshaped_output = torch.reshape(output_tensor, (*output_tensor.shape, 1, 1))

        if self.input_transform is not None:
            input_tensor = self.input_transform(input_tensor)
        if self.output_transform is not None and len(output_tensor) > 0:
            output_tensor = np.squeeze(self.output_transform(reshaped_output))

        DFM3DDataset._check_nans(input_tensor, str_err="Input features contains NaN values, {}".format(idx))

        if len(output_tensor) > 0:
            DFM3DDataset._check_nans(output_tensor, str_err="Output features contains NaN values, idx: {}".format(idx))

        if self._return_centers_bulk_avg:
            return input_tensor, output_tensor, centers, bulk_features_avg

        return input_tensor, output_tensor

    @staticmethod
    def _check_nans(final_features, str_err="Data contains NaN values", file=None):
        has_nan = torch.any(torch.isnan(final_features))
        if has_nan:
            print("str err ", str_err)
            raise ValueError(str_err)
