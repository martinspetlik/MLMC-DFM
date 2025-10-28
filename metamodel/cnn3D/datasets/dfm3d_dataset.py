import zarr
import torch
import numpy as np
from torch.utils.data import Dataset


class DFM3DDataset(Dataset):
    """
    PyTorch Dataset for Discrete Fracture-Matrix (DFM) 3D models.
    Loads data from a Zarr file and supports transformations and splitting.
    """

    def __init__(self, zarr_path, input_transform=None, output_transform=None, init_transform=None,
                 input_channels=None, output_channels=None, fractures_sep=False, cross_section=False, plot=False,
                 init_norm_use_all_features=False, mode="whole", train_size=None, val_size=None, test_size=None, return_centers_bulk_avg=False):
        """
        Initialize the DFM3DDataset by opening a Zarr file and optionally applying transformations.
        :param zarr_path: str
            Path to the Zarr file containing the dataset.
        :param input_transform: callable, optional
            Transformation to apply to input tensors.
        :param output_transform: callable, optional
            Transformation to apply to output tensors.
        :param init_transform: callable, optional
            Initialization transformation applied to both input and output tensors.
        :param input_channels: list of int, optional
            If provided, selects specific channels of the input tensor.
        :param output_channels: list of int, optional
            If provided, selects specific channels of the output tensor.
        :param fractures_sep: bool, optional
            Whether to separate fractures from bulk features (default False).
        :param cross_section: bool, optional
            Whether to use cross-section data (default False).
        :param plot: bool, optional
            Enable plotting during dataset initialization.
        :param init_norm_use_all_features: bool, optional
            Whether to use all features during initial normalization.
        :param mode: str, optional
            Dataset mode: 'whole', 'train', 'val', 'test'.
        :param train_size: int, optional
            Number of training samples if splitting the dataset.
        :param val_size: int, optional
            Number of validation samples if splitting the dataset.
        :param test_size: int, optional
            Number of test samples if splitting the dataset.
        :param return_centers_bulk_avg: bool, optional
            Whether to return centers and bulk average along with the tensors.
        """
        super(DFM3DDataset, self).__init__()
        print("zarr path ", zarr_path)

        # Open Zarr file
        zarr_file = zarr.open(zarr_path, mode='r')
        self.data = zarr_file

        # Load main datasets
        self.inputs = zarr_file['inputs']
        self.outputs = zarr_file['outputs']

        # Optional datasets
        self.bulk_avg = zarr_file['bulk_avg'] if "bulk_avg" in zarr_file else None
        self.centers = zarr_file["centers"] if "centers" in zarr_file else None

        # Store transformations
        self.init_transform = init_transform
        self.input_transform = input_transform
        self.output_transform = output_transform

        # Dataset configuration flags
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._fractures_sep = fractures_sep
        self._cross_section = cross_section
        self._init_transform_use_all_features = init_norm_use_all_features
        self._return_centers_bulk_avg = return_centers_bulk_avg
        self.plot = plot

        # Determine dataset splits
        total_size = self.inputs.shape[0]
        if mode != "whole":
            train_size = train_size or int(total_size * 0.64)
            val_size = val_size or int(total_size * 0.16)
            test_size = test_size or int(total_size * 0.2)
            print("total size: {}, train size: {}, val size: {}, test size: {}".format(total_size, train_size, val_size, test_size))

        # Assign start/end indices for the dataset subset
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

        # Ensure end does not exceed dataset size
        self.end = np.min([self.end, total_size])

    def __len__(self):
        """
        Return the number of samples in the dataset slice.

        :return: int
            Number of samples
        """
        return self.end - self.start

    def __getitem__(self, idx):
        """
        Fetch a single sample from the dataset.

        :param idx: int
            Index of the sample within the selected dataset slice.
        :return: tuple
            input_tensor: torch.Tensor
                Input features tensor.
            output_tensor: torch.Tensor
                Output tensor (targets).
            centers: np.ndarray, optional
                Coordinates of the subdomain centers (if return_centers_bulk_avg=True).
            bulk_features_avg: float, optional
                Average bulk feature for normalization (if return_centers_bulk_avg=True).
        """
        idx = idx + self.start

        if idx > self.end:
            raise IndexError(f"Index has to be between start: {self.start} and end: {self.end}")

        # Load samples from Zarr arrays
        input_sample = self.inputs[idx]
        output_sample = self.outputs[idx]
        bulk_avg = self.bulk_avg[idx] if self.bulk_avg is not None else None
        centers = self.centers[idx] if self.centers is not None else None

        # Ensure integer index (slicing not supported)
        if isinstance(idx, (slice, list)):
            raise NotImplementedError("Dataset index has to be int")

        # Convert to PyTorch tensors
        input_tensor = torch.tensor(input_sample, dtype=torch.float32)
        output_tensor = torch.tensor(output_sample, dtype=torch.float32)

        # Select specific channels if provided
        if self._input_channels is not None:
            input_tensor = input_tensor[self._input_channels]
        if self._output_channels is not None and len(output_tensor) > 0:
            output_tensor = output_tensor[self._output_channels]

        # Apply initialization transform
        if self.init_transform is not None:
            bulk_features_avg = np.mean(bulk_avg)
            self._bulk_features_avg = bulk_features_avg
            input_tensor, output_tensor = self.init_transform(
                (bulk_features_avg, input_tensor, output_tensor, self._cross_section)
            )

        # Reshape output for 3D convolution compatibility
        reshaped_output = torch.reshape(output_tensor, (*output_tensor.shape, 1, 1))

        # Apply optional input/output transforms
        if self.input_transform is not None:
            input_tensor = self.input_transform(input_tensor)
        if self.output_transform is not None and len(output_tensor) > 0:
            output_tensor = np.squeeze(self.output_transform(reshaped_output))

        # Check for NaN values
        DFM3DDataset._check_nans(input_tensor, str_err=f"Input features contains NaN values, {idx}")
        if len(output_tensor) > 0:
            DFM3DDataset._check_nans(output_tensor, str_err=f"Output features contains NaN values, idx: {idx}")

        if self._return_centers_bulk_avg:
            return input_tensor, output_tensor, centers, bulk_features_avg

        return input_tensor, output_tensor

    @staticmethod
    def _check_nans(final_features, str_err="Data contains NaN values"):
        """
        Raise an error if NaNs are found in a tensor.
        :param final_features: torch.Tensor
            Tensor to check for NaNs.
        :param str_err: str, optional
            Error message to display if NaNs are detected.
        """
        has_nan = torch.any(torch.isnan(final_features))
        if has_nan:
            print("final_features ", final_features)
            print("str err ", str_err)
            raise ValueError(str_err)
