import torch
import numpy as np


def log_data(data):
    output_data = torch.empty((data.shape))
    output_data[0][...] = torch.log(data[0])
    output_data[1][...] = data[1]
    output_data[2][...] = torch.log(data[2])

    return output_data

def exp_data(data):
    output_data = torch.empty((data.shape))
    output_data[0][...] = torch.exp(data[0])
    output_data[1][...] = data[1]
    output_data[2][...] = torch.exp(data[2])

    return output_data


def get_mean_std(data_loader):
    channels_sum = None
    channels_sqrd_sum = None
    output_channels_sum = None
    output_channels_sqrd_sum = None
    num_batches = 0
    for input, output in data_loader:
        if channels_sum is None:
            channels_sum = list(np.zeros(input.shape[1]))
            channels_sqrd_sum = list(np.zeros(input.shape[1]))
            output_channels_sum = list(np.zeros(output.shape[1]))
            output_channels_sqrd_sum = list(np.zeros(output.shape[1]))

        channels_sum += (torch.mean(input, dim=[0, 2, 3])).numpy()
        channels_sqrd_sum += (torch.mean(input ** 2, dim=[0, 2, 3])).numpy()
        output_channels_sum += (torch.mean(output, dim=[0])).numpy()
        output_channels_sqrd_sum += (torch.mean(output ** 2, dim=[0])).numpy()
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    output_mean = output_channels_sum / num_batches
    output_std = (output_channels_sqrd_sum / num_batches - output_mean ** 2) ** 0.5

    return mean, std, output_mean, output_std


def reshape_to_tensors(tn_array, dim=2):
    tn = np.eye(dim)
    tn[np.triu_indices(dim)] = tn_array

    diagonal_values = np.diag(tn)
    symmetric_tn = tn + tn.T
    np.fill_diagonal(symmetric_tn, diagonal_values)
    return symmetric_tn


def get_eigendecomp(flatten_values, dim=2):
    tensor = reshape_to_tensors(np.squeeze(flatten_values), dim=dim)

    return np.linalg.eigh(tensor)


def n_layers_to_size_one(kernel_size, stride, input_size=256):
    n_layers = 0
    while True:
        input_size = int(((input_size - kernel_size) / stride)) + 1
        n_layers += 1

        if input_size == 1:
            return n_layers
        elif input_size < 1:
            return -1
