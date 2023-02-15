import torch
import numpy as np


def log_data(data):
    return torch.log(data)


def get_mean_std(data_loader):
    channels_sum = None
    channels_sqrd_sum = None
    num_batches = 0
    for input, output in data_loader:
        if channels_sum is None:
            channels_sum = list(np.zeros(input.shape[1]))
            channels_sqrd_sum = list(np.zeros(input.shape[1]))

        channels_sum += (torch.mean(input, dim=[0, 2, 3])).numpy()
        channels_sqrd_sum += (torch.mean(input ** 2, dim=[0, 2, 3])).numpy()
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std
