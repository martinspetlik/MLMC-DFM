import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def log_data(data):
    output_data = torch.empty((data.shape))
    if data.shape[0] == 3:
        output_data[0][...] = torch.log(data[0])
        output_data[1][...] = data[1]
        output_data[2][...] = torch.log(data[2])
    elif data.shape[0] < 3:
        for i in range(data.shape[0]):
            output_data[i][...] = torch.log(data[i])
    else:
        raise NotImplementedError("Log transformation implemented for 2D case only")

    return output_data

def exp_data(data):
    output_data = torch.empty((data.shape))
    if data.shape[0] == 3:
        output_data[0][...] = torch.exp(data[0])
        output_data[1][...] = data[1]
        output_data[2][...] = torch.exp(data[2])
    elif data.shape[0] < 3:
        for i in range(data.shape[0]):
            output_data[i][...] = torch.exp(data[i])
    else:
        raise NotImplementedError("Log transformation implemented for 2D case only")
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


def check_shapes(n_conv_layers, kernel_size, stride, pool_size, pool_stride, input_size=256):
    #n_layers = 0
    for i in range(n_conv_layers):
        if input_size < kernel_size:
            return -1, input_size

        input_size = int(((input_size - kernel_size) / stride)) + 1
        if pool_size > 0 and pool_stride > 0:
            input_size = int(((input_size - pool_size) / pool_stride)) + 1

    return 0, input_size


def get_mse_nrmse_r2(targets, predictions):
    targets_arr = np.array(targets)
    predictions_arr = np.array(predictions)

    squared_err_k = []
    std_tar_k = []
    r2_k = []
    mse_k = []
    rmse_k = []
    nrmse_k = []

    for i in range(targets_arr.shape[1]):
        targets = np.squeeze(targets_arr[:, i, ...])
        predictions = np.squeeze(predictions_arr[:, i, ...])
        squared_err_k.append((targets - predictions) ** 2)
        std_tar_k.append(np.std(targets))
        r2_k.append(1 - (np.sum(squared_err_k[i]) /
                         np.sum((targets - np.mean(targets)) ** 2)))
        mse_k.append(np.mean(squared_err_k[i]))
        rmse_k.append(np.sqrt(mse_k[i]))
        nrmse_k.append(rmse_k[i] / std_tar_k[i])

    return mse_k, rmse_k, nrmse_k, r2_k


class FrobeniusNorm(nn.Module):
    def __init__(self):
        super(FrobeniusNorm, self).__init__()

    def forward(self, y_pred, y_true):
        k_xx = y_pred[:, 0, ...] - y_true[:, 0, ...]
        k_xy = y_pred[:, 1, ...] - y_true[:, 1, ...]
        k_yy = y_pred[:, 2, ...] - y_true[:, 2, ...]

        k_xx = torch.mean(k_xx ** 2)
        k_xy = torch.mean(k_xy ** 2)
        k_yy = torch.mean(k_yy ** 2)

        return torch.sqrt(k_xx + 2 * k_xy + k_yy)

class CosineSimilarity(nn.Module):
    def __init__(self):
        super(CosineSimilarity, self).__init__()
    def forward(self, y_pred, y_true):
        criterion = nn.CosineSimilarity()
        return 1 - criterion(y_pred, y_true)

class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = torch.Tensor(weights)

    def forward(self, y_pred, y_true):
        mse_loss = F.mse_loss(y_pred, y_true, reduction='none')
        weighted_mse_loss = torch.mean(self.weights * mse_loss)
        return weighted_mse_loss

def get_loss_fn(loss_function):
    loss_fn_name = loss_function[0]
    loss_fn_params = loss_function[1]
    if loss_fn_name == "MSE" or loss_fn_name == "L2":
        return nn.MSELoss()
    elif loss_fn_name == "L1":
        return nn.L1Loss()
    elif loss_fn_name == "Frobenius":
        return FrobeniusNorm()
    elif loss_fn_name == "MSEweighted":
        return WeightedMSELoss(loss_fn_params)
    # elif loss_fn_name == "CosineSimilarity":
    #     return CosineSimilarity

