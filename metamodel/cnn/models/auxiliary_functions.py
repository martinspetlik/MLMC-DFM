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


def check_shapes(n_conv_layers, kernel_size, stride, pool_size, pool_stride, input_size=256):
    #n_layers = 0
    for i in range(n_conv_layers):
        if input_size < kernel_size:
            return -1, input_size

        input_size = int(((input_size - kernel_size) / stride)) + 1
        if pool_size > 0 and pool_stride > 0:
            input_size = int(((input_size - pool_size) / pool_stride)) + 1

    return 0, input_size


def get_mse_nrmse(targets, predictions):
    targets_arr = np.array(targets)
    predictions_arr = np.array(predictions)

    print(targets_arr.shape)
    
    squared_err_k_xx_inv = (targets_arr[:, 0, ...] - predictions_arr[:, 0, ...]) ** 2
    squared_err_k_xy_inv = (targets_arr[:, 1, ...] - predictions_arr[:, 1, ...]) ** 2
    squared_err_k_yy_inv = (targets_arr[:, 2, ...] - predictions_arr[:, 2, ...]) ** 2

    std_tar_k_xx_inv = np.std(targets_arr[:, 0, ...])
    std_tar_k_xy_inv = np.std(targets_arr[:, 1, ...])
    std_tar_k_yy_inv = np.std(targets_arr[:, 2, ...])

    mse_k_xx_inv = np.mean(squared_err_k_xx_inv)
    mse_k_xy_inv = np.mean(squared_err_k_xy_inv)
    mse_k_yy_inv = np.mean(squared_err_k_yy_inv)

    nrmse_k_xx_inv = np.sqrt(mse_k_xx_inv) / std_tar_k_xx_inv
    nrmse_k_xy_inv = np.sqrt(mse_k_xy_inv) / std_tar_k_xy_inv
    nrmse_k_yy_inv = np.sqrt(mse_k_yy_inv) / std_tar_k_yy_inv
    
    return [mse_k_xx_inv, mse_k_xy_inv, mse_k_yy_inv], [nrmse_k_xx_inv, nrmse_k_xy_inv, nrmse_k_yy_inv]


def plot_samples(data_loader, n_samples=10):
    import matplotlib.pyplot as plt
    for idx, data in enumerate(data_loader):
        if idx > n_samples:
            break
        input, output = data
        #img = img / 2 + 0.5  # unnormalize
        #npimg = img.numpy()
        plt_input = input[0]
        print("plt_input ", plt_input)

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
        axes[0].matshow(plt_input[0])
        axes[1].matshow(plt_input[1])
        axes[2].matshow(plt_input[2])
        #fig.colorbar(caxes)
        plt.savefig("input_{}.pdf".format(idx))
        plt.show()

        # plt.matshow(plt_input[0])
        # plt.matshow(plt_input[1])
        # plt.matshow(plt_input[2])
        # plt.show()

    exit()
