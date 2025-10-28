import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import QuantileTransformer, RobustScaler


voigt_coords = {
            1: [(0, 0)],
            2: [(0, 0), (1, 1), (0, 1)],
            3: [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
        }

idx_to_full = {
            1: [(0, 0)],
            3: [[0, 2], [2, 1]],
            6: [[0, 5, 4], [5, 1, 3], [4, 3, 2]]
        }

idx_to_voigt = {
    1: ([0], [0]),
    2: ([0, 1, 0], [0, 1, 1]),
    3: ([0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1])
}

def voigt_to_tn(vec):
    """
    :param vec: (N, n_voigt)
    :return: (N, dim, dim)
    """
    _, n_voigt = vec.shape
    return vec[:, idx_to_full[n_voigt]]


class QuantileTRF():
    def __init__(self):
        self.quantile_trfs_out = None
        self.quantile_trfs_in = None

    def quantile_transform_in(self, passed_data):
        return quantile_transform_trf(passed_data, self.quantile_trfs_in)

    def quantile_transform_out(self, passed_data):
        return quantile_transform_trf(passed_data, self.quantile_trfs_out)

    def quantile_inv_transform_out(self, passed_data):
        return quantile_inv_transform_trf(passed_data, self.quantile_trfs_out)


class NormalizeData():
    def __init__(self):
        self.input_indices = [0, 1, 2, 3, 4, 5]
        self.output_indices = [0, 1, 2, 3, 4, 5]
        self.input_mean = [0, 0, 0, 0, 0, 0]
        self.output_mean = [0, 0, 0, 0, 0, 0]
        self.input_std = [1, 1, 1, 1, 1, 1]
        self.output_std = [1, 1, 1, 1, 1, 1]
        self.output_quantiles = []

    def normalize_input(self, data):
        output_data = torch.empty((data.shape))
        for i in range(data.shape[0]):
            if i in self.input_indices:
                output_data[i][...] = (data[i] - self.input_mean[i]) /self.input_std[i]
            else:
                output_data[i][...] = data[i]

        return output_data

    def normalize_output(self, data):
        output_data = torch.empty((data.shape))
        for i in range(data.shape[0]):
            if i in self.output_indices:
                if hasattr(self, 'output_quantiles') and len(self.output_quantiles) > 0:
                    output_data[i][...] =(data[i] - self.output_quantiles[1, i]) / (self.output_quantiles[2, i] - self.output_quantiles[0, i])
                else:

                    output_data[i][...] = (data[i] - self.output_mean[i]) /self.output_std[i]
            else:
                output_data[i][...] = data[i]
        return output_data


def log_all_data(data):
    output_data = torch.empty((data.shape))
    if data.shape[0] == 3:
        output_data[0][...] = torch.log(data[0])

        # flatten_data = data[1].flatten()
        # positive_data_indices = flatten_data >= 1e-15
        # negative_data_indices = flatten_data < 1e-15
        #
        # preprocessed_negative_k_xy = -torch.log(np.abs(flatten_data[negative_data_indices]))
        # preprocessed_positive_k_xy = torch.log(flatten_data[positive_data_indices])
        #
        # flatten_data[positive_data_indices] = preprocessed_positive_k_xy
        # flatten_data[negative_data_indices] = preprocessed_negative_k_xy

        #print("data[1].shape ", data[1].shape)
        #
        # output_data[1][...] = np.reshape(flatten_data, data[1].shape)
        output_data[1][...] = torch.log(data[1] + 0.03)#torch.abs(torch.min(data[1])))
        output_data[2][...] = torch.log(data[2])
    else:
        raise NotImplementedError("Log transformation implemented for 2D case only")

    return output_data


def log10_all_data(data):
    output_data = torch.empty((data.shape))
    if data.shape[0] == 3:
        output_data[0][...] = torch.log10(data[0])

        flatten_data = data[1].flatten()
        print("log10 flatten data ", flatten_data)
        positive_data_indices = flatten_data >= 1e-15
        negative_data_indices = flatten_data < 1e-15

        preprocessed_negative_k_xy = -torch.log10(np.abs(flatten_data[negative_data_indices]))
        preprocessed_positive_k_xy = torch.log10(flatten_data[positive_data_indices])



        flatten_data[positive_data_indices] = preprocessed_positive_k_xy
        flatten_data[negative_data_indices] = preprocessed_negative_k_xy

        print("log10 final flatten data ", flatten_data)

        output_data[1][...] = np.reshape(flatten_data, data[1].shape)
        output_data[2][...] = torch.log10(data[2])
    else:
        raise NotImplementedError("Log transformation implemented for 2D case only")

    return output_data


def init_norm(data):
    bulk_features_avg, input, output, cross_section_flag = data
    #avg_k = torch.mean(input)

    #print("bulk_features_avg ", bulk_features_avg)
    # print("input shape ", input.shape)
    # print("output.shape ", output.shape)

    # print("bulk features avg ", bulk_features_avg)
    # print("output ", output)

    # if cross_section_flag:
    #     input[:3, :] /= bulk_features_avg
    # else:
    input /= bulk_features_avg#[:, None, None, None]

    #print("bulk_features_avg ", bulk_features_avg)
    #print("orig output ", output)
    output /= bulk_features_avg
    #print("divided output ", output)



    return input, output


def arcsinh_data(data):
    output_data = torch.empty((data.shape))
    #print("data.shape ", data.shape)
    if data.shape[0] == 3:
        output_data[0][...] = data[0]
        output_data[1][...] = torch.arcsinh(data[1])
        print("data[1] ", data[1])
        print("torch.arcsinh(data[1]) ", torch.arcsinh(data[1]))
        output_data[2][...] = data[2]
    # elif data.shape[0] == 4:
    #     output_data[0][...] = torch.log(data[0])
    #     output_data[1][...] = data[1]
    #     output_data[2][...] = torch.log(data[2])
    #     output_data[3][...] = data[3]
    # elif data.shape[0] < 3:
    #     for i in range(data.shape[0]):
    #         output_data[i][...] = torch.log(data[i])
    else:
        raise NotImplementedError("Log transformation implemented for 2D case only")

    return output_data

def log_data(data):
    #output_data = torch.empty((data.shape))
    # print("data shape ", data.shape)
    # print("data ", data)

    if data.shape[0] == 3:
        data[0][...] = torch.log(data[0])
        data[1][...] = data[1]
        data[2][...] = torch.log(data[2])
    elif data.shape[0] == 4:
        data[0] = torch.log(data[0])
        #data[1][...] = data[1]
        data[2] = torch.log(data[2])
        #data[3][...] = data[3]
    elif data.shape[0] == 6:
        data[0][...] = torch.log(data[0])  # k_xx
        data[1][...] = torch.log(data[1])  # k_yy
        data[2][...] = torch.log(data[2])  # k_zz
        data[3][...] = data[3]  # k_yz
        data[4][...] = data[4]  # k_xz
        data[5][...] = data[5]  # k_xy
    elif data.shape[0] < 3:
        for i in range(data.shape[0]):
            data[i][...] = torch.log(data[i])

    #print("log data ", data)

    return data


def log10_data(data):
    #print("data.shape ", data.shape)
    output_data = torch.empty((data.shape))
    if data.shape[0] == 3:
        data[0][...] = torch.log10(data[0])
        data[1][...] = data[1]
        data[2][...] = torch.log10(data[2])
    elif data.shape[0] == 4:
        data[0][...] = torch.log10(data[0])
        data[1][...] = data[1]
        data[2][...] = torch.log10(data[2])
        data[3][...] = data[3]
    elif data.shape[0] == 6:
        data[0][...] = torch.log10(data[0])  # k_xx
        data[1][...] = torch.log10(data[1])  # k_yy
        data[2][...] = torch.log10(data[2])  # k_zz
        data[3][...] = data[3]  # k_yz
        data[4][...] = data[4]  # k_xz
        data[5][...] = data[5]  # k_xy
    elif data.shape[0] < 3:
        for i in range(data.shape[0]):
            data[i][...] = torch.log10(data[i])
    else:
        raise NotImplementedError("Log transformation implemented for 2D case only")

    return data


def quantile_transform_fit(data, indices=[], transform_type=None):
    transform_obj = []
    for i in range(data.shape[0]):
        if i in indices:
            if transform_type == "RobustScaler":
                transformer = RobustScaler()
            else:
                transformer = QuantileTransformer(n_quantiles=10000, random_state=0, output_distribution="normal")
            if torch.is_tensor(data[i]):
                transform_obj.append(transformer.fit(data[i].reshape(-1, 1).numpy()))
            else:
                transform_obj.append(transformer.fit(data[i].reshape(-1, 1)))
        else:
            transform_obj.append(None)
    return transform_obj


def quantile_transform_trf(data, quantile_trfs):
    trf_data = torch.empty((data.shape))
    for i in range(data.shape[0]):
        if quantile_trfs[i] is not None:
            transformed_data = quantile_trfs[i].transform(data[i].reshape(-1, 1).numpy())
            trf_data[i][...] = torch.from_numpy(np.reshape(transformed_data, data[i].shape))
        else:
            trf_data[i][...] = data[i]
    return trf_data

def quantile_inv_transform_trf(data, quantile_trfs):
    trf_data = torch.empty((data.shape))
    for i in range(data.shape[0]):
        if quantile_trfs[i] is not None:
            transformed_data = quantile_trfs[i].inverse_transform(data[i].reshape(-1, 1).numpy())
            trf_data[i][...] = torch.from_numpy(np.reshape(transformed_data, data[i].shape))
        else:
            trf_data[i][...] = data[i]
    return trf_data

# def quantile_transform_offdiagonal_fit(data, transform_type=None):
#     transform_obj = []
#     for i in range(data.shape[0]):
#         if i == 1:
#             if transform_type == "RobustScaler":
#                 transformer = RobustScaler()
#             else:
#                 transformer = QuantileTransformer(n_quantiles=10000, random_state=0, output_distribution="normal")
#             if torch.is_tensor(data[i]):
#                 transform_obj.append(transformer.fit(data[i].reshape(-1, 1).numpy()))
#             else:
#                 transform_obj.append(transformer.fit(data[i].reshape(-1, 1)))
#     return transform_obj
#
#
# def quantile_transform_offdiagonal_trf(data, quantile_trfs):
#     return_data = torch.empty((data.shape))
#     for i in range(data.shape[0]):
#         if i == 1:
#             #print("data ", data[i])
#             transformed_data = quantile_trfs[0].transform(data[i].reshape(-1, 1).numpy())
#             return_data[i][...] = torch.from_numpy(np.reshape(transformed_data, data[i].shape))
#         else:
#             return_data[i][...] = data[i]
#     return return_data

def exp_data(data):
    output_data = torch.empty((data.shape))
    if data.shape[0] == 3 and len(data.shape) != 4:
        output_data[0][...] = torch.exp(data[0])
        output_data[1][...] = data[1]
        output_data[2][...] = torch.exp(data[2])
    elif data.shape[0] < 3 and len(data.shape) != 4:
        for i in range(data.shape[0]):
            output_data[i][...] = torch.exp(data[i])
    elif len(data.shape) == 4:
        if data.shape[1] == 6:
            output_data[:, 0, ...] = torch.exp(data[:, 0])
            output_data[:, 1, ...] = torch.exp(data[:, 1])
            output_data[:, 2, ...] = torch.exp(data[:, 2])
            output_data[:, 3, ...] = data[:, 3]
            output_data[:, 4, ...] = data[:, 4]
            output_data[:, 5, ...] = data[:, 5]
        else:
            output_data[:, 0, ...] = torch.exp(data[:, 0])
            output_data[:, 1, ...] = data[:, 1]
            output_data[:, 2,...] = torch.exp(data[:, 2])
    elif data.shape[0] == 6:
        output_data[0][...] = torch.exp(data[0])  # k_xx
        output_data[1][...] = torch.exp(data[1])  # k_yy
        output_data[2][...] = torch.exp(data[2])  # k_zz
        output_data[3][...] = data[3]  # k_yz
        output_data[4][...] = data[4]  # k_xz
        output_data[5][...] = data[5]  # k_xy
    else:
        raise NotImplementedError("Log transformation implemented for 2D case only")



    return output_data


# def exp_data(data):
#     print("data shape ", data.shape)
#     # print("data ", data)
#
#     output_data = torch.empty((data.shape))
#     if data.shape[0] == 3:
#         output_data[0][...] = torch.exp(data[0])
#         output_data[1][...] = data[1]
#         output_data[2][...] = torch.exp(data[2])
#     elif data.shape[0] < 3:
#         for i in range(data.shape[0]):
#             output_data[i][...] = torch.exp(data[i])
#     elif len(data.shape) == 4:
#         if data.shape[1] == 3:
#             data = np.transpose(data, (1, 2, 3, 0))
#             output_data = np.transpose(output_data, (1, 2, 3, 0))
#             #print("data[:][0] ", data[:][0])
#             output_data[0][...] = torch.exp(data[0][...])
#             output_data[1][...] = data[1][...]
#             output_data[2][...] = torch.exp(data[2][...])
#             output_data = np.transpose(output_data, (3, 0, 1, 2))
#     else:
#         raise NotImplementedError("Log transformation implemented for 2D case only")
#     #print("output data ", output_data)
#     return output_data

def power_10_data(data):
    output_data = torch.empty((data.shape))
    if data.shape[0] == 3:
        output_data[0][...] = torch.pow(10, data[0])
        output_data[1][...] = data[1]
        output_data[2][...] = torch.pow(10, data[2])
    elif data.shape[0] < 3:
        for i in range(data.shape[0]):
            output_data[i][...] = torch.pow(10, data[i])
    else:
        raise NotImplementedError("Log transformation implemented for 2D case only")
    return output_data


def power_10_all_data(data):
    output_data = torch.empty((data.shape))
    if data.shape[0] == 3:
        output_data[0][...] = torch.pow(10, data[0])
        #output_data[0][...] = torch.exp(data[0])

        flatten_data = data[1].flatten()
        print("flatten data ", flatten_data)

        positive_data_indices = flatten_data >= 1e-15
        negative_data_indices = flatten_data < 1e-15

        print("flatten_data[negative_data_indices]", flatten_data[negative_data_indices])
        print("flatten_data[positive_data_indices]", flatten_data[positive_data_indices])

        #preprocessed_positive_k_xy = -torch.pow(torch.from_numpy(10), flatten_data[negative_data_indices])
        #preprocessed_negative_k_xy = torch.pow(torch.from_numpy(10), -flatten_data[positive_data_indices])

        preprocessed_positive_k_xy = - 10 ** flatten_data[negative_data_indices]
        preprocessed_negative_k_xy = 10 ** -flatten_data[positive_data_indices]

        flatten_data[negative_data_indices] = preprocessed_positive_k_xy
        flatten_data[positive_data_indices] = preprocessed_negative_k_xy

        print("final flatten data ", flatten_data)

        output_data[1][...] = np.reshape(flatten_data, data[1].shape)
        output_data[2][...] = torch.pow(10, data[2])
    elif data.shape[0] < 3:
        for i in range(data.shape[0]):
            output_data[i][...] = torch.pow(10, data[i])
    else:
        raise NotImplementedError("Log transformation implemented for 2D case only")
    return output_data


def get_mean_std(data_loader, output_iqr=[], mean_dims=None):
    channels_sum = None
    channels_sqrd_sum = None
    output_channels_sum = None
    output_channels_sqrd_sum = None
    num_batches = 0
    output_data_list = []
    quantiles = []

    if mean_dims is None:
        mean_dims = [0, 2, 3]

    for input, output in data_loader:
        #print("output ", output)
        if channels_sum is None:
            channels_sum = list(np.zeros(input.shape[1]))
            channels_sqrd_sum = list(np.zeros(input.shape[1]))
            output_channels_sum = list(np.zeros(output.shape[1]))
            output_channels_sqrd_sum = list(np.zeros(output.shape[1]))

        if len(output_iqr) > 0:
            output_data_list.extend(output.numpy())

        channels_sum += (torch.nanmean(input, dim=mean_dims)).numpy()
        channels_sqrd_sum += (torch.nanmean(input ** 2, dim=mean_dims)).numpy()
        output_channels_sum += (torch.nanmean(output, dim=[0])).numpy()
        output_channels_sqrd_sum += (torch.nanmean(output ** 2, dim=[0])).numpy()
        num_batches += 1

    if len(output_iqr) > 0:
        output_data = np.array(output_data_list)
        quantiles = np.quantile(output_data, output_iqr, axis=0)

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    output_mean = output_channels_sum / num_batches
    output_std = (output_channels_sqrd_sum / num_batches - output_mean ** 2) ** 0.5

    return mean, std, output_mean, output_std, quantiles
    #return mean, std, output_mean, output_std



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


def check_shapes(n_conv_layers, kernel_size, stride, pool_size, pool_stride, pool_indices, input_size=256, padding=0, debug=False):
    """
    Compute the output size after a sequence of conv and optional pooling layers.

    Parameters:
    - n_conv_layers (int): number of convolutional layers
    - kernel_size (int): convolution kernel size (assumed square)
    - stride (int): convolution stride
    - pool_size (int): pooling window size (assumed square)
    - pool_stride (int): pooling stride
    - pool_indices (dict or None): keys are conv layer indices after which pooling is applied
    - input_size (int): spatial input size (height/width)
    - padding (int): padding size applied on each side of conv input
    - debug (bool): whether to print intermediate sizes

    Returns:
    - tuple: (status, final_output_size)
      status: 0 if successful, -1 if input size is smaller than kernel size at any layer
    """

    for i in range(n_conv_layers):
        if debug:
            print(f"Layer {i}: input_size = {input_size}")

        if input_size < kernel_size:
            if debug:
                print(f"Input size {input_size} smaller than kernel size {kernel_size} at layer {i}")
            return -1, input_size

        # Calculate output size after convolution
        input_size = int(((input_size - kernel_size + 2 * padding) / stride)) + 1

        # Check if pooling applies after this conv layer
        if pool_indices is not None and i in pool_indices:
            if pool_size > 0 and pool_stride > 0:
                input_size = int(((input_size - pool_size) / pool_stride)) + 1
                if debug:
                    print(f"After pooling at layer {i}: size = {input_size}")

    if debug:
        print(f"Final output size after {n_conv_layers} conv layers: {input_size}")

    return 0, input_size


def get_mse_nrmse_r2(targets, predictions):
    """
    :param targets: Ground truth values, shape (samples, outputs, ...)
    :param predictions: Predicted values, same shape as targets
    :return: Lists of MSE, RMSE, NRMSE, and R² values for each output dimension
    """
    targets_arr = np.array(targets)
    predictions_arr = np.array(predictions)

    mse_k = []
    rmse_k = []
    nrmse_k = []
    r2_k = []

    for i in range(targets_arr.shape[1]):
        # Extract i-th output across all samples
        tar_i = np.squeeze(targets_arr[:, i, ...])
        pred_i = np.squeeze(predictions_arr[:, i, ...])

        # Compute squared error
        squared_errors = (tar_i - pred_i) ** 2

        # Mean squared error
        mse = np.mean(squared_errors)

        # Root mean squared error
        rmse = np.sqrt(mse)

        # Standard deviation of targets
        std_tar = np.std(tar_i)

        # Normalized RMSE
        nrmse = rmse / std_tar if std_tar != 0 else np.nan

        # Total variance of targets
        ss_tot = np.sum((tar_i - np.mean(tar_i)) ** 2)

        # Residual sum of squares
        ss_res = np.sum(squared_errors)

        # Coefficient of determination (R²)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

        # Append results
        mse_k.append(mse)
        rmse_k.append(rmse)
        nrmse_k.append(nrmse)
        r2_k.append(r2)

    return mse_k, rmse_k, nrmse_k, r2_k

def calculate_eigen(matrix):
    trace = np.trace(matrix)
    det = np.linalg.det(matrix)
    trace_squared = np.trace(matrix @ matrix)

    # Coefficients of the cubic characteristic polynomial
    a = -trace
    b = (trace ** 2 - trace_squared) / 2
    c = -det

    # Solve the cubic equation using np.roots
    coeffs = [1, a, b, c]
    eigenvalues = np.roots(coeffs)

    eigenvectors = []
    # Solve for eigenvectors corresponding to each eigenvalue
    for eigenvalue in eigenvalues:
        # Construct (A - lambda * I) for eigenvector equation
        A_minus_lambda_I = matrix - eigenvalue * np.eye(3)

        # Find the null space (eigenvector) by solving (A - lambda * I) * v = 0
        # Since it is a homogeneous system, the null space can be found using np.linalg.svd or np.linalg.matrix_rank
        # Find the eigenvector (null space of A - lambda * I)

        # Perform SVD to get the null space
        _, _, vh = np.linalg.svd(A_minus_lambda_I)
        eigenvector = vh[-1]  # The last row of vh corresponds to the null space (eigenvector)

        # Normalize the eigenvector
        eigenvector = eigenvector / np.linalg.norm(eigenvector)
        eigenvectors.append(eigenvector.astype("float"))

    # print("eigenvalues: ", eigenvalues)
    # print("eigenvectors: ", eigenvectors)

    return eigenvalues.astype("float"), eigenvectors


def get_mse_nrmse_r2_eigh_3D(targets, predictions):
    targets_arr = np.array(targets)
    predictions_arr = np.array(predictions)

    target_evals = []
    pred_evals = []
    target_evecs = []
    pred_evecs = []

    pred_evec_1 = []
    pred_evec_2 = []
    pred_evec_3 = []
    target_evec_1 = []
    target_evec_2 = []
    target_evec_3 = []

    targets_arr = voigt_to_tn(targets_arr)
    predictions_arr = voigt_to_tn(predictions_arr)

    # print("targets arr ", targets_arr)
    # print("predictions arr ", predictions_arr)

    for matrix in targets_arr:
        eigenvalues, eigenvectors = calculate_eigen(matrix)

        target_evals.append(eigenvalues)

        target_evec_1.append(eigenvectors[0])
        target_evec_2.append(eigenvectors[1])
        target_evec_3.append(eigenvectors[2])
        #target_evecs.append(eigenvectors)

    #print("target evals ", target_evals)

    for matrix in predictions_arr:
        eigenvalues, eigenvectors = calculate_eigen(matrix)

        pred_evals.append(eigenvalues)
        #pred_evecs.append(eigenvectors)

        pred_evec_1.append(eigenvectors[0])
        pred_evec_2.append(eigenvectors[1])
        pred_evec_3.append(eigenvectors[2])

    target_evals = np.array(target_evals)
    #target_evecs = np.array(target_evecs)

    target_evec_1 = np.array(target_evec_1)
    target_evec_2 = np.array(target_evec_2)
    target_evec_3 = np.array(target_evec_3)

    pred_evals = np.array(pred_evals)
    #pred_evecs = np.array(pred_evecs)

    # print("target evals ", target_evals)
    # print("pred evals ", pred_evals)
    #
    # print("target evals[<=0] ", target_evals[target_evals <= 0])
    # print("pred evals[<=0] ", pred_evals[pred_evals <= 0])

    pred_evec_1 = np.array(pred_evec_1)
    pred_evec_2 = np.array(pred_evec_2)
    pred_evec_3 = np.array(pred_evec_3)

    # Calculate MSE for eigenvalues
    # evals_mse = torch.mean((pred_evals - true_evals) ** 2)

    squared_err_k = []
    std_tar_k = []
    r2_k = []
    mse_k = []
    rmse_k = []
    nrmse_k = []

    for i in range(target_evals.shape[1]):
        targets = np.squeeze(target_evals[:, i, ...])
        predictions = np.squeeze(pred_evals[:, i, ...])
        squared_err_k.append((targets - predictions) ** 2)
        std_tar_k.append(np.std(targets))
        r2_k.append(1 - (np.sum(squared_err_k[i]) /
                         np.sum((targets - np.mean(targets)) ** 2)))
        mse_k.append(np.mean(squared_err_k[i]))
        rmse_k.append(np.sqrt(mse_k[i]))
        nrmse_k.append(rmse_k[i] / std_tar_k[i])

    squared_err_evec_1 = []
    std_tar_evec_1 = []
    r2_evec_1 = []
    mse_evec_1 = []
    rmse_evec_1 = []
    nrmse_evec_1 = []

    # print("target evec 1 shape ", target_evec_1.shape)
    # print("pred evec 1 shape ", pred_evec_1.shape)
    #
    # print("target evec 1 ", target_evec_1)
    # print("pred evec 1 ", pred_evec_1)
    #
    # print("pred evec 2", pred_evec_2)
    # print("pred evec 3 ", pred_evec_3)


    for i in range(target_evec_1.shape[1]):
        targets_evec_1 = np.squeeze(target_evec_1[:, i, ...])
        predictions_evec_1 = np.squeeze(pred_evec_1[:, i, ...])

        # print("targets_evec_1 ", targets_evec_1.shape)
        # print("predictions_evec_1 ", predictions_evec_1.shape)

        squared_err_evec_1.append((targets_evec_1 - predictions_evec_1) ** 2)

        # print("squared err evec 1 ", squared_err_evec_1[i])
        # print("(targets_evec_1 - np.mean(targets_evec_1)) ** 2) ", (targets_evec_1 - np.mean(targets_evec_1)) ** 2)
        #
        # print("np.sum(squared_err_evec_1[i]) ", np.sum(squared_err_evec_1[i]))
        # print("np.sum((targets_evec_1 - np.mean(targets_evec_1)) ** 2) ", np.sum((targets_evec_1 - np.mean(targets_evec_1)) ** 2))

        std_tar_evec_1.append(np.std(targets_evec_1))
        r2_evec_1.append(1 - (np.sum(squared_err_evec_1[i]) /
                              np.sum((targets_evec_1 - np.mean(targets_evec_1)) ** 2)))
        mse_evec_1.append(np.mean(squared_err_evec_1[i]))
        rmse_evec_1.append(np.sqrt(mse_evec_1[i]))
        nrmse_evec_1.append(rmse_evec_1[i] / std_tar_evec_1[i])

    squared_err_evec_2 = []
    std_tar_evec_2 = []
    r2_evec_2 = []
    mse_evec_2 = []
    rmse_evec_2 = []
    nrmse_evec_2 = []

    for i in range(target_evec_2.shape[1]):
        targets_evec_2 = np.squeeze(target_evec_2[:, i, ...])
        predictions_evec_2 = np.squeeze(pred_evec_2[:, i, ...])
        squared_err_evec_2.append((targets_evec_2 - predictions_evec_2) ** 2)
        std_tar_evec_2.append(np.std(targets_evec_2))
        r2_evec_2.append(1 - (np.sum(squared_err_evec_2[i]) /
                              np.sum((targets_evec_2 - np.mean(targets_evec_2)) ** 2)))
        mse_evec_2.append(np.mean(squared_err_evec_2[i]))
        rmse_evec_2.append(np.sqrt(mse_evec_2[i]))
        nrmse_evec_2.append(rmse_evec_2[i] / std_tar_evec_2[i])

    squared_err_evec_3 = []
    std_tar_evec_3 = []
    r2_evec_3 = []
    mse_evec_3 = []
    rmse_evec_3 = []
    nrmse_evec_3 = []

    for i in range(target_evec_3.shape[1]):
        targets_evec_3 = np.squeeze(target_evec_3[:, i, ...])
        predictions_evec_3 = np.squeeze(pred_evec_3[:, i, ...])
        squared_err_evec_3.append((targets_evec_3 - predictions_evec_3) ** 2)
        std_tar_evec_3.append(np.std(targets_evec_3))
        r2_evec_3.append(1 - (np.sum(squared_err_evec_3[i]) /
                              np.sum((targets_evec_3 - np.mean(targets_evec_3)) ** 2)))
        mse_evec_3.append(np.mean(squared_err_evec_3[i]))
        rmse_evec_3.append(np.sqrt(mse_evec_3[i]))
        nrmse_evec_3.append(rmse_evec_3[i] / std_tar_evec_3[i])

    # targets = true_evec_2  # np.squeeze(targets_arr[:, i, ...])
    # predictions = pred_evec_2  # np.squeeze(predictions_arr[:, i, ...])
    # squared_err_evec_2 = (targets - predictions) ** 2
    # std_tar_evec_2 = np.std(targets)
    # r2_evec_2 = 1 - (np.sum(squared_err_evec_2) /
    #                  np.sum((targets - np.mean(targets)) ** 2))
    # mse_evec_2 = np.mean(squared_err_evec_2)
    # rmse_evec_2 = np.sqrt(mse_evec_2)
    # nrmse_evec_2 = rmse_evec_2 / std_tar_evec_2

    print("EVALS: mse: {}, r2: {}, rmse: {}, nrmse: {}".format(mse_k, r2_k, rmse_k, nrmse_k))
    print("EVEC_1: mse: {}, r2: {}, rmse: {}, nrmse: {}".format(mse_evec_1, r2_evec_1, rmse_evec_1, nrmse_evec_1))
    print("EVEC_2: mse: {}, r2: {}, rmse: {}, nrmse: {}".format(mse_evec_2, r2_evec_2, rmse_evec_2, nrmse_evec_2))
    print("EVEC_3: mse: {}, r2: {}, rmse: {}, nrmse: {}".format(mse_evec_3, r2_evec_3, rmse_evec_3, nrmse_evec_3))

    # return mse_k, rmse_k, nrmse_k, r2_k

def get_mse_nrmse_r2_eigh(targets, predictions):
    targets_arr = np.array(targets)
    predictions_arr = np.array(predictions)

    #eigenvalues, eigenvectors = calculate_eigen(targets)
    #print("eigenvalues ", eigenvalues)

    # print("targets arr shape", targets_arr.shape)
    # print("predictions arr shape ", predictions_arr.shape)

    y_pred_resized = np.empty((predictions_arr.shape[0], 2, 2))
    y_pred_resized[:, 0, 0] = predictions_arr[:, 0]
    y_pred_resized[:, 0, 1] = predictions_arr[:, 1]
    y_pred_resized[:, 1, 0] = predictions_arr[:, 1]
    y_pred_resized[:, 1, 1] = predictions_arr[:, 2]

    y_true_resized = np.empty((targets_arr.shape[0], 2, 2))
    y_true_resized[:, 0, 0] = targets_arr[:, 0]
    y_true_resized[:, 0, 1] = targets_arr[:, 1]
    y_true_resized[:, 1, 0] = targets_arr[:, 1]
    y_true_resized[:, 1, 1] = targets_arr[:, 2]

    # pred_evals = self._calc_evals(y_pred_resized)

    ####
    ## Eigenvalues of predicted tensors
    ####
    eval_1 = ((y_pred_resized[:, 0, 0] + y_pred_resized[:, 1, 1]) +
              np.sqrt((-y_pred_resized[:, 0, 0] - y_pred_resized[:, 1, 1]) ** 2 -
                         4 * (y_pred_resized[:, 0, 0] * y_pred_resized[:, 1, 1] -
                              y_pred_resized[:, 0, 1] * y_pred_resized[:, 1, 0]))) / 2

    eval_2 = ((y_pred_resized[:, 0, 0] + y_pred_resized[:, 1, 1]) -
              np.sqrt((-y_pred_resized[:, 0, 0] - y_pred_resized[:, 1, 1]) ** 2 -
                         4 * (y_pred_resized[:, 0, 0] * y_pred_resized[:, 1, 1] -
                              y_pred_resized[:, 0, 1] * y_pred_resized[:, 1, 0]))) / 2

    pred_evals = np.stack([eval_1, eval_2], axis=1)
    pred_eigenvalues, pred_eigenvectors = np.linalg.eigh(y_pred_resized)

    # true_evals = self._calc_evals(y_true_resized)

    ####
    ## Eigenvalues of true/target tensors
    ####
    eval_1 = ((y_true_resized[:, 0, 0] + y_true_resized[:, 1, 1]) +
              np.sqrt((-y_true_resized[:, 0, 0] - y_true_resized[:, 1, 1]) ** 2 -
                         4 * (y_true_resized[:, 0, 0] * y_true_resized[:, 1, 1] -
                              y_true_resized[:, 0, 1] * y_true_resized[:, 1, 0]))) / 2

    eval_2 = ((y_true_resized[:, 0, 0] + y_true_resized[:, 1, 1]) -
              np.sqrt((-y_true_resized[:, 0, 0] - y_true_resized[:, 1, 1]) ** 2 -
                         4 * (y_true_resized[:, 0, 0] * y_true_resized[:, 1, 1] -
                              y_true_resized[:, 0, 1] * y_true_resized[:, 1, 0]))) / 2

    true_evals = np.stack([eval_1, eval_2], axis=1)

    print("target eigenvalues ", true_evals)



    # print("pred evals ", pred_evals)
    # print("true evals ", true_evals)
    # print("pred_evals - true_evals ", pred_evals - true_evals)

    true_eigenvalues, true_eigenvectors = np.linalg.eigh(y_true_resized)

    # Calculate MSE for eigenvalues
    #evals_mse = torch.mean((pred_evals - true_evals) ** 2)

    squared_err_k = []
    std_tar_k = []
    r2_k = []
    mse_k = []
    rmse_k = []
    nrmse_k = []

    for i in range(true_evals.shape[1]):
        targets = np.squeeze(true_evals[:, i, ...])
        predictions = np.squeeze(pred_evals[:, i, ...])
        squared_err_k.append((targets - predictions) ** 2)
        std_tar_k.append(np.std(targets))
        r2_k.append(1 - (np.sum(squared_err_k[i]) /
                         np.sum((targets - np.mean(targets)) ** 2)))
        mse_k.append(np.mean(squared_err_k[i]))
        rmse_k.append(np.sqrt(mse_k[i]))
        nrmse_k.append(rmse_k[i] / std_tar_k[i])


    all_pred_evec_1 = []
    all_pred_evec_2 = []
    all_true_evec_1 = []
    all_true_evec_2 = []
    for i in range(len(pred_eigenvalues)):
        linalg_pred_evals = pred_eigenvalues[i]
        linalg_true_evals = true_eigenvalues[i]
        linalg_pred_evecs = pred_eigenvectors[i]
        linalg_true_evecs = true_eigenvectors[i]

        pred_evec_1 = linalg_pred_evecs[:, np.argmin(np.abs(linalg_pred_evals - pred_evals[i][0]))]
        pred_evec_2 = linalg_pred_evecs[:, np.argmin(np.abs(linalg_pred_evals - pred_evals[i][1]))]

        all_pred_evec_1.append(pred_evec_1)
        all_pred_evec_2.append(pred_evec_2)

        true_evec_1 = linalg_true_evecs[:, np.argmin(np.abs(linalg_true_evals - true_evals[i][0]))]
        true_evec_2 = linalg_true_evecs[:, np.argmin(np.abs(linalg_true_evals - true_evals[i][1]))]

        all_true_evec_1.append(true_evec_1)
        all_true_evec_2.append(true_evec_2)


    all_true_evec_1 = np.array(all_true_evec_1)
    all_true_evec_2 = np.array(all_true_evec_2)
    all_pred_evec_1 = np.array(all_pred_evec_1)
    all_pred_evec_2 = np.array(all_pred_evec_2)


    # squared_err_k = []
    # std_tar_k = []
    # r2_k = []
    # mse_k = []
    # rmse_k = []
    # nrmse_k = []

    # print("all true evec 1 shape", all_true_evec_1.shape)
    # print("all true evec 2 shape", all_true_evec_2.shape)

    # targets = true_evec_1 #np.squeeze(targets_arr[:, i, ...])
    # predictions = pred_evec_1 #np.squeeze(predictions_arr[:, i, ...])
    # squared_err_evec_1 = (targets - predictions) ** 2
    # std_tar_evec_1 = np.std(targets)
    # r2_evec_1 = 1 - (np.sum(squared_err_evec_1) /
    #                  np.sum((targets - np.mean(targets)) ** 2))
    # mse_evec_1 = np.mean(squared_err_evec_1)
    # rmse_evec_1 = np.sqrt(mse_evec_1)
    # nrmse_evec_1 = rmse_evec_1 / std_tar_evec_1


    squared_err_evec_1 = []
    std_tar_evec_1 = []
    r2_evec_1 = []
    mse_evec_1 = []
    rmse_evec_1 = []
    nrmse_evec_1 = []

    for i in range(all_true_evec_1.shape[1]):
        targets_evec_1 = np.squeeze(all_true_evec_1[:, i, ...])
        predictions_evec_1 = np.squeeze(all_pred_evec_1[:, i, ...])
        squared_err_evec_1.append((targets_evec_1 - predictions_evec_1) ** 2)
        std_tar_evec_1.append(np.std(targets_evec_1))
        r2_evec_1.append(1 - (np.sum(squared_err_evec_1[i]) /
                         np.sum((targets_evec_1 - np.mean(targets_evec_1)) ** 2)))
        mse_evec_1.append(np.mean(squared_err_evec_1[i]))
        rmse_evec_1.append(np.sqrt(mse_evec_1[i]))
        nrmse_evec_1.append(rmse_evec_1[i] / std_tar_evec_1[i])

    squared_err_evec_2 = []
    std_tar_evec_2 = []
    r2_evec_2 = []
    mse_evec_2 = []
    rmse_evec_2 = []
    nrmse_evec_2 = []

    for i in range(all_true_evec_2.shape[1]):
        targets_evec_2 = np.squeeze(all_true_evec_2[:, i, ...])
        predictions_evec_2 = np.squeeze(all_pred_evec_2[:, i, ...])
        squared_err_evec_2.append((targets_evec_2 - predictions_evec_2) ** 2)
        std_tar_evec_2.append(np.std(targets_evec_2))
        r2_evec_2.append(1 - (np.sum(squared_err_evec_2[i]) /
                              np.sum((targets_evec_2 - np.mean(targets_evec_2)) ** 2)))
        mse_evec_2.append(np.mean(squared_err_evec_2[i]))
        rmse_evec_2.append(np.sqrt(mse_evec_2[i]))
        nrmse_evec_2.append(rmse_evec_2[i] / std_tar_evec_2[i])


    # targets = true_evec_2  # np.squeeze(targets_arr[:, i, ...])
    # predictions = pred_evec_2  # np.squeeze(predictions_arr[:, i, ...])
    # squared_err_evec_2 = (targets - predictions) ** 2
    # std_tar_evec_2 = np.std(targets)
    # r2_evec_2 = 1 - (np.sum(squared_err_evec_2) /
    #                  np.sum((targets - np.mean(targets)) ** 2))
    # mse_evec_2 = np.mean(squared_err_evec_2)
    # rmse_evec_2 = np.sqrt(mse_evec_2)
    # nrmse_evec_2 = rmse_evec_2 / std_tar_evec_2


    print("EVALS: mse: {}, r2: {}, rmse: {}, nrmse: {}".format(mse_k, r2_k, rmse_k, nrmse_k))
    print("EVEC_1: mse: {}, r2: {}, rmse: {}, nrmse: {}".format(mse_evec_1, r2_evec_1, rmse_evec_1, nrmse_evec_1))
    print("EVEC_2: mse: {}, r2: {}, rmse: {}, nrmse: {}".format(mse_evec_2, r2_evec_2, rmse_evec_2, nrmse_evec_2))

    #return mse_k, rmse_k, nrmse_k, r2_k


class FrobeniusNorm(nn.Module):
    def __init__(self):
        super(FrobeniusNorm, self).__init__()

    def forward(self, y_pred, y_true):
        k_xx = y_pred[:, 0, ...] - y_true[:, 0, ...]
        k_xy = y_pred[:, 1, ...] - y_true[:, 1, ...]
        k_yy = y_pred[:, 2, ...] - y_true[:, 2, ...]

        #k_xx = torch.mean(k_xx ** 2)
        #k_xy = torch.mean(k_xy ** 2)
        #k_yy = torch.mean(k_yy ** 2)

        return torch.mean(torch.sqrt(k_xx**2 + 2 * k_xy**2 + k_yy**2))

class FrobeniusNorm2(nn.Module):
    def __init__(self):
        super(FrobeniusNorm2, self).__init__()

    def forward(self, y_pred, y_true):
        k_xx = y_pred[:, 0, ...] - y_true[:, 0, ...]
        k_xy = y_pred[:, 1, ...] - y_true[:, 1, ...]
        k_yy = y_pred[:, 2, ...] - y_true[:, 2, ...]

        k_xx = torch.mean(k_xx ** 2)
        k_xy = torch.mean(k_xy ** 2)
        k_yy = torch.mean(k_yy ** 2)

        return torch.sqrt(k_xx**2 + 2 * k_xy**2 + k_yy**2)


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
        if torch.cuda.is_available():
            self.weights = self.weights.cuda()

    def forward(self, y_pred, y_true):
        mse_loss = F.mse_loss(y_pred, y_true, reduction='none')
        if str(mse_loss.device) == "cpu":
            self.weights = self.weights.cpu()
        weighted_mse_loss = torch.mean(self.weights * mse_loss)
        return weighted_mse_loss


class MSELossLargeEmph(nn.Module):
    def __init__(self, params):
        super(MSELossLargeEmph, self).__init__()
        #print("params ", params)
        self.weights_min_abs = torch.Tensor(params[0])
        self.mult_coef = params[1]
        self.fce = params[2]
        if torch.cuda.is_available():
            self.weights_min_abs = self.weights_min_abs.cuda()
            #self.mult_coef = self.mult_coef.cuda()

        #print("weights min abs ", self.weights_min_abs)
        #print("self.mult_coef ", self.mult_coef)

    def forward(self, y_pred, y_true):
        if str(y_true.device) == "cpu":
            self.weights_min_abs = self.weights_min_abs.cpu()

        error = y_true - y_pred
        #print("y_true ", len(y_true.shape))
        weights = (y_true + self.weights_min_abs) * self.mult_coef + 1

        if self.fce == "2":
            weights = weights ** 2
        elif self.fce == "3":
            weights = weights ** 3
        elif self.fce == "4":
            weights = weights ** 4
        elif self.fce == "exp":
            weights = torch.exp(weights)
        elif self.fce == "exp_2":
            weights = torch.exp(weights)**2

        if len(y_true.shape) == 1:
            weights[1] = 1
        else:
            weights[:, 1] = 1

        weighted_error = torch.square(error) * weights
        #print("weighted error ", weighted_error)
        #print("y_true: {}, squared error: {}, weighted error: {}".format(y_true, torch.square(error), weighted_error))
        return torch.mean(weighted_error)


class MSELossLargeEmphAvg(nn.Module):
    def __init__(self, params):
        super(MSELossLargeEmphAvg, self).__init__()
        #print("params ", params)
        self.weights_min_abs = torch.Tensor(params[0])
        self.mult_coef =  params[1]
        self.fce = params[2]
        if torch.cuda.is_available():
            self.weights_min_abs = self.weights_min_abs.cuda()
            #self.mult_coef = self.mult_coef.cuda()

        # print("weights min abs ", self.weights_min_abs)
        # print("self.mult_coef ", self.mult_coef)

    def forward(self, y_pred, y_true):
        if str(y_true.device) == "cpu":
            self.weights_min_abs = self.weights_min_abs.cpu()

        error = y_true - y_pred
        #print("y_true ", y_true)
        weights = (y_true + self.weights_min_abs) * self.mult_coef + 1

        if self.fce == "2":
            weights = weights ** 2
        elif self.fce == "3":
            weights = weights ** 3
        elif self.fce == "4":
            weights = weights ** 4
        elif self.fce == "exp":
            weights = torch.exp(weights)
        elif self.fce == "exp_2":
            weights = torch.exp(weights)**2

        if len(y_true.shape) == 1:
            weights[1] = (weights[0] + weights[1])/2
        else:
            weights[:, 1] = weights[:, [0, 2]].mean(dim=1)

        #print("wieghts ", weights)
        weighted_error = torch.square(error) * weights
        #print("weighted error ", weighted_error)
        #print("y_true: {}, squared error: {}, weighted error: {}".format(y_true, torch.square(error), weighted_error))
        return torch.mean(weighted_error)


class WeightedMSELossSum(nn.Module):
    def __init__(self, weights):
        super(WeightedMSELossSum, self).__init__()
        self.weights = torch.Tensor(weights)
        if torch.cuda.is_available():
            self.weights = self.weights.cuda()

    def forward(self, y_pred, y_true):
        mse_loss = F.mse_loss(y_pred, y_true, reduction='none')
        if str(mse_loss.device) == "cpu":
            self.weights = self.weights.cpu()

        #print("mse loss ", len(mse_loss.shape))

        if len(mse_loss.shape) == 1:
            mse_loss_sum = self.weights * mse_loss
        else:
            mse_loss_sum = torch.sum(self.weights * mse_loss, dim=1)
        #print("mse loss sum ", mse_loss_sum)
        weighted_mse_loss = torch.mean(mse_loss_sum)
        return weighted_mse_loss


class RelMSELoss(nn.Module):
    def __init__(self, weights):
        super(RelMSELoss, self).__init__()
        self.weights = torch.Tensor(weights)
        if torch.cuda.is_available():
            self.weights = self.weights.cuda()

    def forward(self, y_pred, y_true):
        # if len(y_true.shape) == 1:
        #     y_pred = y_pred.unsqueeze(0)
        #     y_true = y_true.unsqueeze(0)

        # k_xx_mse = F.mse_loss(y_pred[:, 0, ...], y_true[:, 0, ...])
        # print("k xx mse ", k_xx_mse)
        # k_xx_rmse = torch.sqrt(k_xx_mse / (y_true[:, 0, ...] ** 2))
        #
        # k_xy_mse = F.mse_loss(y_pred[:, 1, ...], y_true[:, 1, ...])
        # k_xy_rmse = torch.sqrt(k_xx_mse / (y_true[:, 1, ...] ** 2))
        #
        # k_yy_mse = F.mse_loss(y_pred[:, 2, ...], y_true[:, 2, ...])
        # k_yy_rmse = torch.sqrt(k_xx_mse / (y_true[:, 2, ...] ** 2))
        #print("y_pred[:, 0, ...] ", y_pred[:, 0, ...])
        #print("y_true[:, 0, ...] ", y_true[:, 0, ...])
        #
        #print("(y_pred[:, 0, ...] - y_true[:, 0, ...]) ", (y_pred[:, 0, ...] - y_true[:, 0, ...]))


        k_xx = torch.square((y_pred[:, 0, ...] - y_true[:, 0, ...]))/torch.square(y_true[:, 0, ...])
        k_xy = torch.square(y_pred[:, 1, ...] - y_true[:, 1, ...])/torch.square(y_true[:, 1, ...])
        k_yy = torch.square(y_pred[:, 2, ...] - y_true[:, 2, ...])/torch.square(y_true[:, 2, ...])

        #print("torch.square((y_pred[:, 0, ...] - y_true[:, 0, ...])) ", torch.square((y_pred[:, 0, ...] - y_true[:, 0, ...])))
        #print("torch.square(y_true[:, 0, ...]) ", torch.square(y_true[:, 0, ...]))

        #print("k xx ", k_xx)
        # print("k xy ", k_xy)
        # print("k yy ", k_yy)

        rel_mse = torch.mean(k_xx) + torch.mean(k_xy) + torch.mean(k_yy)#torch.mean(k_xx ** 2 + k_xy ** 2 + k_yy ** 2)
        #total_loss = k_xx_rmse + k_xy_rmse + k_yy_rmse
        print("RelMSE: {},  k_xx**2: {},  k_xy**2: {},  k_yy**2: {}".format(rel_mse, torch.mean(k_xx), torch.mean(k_xy), torch.mean(k_yy)))
        #print("total loss ", total_loss)
        #exit()

        return rel_mse


class MASELoss(nn.Module):
    def __init__(self, weights):
        super(MASELoss, self).__init__()
        self.weights = torch.Tensor(weights)
        if torch.cuda.is_available():
            self.weights = self.weights.cuda()

    def forward(self, y_pred, y_true):
        # if len(y_true.shape) == 1:
        #     y_pred = y_pred.unsqueeze(0)
        #     y_true = y_true.unsqueeze(0)

        numerator = torch.abs(y_true[:, 0, ...] - y_pred[:, 0, ...]).mean()
        denominator = torch.abs(y_true[1:, 0, ...] - y_true[:-1, 0, ...]).mean()
        #print("numerator kxx", numerator)
        #print("denomination kxx", denominator)

        # Calculate MASE
        mase_k_xx = numerator / denominator

        numerator = torch.abs(y_true[:, 1, ...] - y_pred[:, 1, ...]).mean()
        denominator = torch.abs(y_true[1:, 1, ...] - y_true[:-1, 1, ...]).mean()
        #print("numerator kxy", numerator)
        #print("denomination kxy ", denominator)

        # Calculate MASE
        mase_k_xy = numerator / denominator

        numerator = torch.abs(y_true[:, 2, ...] - y_pred[:, 2, ...]).mean()
        denominator = torch.abs(y_true[1:, 2, ...] - y_true[:-1, 2, ...]).mean()
        #print("numerator kyy", numerator)
        #print("denomination kyy ", denominator)

        # Calculate MASE
        mase_k_yy = numerator / denominator

        #print("mase k xx ", mase_k_xx)
        #print("mase k xy ", mase_k_xy)
        #print("mase k yy ", mase_k_yy)

        total_mase = mase_k_xx + mase_k_xy + mase_k_yy

        print("MASE: {},  k_xx**2: {},  k_xy**2: {},  k_yy**2: {}".format(total_mase, mase_k_xx, mase_k_xy, mase_k_yy))

        return total_mase


class RelMSELoss2(nn.Module):
    def __init__(self, weights):
        super(RelMSELoss2, self).__init__()
        self.weights = torch.Tensor(weights)
        if torch.cuda.is_available():
            self.weights = self.weights.cuda()

    def forward(self, y_pred, y_true):
        # if len(y_true.shape) == 1:
        #     y_pred = y_pred.unsqueeze(0)
        #     y_true = y_true.unsqueeze(0)

        # k_xx_mse = F.mse_loss(y_pred[:, 0, ...], y_true[:, 0, ...])
        # print("k xx mse ", k_xx_mse)
        # k_xx_rmse = torch.sqrt(k_xx_mse / (y_true[:, 0, ...] ** 2))
        #
        # k_xy_mse = F.mse_loss(y_pred[:, 1, ...], y_true[:, 1, ...])
        # k_xy_rmse = torch.sqrt(k_xx_mse / (y_true[:, 1, ...] ** 2))
        #
        # k_yy_mse = F.mse_loss(y_pred[:, 2, ...], y_true[:, 2, ...])
        # k_yy_rmse = torch.sqrt(k_xx_mse / (y_true[:, 2, ...] ** 2))
        print("y_pred[:, 0, ...] ", y_pred[:, 0, ...])
        print("y_true[:, 0, ...] ", y_true[:, 0, ...])
        print("(y_pred[:, 0, ...]/y_true[:, 0, ...]) ", (y_pred[:, 0, ...] /y_true[:, 0, ...]))
        print("(y_pred[:, 0, ...]/y_true[:, 0, ...]) -1  ", (y_pred[:, 0, ...] / y_true[:, 0, ...])-1)

        k_xx = torch.square((y_pred[:, 0, ...]/y_true[:, 0, ...]) - 1)
        k_xy = torch.square((y_pred[:, 1, ...]/y_true[:, 1, ...]) - 1)
        k_yy = torch.square((y_pred[:, 2, ...]/y_true[:, 2, ...]) - 1)

        print("torch.square((y_pred[:, 0, ...]/y_true[:, 0, ...]) -1 ) ", torch.square((y_pred[:, 0, ...]/y_true[:, 0, ...]) - 1))
        #print("torch.square(y_true[:, 0, ...]) ", torch.square(y_true[:, 0, ...]))

        print("k xx ", k_xx)
        # print("k xy ", k_xy)
        # print("k yy ", k_yy)

        rel_mse = torch.mean(k_xx) + torch.mean(k_xy) + torch.mean(k_yy)#torch.mean(k_xx ** 2 + k_xy ** 2 + k_yy ** 2)
        #total_loss = k_xx_rmse + k_xy_rmse + k_yy_rmse
        print("RelMSE: {},  k_xx**2: {},  k_xy**2: {},  k_yy**2: {}".format(rel_mse, torch.mean(k_xx), torch.mean(k_xy), torch.mean(k_yy)))
        #print("total loss ", total_loss)
        #exit()

        return rel_mse


class RelXYMSELoss(nn.Module):
    def __init__(self, weights):
        super(RelXYMSELoss, self).__init__()
        print("weights ", weights)
        self.weights = torch.Tensor(weights)
        if torch.cuda.is_available():
            self.weights = self.weights.cuda()

    def forward(self, y_pred, y_true):

        if len(y_true.shape) == 1:
            y_pred = y_pred.unsqueeze(0)
            y_true = y_true.unsqueeze(0)

        k_xx = y_pred[:, 0, ...] - y_true[:, 0, ...]
        k_xy = ((y_pred[:, 1, ...] - y_true[:, 1, ...]))/y_true[:, 1, ...]
        k_yy = y_pred[:, 2, ...] - y_true[:, 2, ...]

        rel_mse = torch.mean(k_xx ** 2 + k_xy ** 2 + k_yy ** 2)

        print("RelMSE: {},  k_xx**2: {},  k_xy**2: {},  k_yy**2: {}".format(rel_mse, torch.mean(k_xx ** 2),
                                                                            torch.mean(k_xy ** 2),
                                                                            torch.mean(k_yy ** 2)))

        return rel_mse


class EighMSE(nn.Module):
    def __init__(self, weights):
        super(EighMSE, self).__init__()
        print("weights ", weights)
        self.weights = torch.Tensor(weights)
        if torch.cuda.is_available():
            self.weights = self.weights.cuda()

    def _calc_evals(self, batch_data):
        eval_1 = ((batch_data[:, 0, 0] + batch_data[:, 1, 1]) +
                       torch.sqrt((-batch_data[:, 0, 0] - batch_data[:, 1, 1]) ** 2 -
                                  4 * (batch_data[:, 0, 0] * batch_data[:, 1, 1] -
                                       batch_data[:, 0, 1] * batch_data[:, 1, 0]))) / 2

        eval_2 = ((batch_data[:, 0, 0] + batch_data[:, 1, 1]) -
                       torch.sqrt((-batch_data[:, 0, 0] - batch_data[:, 1, 1]) ** 2 -
                                  4 * (batch_data[:, 0, 0] * batch_data[:, 1, 1] -
                                       batch_data[:, 0, 1] * batch_data[:, 1, 0]))) / 2

        evals = torch.stack([eval_1, eval_2], axis=1)

        return evals

    def forward(self, y_pred, y_true):
        if len(y_true.shape) == 1:
            y_pred = y_pred.unsqueeze(0)
            y_true = y_true.unsqueeze(0)

        #tensor_loss = F.mse_loss(y_pred, y_true)

        y_pred_resized = torch.empty(y_pred.shape[0], 2, 2)
        y_pred_resized[:, 0, 0] = y_pred[:, 0]
        y_pred_resized[:, 0, 1] = y_pred[:, 1]
        y_pred_resized[:, 1, 0] = y_pred[:, 1]
        y_pred_resized[:, 1, 1] = y_pred[:, 2]

        y_true_resized = torch.empty(y_true.shape[0], 2, 2)
        y_true_resized[:, 0, 0] = y_true[:, 0]
        y_true_resized[:, 0, 1] = y_true[:, 1]
        y_true_resized[:, 1, 0] = y_true[:, 1]
        y_true_resized[:, 1, 1] = y_true[:, 2]

        #pred_evals = self._calc_evals(y_pred_resized)

        ####
        ## Eigenvalues of predicted tensors
        ####
        eval_1 = ((y_pred_resized[:, 0, 0] + y_pred_resized[:, 1, 1]) +
                  torch.sqrt((-y_pred_resized[:, 0, 0] - y_pred_resized[:, 1, 1]) ** 2 -
                             4 * (y_pred_resized[:, 0, 0] * y_pred_resized[:, 1, 1] -
                                  y_pred_resized[:, 0, 1] * y_pred_resized[:, 1, 0]))) / 2

        eval_2 = ((y_pred_resized[:, 0, 0] + y_pred_resized[:, 1, 1]) -
                  torch.sqrt((-y_pred_resized[:, 0, 0] - y_pred_resized[:, 1, 1]) ** 2 -
                             4 * (y_pred_resized[:, 0, 0] * y_pred_resized[:, 1, 1] -
                                  y_pred_resized[:, 0, 1] * y_pred_resized[:, 1, 0]))) / 2

        pred_evals = torch.stack([eval_1, eval_2], axis=1)
        pred_eigenvalues, pred_eigenvectors = torch.linalg.eigh(y_pred_resized)

        #true_evals = self._calc_evals(y_true_resized)

        ####
        ## Eigenvalues of true/target tensors
        ####
        eval_1 = ((y_true_resized[:, 0, 0] + y_true_resized[:, 1, 1]) +
                  torch.sqrt((-y_true_resized[:, 0, 0] - y_true_resized[:, 1, 1]) ** 2 -
                             4 * (y_true_resized[:, 0, 0] * y_true_resized[:, 1, 1] -
                                  y_true_resized[:, 0, 1] * y_true_resized[:, 1, 0]))) / 2

        eval_2 = ((y_true_resized[:, 0, 0] + y_true_resized[:, 1, 1]) -
                  torch.sqrt((-y_true_resized[:, 0, 0] - y_true_resized[:, 1, 1]) ** 2 -
                             4 * (y_true_resized[:, 0, 0] * y_true_resized[:, 1, 1] -
                                  y_true_resized[:, 0, 1] * y_true_resized[:, 1, 0]))) / 2

        true_evals = torch.stack([eval_1, eval_2], axis=1)

        # print("pred evals ", pred_evals)
        # print("true evals ", true_evals)
        # print("pred_evals - true_evals ", pred_evals - true_evals)

        true_eigenvalues, true_eigenvectors = torch.linalg.eigh(y_true_resized)

        # Calculate MSE for eigenvalues
        evals_mse = torch.mean((pred_evals - true_evals) ** 2)

        mse_evec_1 = 0
        mse_evec_2 = 0
        if self.weights[1] == 0 and self.weights[1] == 0:
            mse = self.weights[0] * evals_mse
            print("MSE evals: {}".format(mse))
            #print("MSE evals: {}, tensor MSE: {}".format(mse, tensor_loss))
        else:
            for i in range(len(pred_eigenvalues)):
                linalg_pred_evals = pred_eigenvalues[i]
                linalg_true_evals = true_eigenvalues[i]
                linalg_pred_evecs = pred_eigenvectors[i]
                linalg_true_evecs = true_eigenvectors[i]

                pred_evec_1 = linalg_pred_evecs[:, torch.argmin(torch.abs(linalg_pred_evals - pred_evals[i][0]))]
                pred_evec_2 = linalg_pred_evecs[:, torch.argmin(torch.abs(linalg_pred_evals - pred_evals[i][1]))]

                true_evec_1 = linalg_true_evecs[:, torch.argmin(torch.abs(linalg_true_evals - true_evals[i][0]))]
                true_evec_2 = linalg_true_evecs[:, torch.argmin(torch.abs(linalg_true_evals - true_evals[i][1]))]

                # Calculate MSE for eigenvectors
                mse_evec_1 += torch.mean((pred_evec_1 - true_evec_1) ** 2)
                mse_evec_2 += torch.mean((pred_evec_2 - true_evec_2) ** 2)

            total_evec_mse = self.weights[1] * mse_evec_1 + self.weights[2] * mse_evec_2
            mse = self.weights[0] * evals_mse + (total_evec_mse / (i+1))

            print("MSE evals: {}, evec1: {}, evec2: {}, total evec mse / batch size : {}".format(evals_mse, mse_evec_1,
                                                                                                 mse_evec_2,
                                                                                                 total_evec_mse / (i+1)))

        return mse



class EighMSE_2(nn.Module):
    def __init__(self, weights):
        super(EighMSE_2, self).__init__()
        print("weights ", weights)
        self.weights = torch.Tensor(weights)
        if torch.cuda.is_available():
            self.weights = self.weights.cuda()

    # def _calc_evals(self, batch_data):
    #     eval_1 = ((batch_data[:, 0, 0] + batch_data[:, 1, 1]) +
    #                    torch.sqrt((-batch_data[:, 0, 0] - batch_data[:, 1, 1]) ** 2 -
    #                               4 * (batch_data[:, 0, 0] * batch_data[:, 1, 1] -
    #                                    batch_data[:, 0, 1] * batch_data[:, 1, 0]))) / 2
    #
    #     eval_2 = ((batch_data[:, 0, 0] + batch_data[:, 1, 1]) -
    #                    torch.sqrt((-batch_data[:, 0, 0] - batch_data[:, 1, 1]) ** 2 -
    #                               4 * (batch_data[:, 0, 0] * batch_data[:, 1, 1] -
    #                                    batch_data[:, 0, 1] * batch_data[:, 1, 0]))) / 2
    #
    #     evals = torch.stack([eval_1, eval_2], axis=1)
    #
    #     return evals

    def forward(self, y_pred, y_true):
        if len(y_true.shape) == 1:
            y_pred = y_pred.unsqueeze(0)
            y_true = y_true.unsqueeze(0)

        #tensor_loss = F.mse_loss(y_pred, y_true)

        y_pred_resized = torch.empty(y_pred.shape[0], 2, 2)
        y_pred_resized[:, 0, 0] = y_pred[:, 0]
        y_pred_resized[:, 0, 1] = y_pred[:, 1]
        y_pred_resized[:, 1, 0] = y_pred[:, 1]
        y_pred_resized[:, 1, 1] = y_pred[:, 2]

        y_true_resized = torch.empty(y_true.shape[0], 2, 2)
        y_true_resized[:, 0, 0] = y_true[:, 0]
        y_true_resized[:, 0, 1] = y_true[:, 1]
        y_true_resized[:, 1, 0] = y_true[:, 1]
        y_true_resized[:, 1, 1] = y_true[:, 2]

        #pred_evals = self._calc_evals(y_pred_resized)

        ####
        ## Eigenvalues of predicted tensors
        ####
        eval_1 = ((y_pred_resized[:, 0, 0] + y_pred_resized[:, 1, 1]) +
                  torch.sqrt((-y_pred_resized[:, 0, 0] - y_pred_resized[:, 1, 1]) ** 2 -
                             4 * (y_pred_resized[:, 0, 0] * y_pred_resized[:, 1, 1] -
                                  y_pred_resized[:, 0, 1] * y_pred_resized[:, 1, 0]))) / 2

        eval_2 = ((y_pred_resized[:, 0, 0] + y_pred_resized[:, 1, 1]) -
                  torch.sqrt((-y_pred_resized[:, 0, 0] - y_pred_resized[:, 1, 1]) ** 2 -
                             4 * (y_pred_resized[:, 0, 0] * y_pred_resized[:, 1, 1] -
                                  y_pred_resized[:, 0, 1] * y_pred_resized[:, 1, 0]))) / 2

        pred_evals = torch.stack([eval_1, eval_2], axis=1)
        pred_eigenvalues, pred_eigenvectors = torch.linalg.eigh(y_pred_resized)

        #true_evals = self._calc_evals(y_true_resized)

        ####
        ## Eigenvalues of true/target tensors
        ####
        eval_1 = ((y_true_resized[:, 0, 0] + y_true_resized[:, 1, 1]) +
                  torch.sqrt((-y_true_resized[:, 0, 0] - y_true_resized[:, 1, 1]) ** 2 -
                             4 * (y_true_resized[:, 0, 0] * y_true_resized[:, 1, 1] -
                                  y_true_resized[:, 0, 1] * y_true_resized[:, 1, 0]))) / 2

        eval_2 = ((y_true_resized[:, 0, 0] + y_true_resized[:, 1, 1]) -
                  torch.sqrt((-y_true_resized[:, 0, 0] - y_true_resized[:, 1, 1]) ** 2 -
                             4 * (y_true_resized[:, 0, 0] * y_true_resized[:, 1, 1] -
                                  y_true_resized[:, 0, 1] * y_true_resized[:, 1, 0]))) / 2

        true_evals = torch.stack([eval_1, eval_2], axis=1)

        # print("pred evals ", pred_evals)
        # print("true evals ", true_evals)
        # print("pred_evals - true_evals ", pred_evals - true_evals)

        true_eigenvalues, true_eigenvectors = torch.linalg.eigh(y_true_resized)

        # Calculate MSE for eigenvalues
        evals_mse = torch.mean((pred_evals - true_evals) ** 2)

        mse_evec_1_0 = 0
        mse_evec_1_1 = 0
        mse_evec_2_0 = 0
        mse_evec_2_1 = 0
        if self.weights[1] == 0 and self.weights[1] == 0:
            mse = self.weights[0] * evals_mse
            print("MSE evals: {}".format(mse))
            #print("MSE evals: {}, tensor MSE: {}".format(mse, tensor_loss))
        else:
            all_pred_evec_1 = torch.empty(len(pred_eigenvalues), 2)
            all_pred_evec_2 = torch.empty(len(pred_eigenvalues), 2)
            all_true_evec_1 = torch.empty(len(pred_eigenvalues), 2)
            all_true_evec_2 = torch.empty(len(pred_eigenvalues), 2)
            for i in range(len(pred_eigenvalues)):
                linalg_pred_evals = pred_eigenvalues[i]
                linalg_true_evals = true_eigenvalues[i]
                linalg_pred_evecs = pred_eigenvectors[i]
                linalg_true_evecs = true_eigenvectors[i]

                pred_evec_1 = linalg_pred_evecs[:, torch.argmin(torch.abs(linalg_pred_evals - pred_evals[i][0]))]
                pred_evec_2 = linalg_pred_evecs[:, torch.argmin(torch.abs(linalg_pred_evals - pred_evals[i][1]))]

                all_pred_evec_1[i] = pred_evec_1
                all_pred_evec_2[i] = pred_evec_2

                true_evec_1 = linalg_true_evecs[:, torch.argmin(torch.abs(linalg_true_evals - true_evals[i][0]))]
                true_evec_2 = linalg_true_evecs[:, torch.argmin(torch.abs(linalg_true_evals - true_evals[i][1]))]

                all_true_evec_1[i] = true_evec_1
                all_true_evec_2[i] = true_evec_2

            mse_evec_1_0 = torch.mean((all_pred_evec_1[:, 0] - all_true_evec_1[:, 0]) ** 2)
            mse_evec_1_1 = torch.mean((all_pred_evec_1[:, 1] - all_true_evec_1[:, 1]) ** 2)

            mse_evec_2_0 = torch.mean((all_pred_evec_2[:, 0] - all_true_evec_2[:, 0]) ** 2)
            mse_evec_2_1 = torch.mean((all_pred_evec_2[:, 1] - all_true_evec_2[:, 1]) ** 2)


            total_evec_mse = self.weights[1] * mse_evec_1_0 + self.weights[2] * mse_evec_1_1 + self.weights[3] * mse_evec_2_0 + self.weights[4] * mse_evec_2_1
            mse = self.weights[0] * evals_mse + total_evec_mse

            print("MSE evals: {}, evec_1_0: {}, evec_1_1: {},  evec_1_0: {}, evec_1_1: {}".format(evals_mse, mse_evec_1_0,mse_evec_1_1,
                                                                                                 mse_evec_2_0, mse_evec_2_1,
                                                                                                 ))

        return mse


class EighMSE_2_MSE(nn.Module):
    def __init__(self, weights):
        super(EighMSE_2_MSE, self).__init__()
        print("weights ", weights)
        self.weights = torch.Tensor(weights)
        if torch.cuda.is_available():
            self.weights = self.weights.cuda()

    # def _calc_evals(self, batch_data):
    #     eval_1 = ((batch_data[:, 0, 0] + batch_data[:, 1, 1]) +
    #                    torch.sqrt((-batch_data[:, 0, 0] - batch_data[:, 1, 1]) ** 2 -
    #                               4 * (batch_data[:, 0, 0] * batch_data[:, 1, 1] -
    #                                    batch_data[:, 0, 1] * batch_data[:, 1, 0]))) / 2
    #
    #     eval_2 = ((batch_data[:, 0, 0] + batch_data[:, 1, 1]) -
    #                    torch.sqrt((-batch_data[:, 0, 0] - batch_data[:, 1, 1]) ** 2 -
    #                               4 * (batch_data[:, 0, 0] * batch_data[:, 1, 1] -
    #                                    batch_data[:, 0, 1] * batch_data[:, 1, 0]))) / 2
    #
    #     evals = torch.stack([eval_1, eval_2], axis=1)
    #
    #     return evals

    def forward(self, y_pred, y_true):
        if len(y_true.shape) == 1:
            y_pred = y_pred.unsqueeze(0)
            y_true = y_true.unsqueeze(0)

        #tensor_loss = F.mse_loss(y_pred, y_true)

        y_pred_resized = torch.empty(y_pred.shape[0], 2, 2)
        y_pred_resized[:, 0, 0] = y_pred[:, 0]
        y_pred_resized[:, 0, 1] = y_pred[:, 1]
        y_pred_resized[:, 1, 0] = y_pred[:, 1]
        y_pred_resized[:, 1, 1] = y_pred[:, 2]

        y_true_resized = torch.empty(y_true.shape[0], 2, 2)
        y_true_resized[:, 0, 0] = y_true[:, 0]
        y_true_resized[:, 0, 1] = y_true[:, 1]
        y_true_resized[:, 1, 0] = y_true[:, 1]
        y_true_resized[:, 1, 1] = y_true[:, 2]

        #pred_evals = self._calc_evals(y_pred_resized)

        ####
        ## Eigenvalues of predicted tensors
        ####
        eval_1 = ((y_pred_resized[:, 0, 0] + y_pred_resized[:, 1, 1]) +
                  torch.sqrt((-y_pred_resized[:, 0, 0] - y_pred_resized[:, 1, 1]) ** 2 -
                             4 * (y_pred_resized[:, 0, 0] * y_pred_resized[:, 1, 1] -
                                  y_pred_resized[:, 0, 1] * y_pred_resized[:, 1, 0]))) / 2

        eval_2 = ((y_pred_resized[:, 0, 0] + y_pred_resized[:, 1, 1]) -
                  torch.sqrt((-y_pred_resized[:, 0, 0] - y_pred_resized[:, 1, 1]) ** 2 -
                             4 * (y_pred_resized[:, 0, 0] * y_pred_resized[:, 1, 1] -
                                  y_pred_resized[:, 0, 1] * y_pred_resized[:, 1, 0]))) / 2

        pred_evals = torch.stack([eval_1, eval_2], axis=1)
        pred_eigenvalues, pred_eigenvectors = torch.linalg.eigh(y_pred_resized)

        #true_evals = self._calc_evals(y_true_resized)

        ####
        ## Eigenvalues of true/target tensors
        ####
        eval_1 = ((y_true_resized[:, 0, 0] + y_true_resized[:, 1, 1]) +
                  torch.sqrt((-y_true_resized[:, 0, 0] - y_true_resized[:, 1, 1]) ** 2 -
                             4 * (y_true_resized[:, 0, 0] * y_true_resized[:, 1, 1] -
                                  y_true_resized[:, 0, 1] * y_true_resized[:, 1, 0]))) / 2

        eval_2 = ((y_true_resized[:, 0, 0] + y_true_resized[:, 1, 1]) -
                  torch.sqrt((-y_true_resized[:, 0, 0] - y_true_resized[:, 1, 1]) ** 2 -
                             4 * (y_true_resized[:, 0, 0] * y_true_resized[:, 1, 1] -
                                  y_true_resized[:, 0, 1] * y_true_resized[:, 1, 0]))) / 2

        true_evals = torch.stack([eval_1, eval_2], axis=1)

        # print("pred evals ", pred_evals)
        # print("true evals ", true_evals)
        # print("pred_evals - true_evals ", pred_evals - true_evals)

        true_eigenvalues, true_eigenvectors = torch.linalg.eigh(y_true_resized)

        # Calculate MSE for eigenvalues
        evals_mse = torch.mean((pred_evals - true_evals) ** 2)

        mse_loss = F.mse_loss(y_pred, y_true)
        if str(mse_loss.device) == "cpu":
            self.weights = self.weights.cpu()
        weighted_mse_loss = self.weights[5] * mse_loss

        mse_evec_1_0 = 0
        mse_evec_1_1 = 0
        mse_evec_2_0 = 0
        mse_evec_2_1 = 0
        if self.weights[1] == 0 and self.weights[1] == 0:
            mse = self.weights[0] * evals_mse
            print("MSE evals: {}".format(mse))
            #print("MSE evals: {}, tensor MSE: {}".format(mse, tensor_loss))
        else:
            all_pred_evec_1 = torch.empty(len(pred_eigenvalues), 2)
            all_pred_evec_2 = torch.empty(len(pred_eigenvalues), 2)
            all_true_evec_1 = torch.empty(len(pred_eigenvalues), 2)
            all_true_evec_2 = torch.empty(len(pred_eigenvalues), 2)
            for i in range(len(pred_eigenvalues)):
                linalg_pred_evals = pred_eigenvalues[i]
                linalg_true_evals = true_eigenvalues[i]
                linalg_pred_evecs = pred_eigenvectors[i]
                linalg_true_evecs = true_eigenvectors[i]

                pred_evec_1 = linalg_pred_evecs[:, torch.argmin(torch.abs(linalg_pred_evals - pred_evals[i][0]))]
                pred_evec_2 = linalg_pred_evecs[:, torch.argmin(torch.abs(linalg_pred_evals - pred_evals[i][1]))]

                all_pred_evec_1[i] = pred_evec_1
                all_pred_evec_2[i] = pred_evec_2

                true_evec_1 = linalg_true_evecs[:, torch.argmin(torch.abs(linalg_true_evals - true_evals[i][0]))]
                true_evec_2 = linalg_true_evecs[:, torch.argmin(torch.abs(linalg_true_evals - true_evals[i][1]))]

                all_true_evec_1[i] = true_evec_1
                all_true_evec_2[i] = true_evec_2

            mse_evec_1_0 = torch.mean((all_pred_evec_1[:, 0] - all_true_evec_1[:, 0]) ** 2)
            mse_evec_1_1 = torch.mean((all_pred_evec_1[:, 1] - all_true_evec_1[:, 1]) ** 2)

            mse_evec_2_0 = torch.mean((all_pred_evec_2[:, 0] - all_true_evec_2[:, 0]) ** 2)
            mse_evec_2_1 = torch.mean((all_pred_evec_2[:, 1] - all_true_evec_2[:, 1]) ** 2)


            total_evec_mse = self.weights[1] * mse_evec_1_0 + self.weights[2] * mse_evec_1_1 + self.weights[3] * mse_evec_2_0 + self.weights[4] * mse_evec_2_1
            mse = self.weights[0] * evals_mse + total_evec_mse

            print("MSE evals: {}, evec_1_0: {}, evec_1_1: {},  evec_1_0: {}, evec_1_1: {}, data MSE: {}".format(evals_mse, mse_evec_1_0,mse_evec_1_1,
                                                                                                 mse_evec_2_0, mse_evec_2_1, weighted_mse_loss
                                                                                                 ))
        final_mse = mse + weighted_mse_loss
        return final_mse


class EighMSE_MSE(nn.Module):
    def __init__(self, weights):
        super(EighMSE_MSE, self).__init__()
        print("weights ", weights)
        self.weights = torch.Tensor(weights)
        if torch.cuda.is_available():
            self.weights = self.weights.cuda()

    def _calc_evals(self, batch_data):
        eval_1 = ((batch_data[:, 0, 0] + batch_data[:, 1, 1]) +
                       torch.sqrt((-batch_data[:, 0, 0] - batch_data[:, 1, 1]) ** 2 -
                                  4 * (batch_data[:, 0, 0] * batch_data[:, 1, 1] -
                                       batch_data[:, 0, 1] * batch_data[:, 1, 0]))) / 2

        eval_2 = ((batch_data[:, 0, 0] + batch_data[:, 1, 1]) -
                       torch.sqrt((-batch_data[:, 0, 0] - batch_data[:, 1, 1]) ** 2 -
                                  4 * (batch_data[:, 0, 0] * batch_data[:, 1, 1] -
                                       batch_data[:, 0, 1] * batch_data[:, 1, 0]))) / 2

        evals = torch.stack([eval_1, eval_2], axis=1)

        return evals

    def forward(self, y_pred, y_true):
        if len(y_true.shape) == 1:
            y_pred = y_pred.unsqueeze(0)
            y_true = y_true.unsqueeze(0)

        #tensor_loss = F.mse_loss(y_pred, y_true)

        y_pred_resized = torch.empty(y_pred.shape[0], 2, 2)
        y_pred_resized[:, 0, 0] = y_pred[:, 0]
        y_pred_resized[:, 0, 1] = y_pred[:, 1]
        y_pred_resized[:, 1, 0] = y_pred[:, 1]
        y_pred_resized[:, 1, 1] = y_pred[:, 2]

        y_true_resized = torch.empty(y_true.shape[0], 2, 2)
        y_true_resized[:, 0, 0] = y_true[:, 0]
        y_true_resized[:, 0, 1] = y_true[:, 1]
        y_true_resized[:, 1, 0] = y_true[:, 1]
        y_true_resized[:, 1, 1] = y_true[:, 2]

        #pred_evals = self._calc_evals(y_pred_resized)

        ####
        ## Eigenvalues of predicted tensors
        ####
        eval_1 = ((y_pred_resized[:, 0, 0] + y_pred_resized[:, 1, 1]) +
                  torch.sqrt((-y_pred_resized[:, 0, 0] - y_pred_resized[:, 1, 1]) ** 2 -
                             4 * (y_pred_resized[:, 0, 0] * y_pred_resized[:, 1, 1] -
                                  y_pred_resized[:, 0, 1] * y_pred_resized[:, 1, 0]))) / 2

        eval_2 = ((y_pred_resized[:, 0, 0] + y_pred_resized[:, 1, 1]) -
                  torch.sqrt((-y_pred_resized[:, 0, 0] - y_pred_resized[:, 1, 1]) ** 2 -
                             4 * (y_pred_resized[:, 0, 0] * y_pred_resized[:, 1, 1] -
                                  y_pred_resized[:, 0, 1] * y_pred_resized[:, 1, 0]))) / 2

        pred_evals = torch.stack([eval_1, eval_2], axis=1)
        pred_eigenvalues, pred_eigenvectors = torch.linalg.eigh(y_pred_resized)

        #true_evals = self._calc_evals(y_true_resized)

        ####
        ## Eigenvalues of true/target tensors
        ####
        eval_1 = ((y_true_resized[:, 0, 0] + y_true_resized[:, 1, 1]) +
                  torch.sqrt((-y_true_resized[:, 0, 0] - y_true_resized[:, 1, 1]) ** 2 -
                             4 * (y_true_resized[:, 0, 0] * y_true_resized[:, 1, 1] -
                                  y_true_resized[:, 0, 1] * y_true_resized[:, 1, 0]))) / 2

        eval_2 = ((y_true_resized[:, 0, 0] + y_true_resized[:, 1, 1]) -
                  torch.sqrt((-y_true_resized[:, 0, 0] - y_true_resized[:, 1, 1]) ** 2 -
                             4 * (y_true_resized[:, 0, 0] * y_true_resized[:, 1, 1] -
                                  y_true_resized[:, 0, 1] * y_true_resized[:, 1, 0]))) / 2

        true_evals = torch.stack([eval_1, eval_2], axis=1)

        # print("pred evals ", pred_evals)
        # print("true evals ", true_evals)
        # print("pred_evals - true_evals ", pred_evals - true_evals)

        true_eigenvalues, true_eigenvectors = torch.linalg.eigh(y_true_resized)

        # Calculate MSE for eigenvalues
        evals_mse = torch.mean((pred_evals - true_evals) ** 2)

        mse_loss = F.mse_loss(y_pred, y_true)
        if str(mse_loss.device) == "cpu":
            self.weights = self.weights.cpu()
        weighted_mse_loss = self.weights[3] * mse_loss

        mse_evec_1 = 0
        mse_evec_2 = 0
        if self.weights[1] == 0 and self.weights[1] == 0:
            mse = self.weights[0] * evals_mse
            print("MSE evals: {}, data mse: {}".format(mse, weighted_mse_loss))
            #print("MSE evals: {}, tensor MSE: {}".format(mse, tensor_loss))
        else:
            for i in range(len(pred_eigenvalues)):
                linalg_pred_evals = pred_eigenvalues[i]
                linalg_true_evals = true_eigenvalues[i]
                linalg_pred_evecs = pred_eigenvectors[i]
                linalg_true_evecs = true_eigenvectors[i]

                pred_evec_1 = linalg_pred_evecs[:, torch.argmin(torch.abs(linalg_pred_evals - pred_evals[i][0]))]
                pred_evec_2 = linalg_pred_evecs[:, torch.argmin(torch.abs(linalg_pred_evals - pred_evals[i][1]))]

                true_evec_1 = linalg_true_evecs[:, torch.argmin(torch.abs(linalg_true_evals - true_evals[i][0]))]
                true_evec_2 = linalg_true_evecs[:, torch.argmin(torch.abs(linalg_true_evals - true_evals[i][1]))]

                # Calculate MSE for eigenvectors
                mse_evec_1 += torch.mean((pred_evec_1 - true_evec_1) ** 2)
                mse_evec_2 += torch.mean((pred_evec_2 - true_evec_2) ** 2)

            total_evec_mse = self.weights[1] * mse_evec_1 + self.weights[2] * mse_evec_2
            mse = self.weights[0] * evals_mse + (total_evec_mse / (i+1))

            print("MSE evals: {}, evec1: {}, evec2: {}, data mse: {} total evec mse / batch size : {}".format(evals_mse, mse_evec_1,
                                                                                                 mse_evec_2, weighted_mse_loss,
                                                                                                 total_evec_mse / (i+1)))


        final_mse = mse + weighted_mse_loss
        return final_mse


class WeightedL1Loss(nn.Module):
    def __init__(self, weights):
        super(WeightedL1Loss, self).__init__()
        self.weights = torch.Tensor(weights)
        if torch.cuda.is_available():
            self.weights = self.weights.cuda()

    def forward(self, y_pred, y_true):
        abs_values = torch.abs(y_pred - y_true)
        if str(abs_values.device) == "cpu":
            self.weights = self.weights.cpu()
        abs_values = self.weights * abs_values

        if len(abs_values.shape)> 1:
            abs_values = torch.mean(abs_values, dim=1)

        #mean_abs_values = torch.mean(abs_values, dim=1)
        return torch.mean(abs_values)


class WeightedL1LossSum(nn.Module):
    def __init__(self, weights):
        super(WeightedL1LossSum, self).__init__()
        self.weights = torch.Tensor(weights)
        if torch.cuda.is_available():
            self.weights = self.weights.cuda()

    def forward(self, y_pred, y_true):
        abs_values = torch.abs(y_pred - y_true)
        if str(abs_values.device) == "cpu":
            self.weights = self.weights.cpu()
        abs_values = self.weights * abs_values

        if len(abs_values.shape)> 1:
            abs_values = torch.sum(abs_values, dim=1)
        return torch.mean(abs_values)


import torch
import torch.nn as nn


class CorrelatedOutputLoss(nn.Module):
    def __init__(self, loss_fn_params):
        super(CorrelatedOutputLoss, self).__init__()
        # Store the inverse covariance matrix (Sigma^{-1})
        self.covariance_matrix = None
        self.inv_cov_matrix = None #torch.linalg.inv(covariance_matrix)


    @staticmethod
    def calculate_cov(dataloader):
        mean_sum = None
        total_samples = 0

        for batch in dataloader:
            inputs, outputs = batch

            if torch.cuda.is_available():
                outputs = outputs.cuda()

            if mean_sum is None:
                mean_sum = outputs.sum(dim=0)
            else:
                mean_sum += outputs.sum(dim=0)
            total_samples += outputs.size(0)

        # Mean vector
        mean_vector = mean_sum / total_samples

        # Step 2: Calculate the covariance matrix
        cov_sum = None
        for batch in dataloader:
            inputs, outputs = batch
            if torch.cuda.is_available():
                outputs = outputs.cuda()

            centered_data = outputs - mean_vector  # Center the data
            batch_cov = torch.einsum('bi,bj->ij', centered_data, centered_data)  # Outer product
            if cov_sum is None:
                cov_sum = batch_cov
            else:
                cov_sum += batch_cov

        # Final covariance matrix
        covariance_matrix = cov_sum / (total_samples - 1)

        return covariance_matrix

    def set_cov(self, dataloader):
        self.covariance_matrix = CorrelatedOutputLoss.calculate_cov(dataloader)
        self.inv_cov_matrix = torch.linalg.inv(self.covariance_matrix)

        print("self. inv cov matrix ", self.inv_cov_matrix)

    def forward(self, predicted, target):

        # get_mse_nrmse_r2_eigh_3D(target.cpu(), predicted.cpu().detach().numpy())
        # exit()

        # Difference between true and predicted values
        diff = target - predicted  # Shape: (batch_size, 6)

        #print("diff ", diff)

        # Mahalanobis distance calculation
        # diff.T @ Sigma^{-1} @ diff
        mahalanobis = torch.einsum('bi,ij,bj->b', diff, self.inv_cov_matrix, diff)  # Shape: (batch_size,)

        # Return the mean Mahalanobis distance across the batch
        return mahalanobis.mean()


# class MaxEigenValue(nn.Module):
#     def __init__(self, weights):
#         super(MaxEigenValue, self).__init__()
#
#     def forward(self, y_pred, y_true):
#         print("y pred shape", y_pred.shape)
#
#         abs_values = torch.abs(y_pred - y_true)
#
#         tn_abs_values = voigt_to_tn(abs_values)
#
#         eigenvalues = torch.linalg.eigvals(tn_abs_values)
#         print("eigenvalues ", eigenvalues)
#         max_eigenvalues = eigenvalues.max(dim=-1).values
#
#         print("abs value ", abs_values.shape)
#         print("tn abs values ", tn_abs_values.shape)
#
#         print("max eigenvalues ", max_eigenvalues)
#         return torch.mean(abs_values)


def get_loss_fn(loss_function):
    loss_fn_name = loss_function[0]  # Extract the loss function name
    loss_fn_params = loss_function[1]  # Extract the parameters for the loss function

    # Return the corresponding loss function object based on the name
    if loss_fn_name == "MSE" or loss_fn_name == "L2":
        return nn.MSELoss()  # Mean Squared Error loss
    elif loss_fn_name == "L1":
        return nn.L1Loss()  # L1 loss (mean absolute error)
    elif loss_fn_name == "L1WeightedMean":
        return WeightedL1Loss(loss_fn_params)  # Weighted L1 loss with mean reduction
    elif loss_fn_name == "L1WeightedSum":
        return WeightedL1LossSum(loss_fn_params)  # Weighted L1 loss with sum reduction
    elif loss_fn_name == "Frobenius":
        return FrobeniusNorm()  # Frobenius norm loss
    elif loss_fn_name == "Frobenius2":
        return FrobeniusNorm2()  # Alternative Frobenius norm loss
    elif loss_fn_name == "MSEweighted":
        return WeightedMSELoss(loss_fn_params)  # Weighted MSE loss
    elif loss_fn_name == "RelMSELoss":
        return RelMSELoss(loss_fn_params)  # Relative MSE loss variant
    elif loss_fn_name == "MASELoss":
        return MASELoss(loss_fn_params)  # Mean Absolute Scaled Error loss
    elif loss_fn_name == "RelMSELoss2":
        return RelMSELoss2(loss_fn_params)  # Another variant of relative MSE loss
    elif loss_fn_name == "RelXYMSELoss":
        return RelXYMSELoss(loss_fn_params)  # Relative MSE loss for XY components
    elif loss_fn_name == "MSEweightedSum":
        return WeightedMSELossSum(loss_fn_params)  # Weighted MSE loss with sum reduction
    elif loss_fn_name == "EighMSE":
        return EighMSE(loss_fn_params)  # Eigenvalue-based MSE loss
    elif loss_fn_name == "EighMSE_2":
        return EighMSE_2(loss_fn_params)  # Variant 2 of Eigenvalue MSE loss
    elif loss_fn_name == "EighMSE_MSE":
        return EighMSE_MSE(loss_fn_params)  # Combined Eigenvalue and MSE loss
    elif loss_fn_name == "EighMSE_2_MSE":
        return EighMSE_2_MSE(loss_fn_params)  # Variant 2 combined Eigenvalue and MSE loss
    elif loss_fn_name == "MSELossLargeEmph":
        return MSELossLargeEmph(loss_fn_params)  # MSE loss with larger emphasis on some components
    elif loss_fn_name == "MSELossLargeEmphAvg":
        return MSELossLargeEmphAvg(loss_fn_params)  # Average version of large emphasis MSE loss
    elif loss_fn_name == "CorrelatedOutputLoss":
        return CorrelatedOutputLoss(loss_fn_params)  # Loss accounting for correlated outputs
    # The following are commented out and thus not currently used:
    # elif loss_fn_name == "MaxEigenValue":
    #     return MaxEigenValue(loss_fn_params)  # Loss related to max eigenvalue
    # elif loss_fn_name == "CosineSimilarity":
    #     return CosineSimilarity  # Cosine similarity loss


