import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import QuantileTransformer, RobustScaler


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
        self.input_indices = [0, 1, 2]
        self.output_indices = [0, 1, 2]
        self.input_mean = [0, 0, 0, 0]
        self.output_mean = [0, 0, 0, 0]
        self.input_std = [1, 1, 1, 1]
        self.output_std = [1,1,1, 1]
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

        flatten_data = data[1].flatten()
        positive_data_indices = flatten_data >= 1e-15
        negative_data_indices = flatten_data < 1e-15

        flatten_data[positive_data_indices] = torch.log(flatten_data[positive_data_indices])
        flatten_data[negative_data_indices] = torch.log(np.abs(flatten_data[negative_data_indices]))

        output_data[1][...] = np.reshape(flatten_data, data[1].shape)
        output_data[2][...] = torch.log(data[2])
    else:
        raise NotImplementedError("Log transformation implemented for 2D case only")

    return output_data


def init_norm(data):
    bulk_features_avg, input, output, cross_section_flag = data
    #avg_k = torch.mean(input)

    if cross_section_flag:
        input[:3, :] /= bulk_features_avg
    else:
        input /= bulk_features_avg
    output /= bulk_features_avg

    return input, output


def log_data(data):
    output_data = torch.empty((data.shape))
    if data.shape[0] == 3:
        output_data[0][...] = torch.log(data[0])
        output_data[1][...] = data[1]
        output_data[2][...] = torch.log(data[2])
    elif data.shape[0] == 4:
        output_data[0][...] = torch.log(data[0])
        output_data[1][...] = data[1]
        output_data[2][...] = torch.log(data[2])
        output_data[3][...] = data[3]
    elif data.shape[0] < 3:
        for i in range(data.shape[0]):
            output_data[i][...] = torch.log(data[i])
    else:
        raise NotImplementedError("Log transformation implemented for 2D case only")

    return output_data


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


def get_mean_std(data_loader, output_iqr=[]):
    channels_sum = None
    channels_sqrd_sum = None
    output_channels_sum = None
    output_channels_sqrd_sum = None
    num_batches = 0
    output_data_list = []
    quantiles = []
    for input, output in data_loader:
        if channels_sum is None:
            channels_sum = list(np.zeros(input.shape[1]))
            channels_sqrd_sum = list(np.zeros(input.shape[1]))
            output_channels_sum = list(np.zeros(output.shape[1]))
            output_channels_sqrd_sum = list(np.zeros(output.shape[1]))

        if len(output_iqr) > 0:
            output_data_list.extend(output.numpy())

        channels_sum += (torch.nanmean(input, dim=[0, 2, 3])).numpy()
        channels_sqrd_sum += (torch.nanmean(input ** 2, dim=[0, 2, 3])).numpy()
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


def check_shapes(n_conv_layers, kernel_size, stride, pool_size, pool_stride, pool_indices, input_size=256):
    #n_layers = 0

    for i in range(n_conv_layers):
        # print("input size ", input_size)
        # print("kernel size ", kernel_size)

        if input_size < kernel_size:
            return -1, input_size

        input_size = int(((input_size - kernel_size) / stride)) + 1

        if pool_indices is not None:
            if i in list(pool_indices.keys()):
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


def get_mse_nrmse_r2_eigh(targets, predictions):
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


def get_loss_fn(loss_function):
    loss_fn_name = loss_function[0]
    loss_fn_params = loss_function[1]
    if loss_fn_name == "MSE" or loss_fn_name == "L2":
        return nn.MSELoss()
    elif loss_fn_name == "L1":
        return nn.L1Loss()
    elif loss_fn_name == "L1WeightedMean":
        return WeightedL1Loss(loss_fn_params)
    elif loss_fn_name == "L1WeightedSum":
        return WeightedL1LossSum(loss_fn_params)
    elif loss_fn_name == "Frobenius":
        return FrobeniusNorm()
    elif loss_fn_name == "Frobenius2":
        return FrobeniusNorm2()
    elif loss_fn_name == "MSEweighted":
        return WeightedMSELoss(loss_fn_params)
    elif loss_fn_name == "MSEweightedSum":
        return WeightedMSELossSum(loss_fn_params)
    elif loss_fn_name == "EighMSE":
        return EighMSE(loss_fn_params)
    elif loss_fn_name == "EighMSE_MSE":
        return EighMSE_MSE(loss_fn_params)
    # elif loss_fn_name == "CosineSimilarity":
    #     return CosineSimilarity


