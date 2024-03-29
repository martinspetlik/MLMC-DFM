import os
import sys
import argparse
import joblib
import pickle
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from metamodel.cnn.models.net1 import Net1
from metamodel.cnn.models.trials.net_optuna_2 import Net
from metamodel.cnn.datasets.dfm_dataset import DFMDataset
#from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from metamodel.cnn.visualization.visualize_tensor import plot_tensors
from metamodel.cnn.visualization.visualize_data import plot_target_prediction, plot_train_valid_loss
from metamodel.cnn.models.auxiliary_functions import exp_data, get_eigendecomp, get_mse_nrmse_r2, get_mean_std, log_data, exp_data,\
    quantile_transform_fit, QuantileTRF, NormalizeData, log_all_data, init_norm
from metamodel.cnn.models.train_pure_cnn_optuna import prepare_dataset
#from sklearn.isotonic import IsotonicRegression
#from statsmodels.regression.quantile_regression import QuantReg
#from sklearn.linear_model import LogisticRegression
#from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import QuantileTransformer, RobustScaler


def preprocess_dataset(config, user_attrs, data_dir, results_dir=None):
    print("user attrs ", user_attrs)
    output_file_name = "output_tensor.npy"
    if "vel_avg" in config and config["vel_avg"]:
        output_file_name = "output_vel_avg.npy"

    # ===================================
    # Get mean and std for each channel
    # ===================================
    input_mean, output_mean, input_std, output_std = 0, 0, 1, 1
    data_normalizer = NormalizeData()

    n_train_samples = None
    if "n_train_samples" in config and config["n_train_samples"] is not None:
        n_train_samples = config["n_train_samples"]

    input_transformations = []
    output_transformations = []

    data_input_transform, data_output_transform = None, None
    # Standardize input
    if config["log_input"]:
        if "log_all_input_channels" in config and config["log_all_input_channels"]:
            input_transformations.append(transforms.Lambda(log_all_data))
        else:
            input_transformations.append(transforms.Lambda(log_data))
    if config["normalize_input"]:
        if "normalize_input_indices" in config:
            data_normalizer.input_indices = config["normalize_input_indices"]
        data_normalizer.input_mean = user_attrs["input_mean"]
        data_normalizer.input_std = user_attrs["input_std"]
        input_transformations.append(data_normalizer.normalize_input)

    if len(input_transformations) > 0:
        data_input_transform = transforms.Compose(input_transformations)

    #print("input transforms ", input_transformations)

    # Standardize output
    if config["log_output"]:
        output_transformations.append(transforms.Lambda(log_data))
    if config["normalize_output"]:
        if "normalize_output_indices" in config:
            data_normalizer.input_indices = config["normalize_output_indices"]
        data_normalizer.output_mean = user_attrs["output_mean"]
        data_normalizer.output_std = user_attrs["output_std"]
        output_transformations.append(data_normalizer.normalize_output)

    print("data dir ", data_dir)
    print("results dir ", results_dir)
    if os.path.exists(os.path.join(results_dir, "output_transform.pkl")):
        out_trans_obj = joblib.load(os.path.join(results_dir, "output_transform.pkl"))
        quantile_trf_obj = QuantileTRF()
        quantile_trf_obj.quantile_trfs_out = out_trans_obj
        output_transformations.append(quantile_trf_obj.quantile_transform_out)
    else:
        if "output_transform" in config and config["output_transform"]:
            raise Exception("{} not exists".format(os.path.join(results_dir, "output_transform.pkl")))

    if len(output_transformations) > 0:
        data_output_transform = transforms.Compose(output_transformations)

    init_transform = []
    data_init_transform = None
    if "init_norm" in study.user_attrs and study.user_attrs["init_norm"]:
        init_transform.append(transforms.Lambda(init_norm))

    if len(init_transform) > 0:
        data_init_transform = transforms.Compose(init_transform)

    print("preprocess data input transform ", data_input_transform)
    print("preprocess data output transform ", data_output_transform)


    # ============================
    # Datasets and data loaders
    # ============================
    dataset = DFMDataset(data_dir=data_dir,
                         output_file_name=output_file_name,
                         init_transform=data_init_transform,
                         input_transform=data_input_transform,
                         output_transform=data_output_transform,
                         two_dim=True,
                         input_channels=config["input_channels"] if "input_channels" in config else None,
                         output_channels=config["output_channels"] if "output_channels" in config else None,
                         fractures_sep=config["fractures_sep"] if "fractures_sep" in config else False,
                         vel_avg=config["vel_avg"] if "vel_avg" in config else False
                         )
    dataset.shuffle(config["seed"])
    print("len dataset ", len(dataset))

    if n_train_samples is None:
        n_train_samples = int(len(dataset) * config["train_samples_ratio"])

    n_train_samples = np.min([n_train_samples, int(len(dataset) * config["train_samples_ratio"])])

    train_val_set = dataset[:n_train_samples]
    train_set = train_val_set[:-int(n_train_samples * config["val_samples_ratio"])]
    validation_set = train_val_set[-int(n_train_samples * config["val_samples_ratio"]):]

    if "n_test_samples" in config and config["n_test_samples"] is not None:
        n_test_samples = config["n_test_samples"]
        test_set = dataset[-n_test_samples:]
    else:
        test_set = dataset[n_train_samples:]

    return train_set, validation_set, test_set

    # print("len(trainset): {}, len(valset): {}, len(testset): {}".format(len(train_set), len(validation_set),
    #                                                                     len(test_set)))


def get_calibrator():
    return IsotonicRegression(out_of_bounds="clip")
    #return LogisticRegression()

def calibrator_fit(calibrator, predictions, targets):
    #predictions = np.array(predictions).reshape(-1, 1)
    calibrator.fit(predictions, targets)

def calibrator_predict(calibrator, data):
    return calibrator.predict(data)
    #return calibrator(data)[:, 1]

def calibrate_data(data_list, calibs):
    data_array = np.array(data_list)
    #print("data_array.shape ", data_array.shape)
    all_data = []
    for idx, calib in enumerate(calibs):
        cal_data = calib.predict(data_array[:, idx])
        all_data.append(cal_data)

    # pred_0, pred_1, pred_2 = pred_array[:, 0], pred_array[:, 1], pred_array[:, 2]
    #
    # cal_pred_0 = calibrator_0.predict(pred_0)
    # cal_pred_1 = calibrator_1.predict(pred_1)
    # cal_pred_2 = calibrator_2.predict(pred_2)

    return list(np.stack(all_data, axis=1))


def get_saved_model_path(results_dir, best_trial):
    #print(best_trial.user_attrs["model_name"])
    model_path = 'trial_{}_losses_model_{}'.format(best_trial.number, best_trial.user_attrs["model_name"])

    #model_path = "trial_2_losses_model_cnn_net"

    #@TODO: remove ASAP
    #return "/home/martin/Documents/MLMC-DFM/optuna_runs/karolina/cond_conv/exp_2/seed_12345/trial_1_losses_model_cnn_net"
    #return "/home/martin/Documents/MLMC-DFM/optuna_runs/karolina/pooling/new_experiments/exp_13_4_s_36974/seed_12345/trial_0_losses_model_cnn_net"

    # for key, value in best_trial.params.items():
    #     model_path += "_{}_{}".format(key, value)

    #@TODO: remove ASAP
    #model_path = "trial_26_train_samples_15000"
    #model_path = "/home/martin/Documents/MLMC-DFM/optuna_runs/karolina/cnn_3_3/exp_2/seed_12345/trial_6_losses_model_cnn_net_max_channel_72_n_conv_layers_1_kernel_size_3_stride_1_pool_None_pool_size_0_pool_stride_0_lr_0.005_use_batch_norm_True_max_hidden_neurons_48_n_hidden_layers_1"
    return os.path.join(results_dir, model_path)


def load_dataset(results_dir, study):
    dataset = joblib.load(os.path.join(results_dir, "dataset.pkl"))
    if not hasattr(dataset, '_fractures_sep'):
        dataset._fractures_sep = False
    if not hasattr(dataset, '_vel_avg'):
        dataset._vel_avg = False
    if not hasattr(dataset, 'init_transform'):
        dataset.init_transform = None
    if dataset.output_transform is not None:
        if not hasattr(dataset.output_transform, 'output_quantiles'):
            dataset.output_transform.output_quantiles = []

    print("dataset.output_transform ", dataset.output_transform)

    if os.path.exists(os.path.join(results_dir, "output_transform.pkl")) or os.path.exists(os.path.join(results_dir, "input_transform.pkl")):
        quantile_trf_obj = QuantileTRF()
        if os.path.exists(os.path.join(results_dir, "output_transform.pkl")):
            out_trans_obj = joblib.load(os.path.join(results_dir, "output_transform.pkl"))
            quantile_trf_obj.quantile_trfs_out = out_trans_obj
            print("out trans obj ", out_trans_obj)
        if os.path.exists(os.path.join(results_dir, "input_transform.pkl")):
            in_trans_obj = joblib.load(os.path.join(results_dir, "input_transform.pkl"))
            quantile_trf_obj.quantile_trfs_in = in_trans_obj


    n_train_samples = study.user_attrs["n_train_samples"]
    n_val_samples = study.user_attrs["n_val_samples"]
    # print("len(dataset) ", len(dataset))
    train_val_set = dataset[:(n_train_samples + n_val_samples)]

    if n_val_samples == 0:
        train_set = train_val_set
        validation_set = []
    else:
        train_set = train_val_set[:-n_val_samples]
        validation_set = train_val_set[-n_val_samples:]
    n_test_samples = study.user_attrs["n_test_samples"]
    test_set = dataset[-n_test_samples:]

    return train_set, validation_set, test_set

    data_dir = "/home/martin/Documents/MLMC-DFM_data/nn_data/homogenization_samples_5LMC_L4"
    data_dir = "/home/martin/Documents/MLMC-DFM_data/nn_data/homogenization_samples_MLMC-DFM_5LMC-L4_cl_0" #_0_7_norm_output"
    data_dir = "/home/martin/Documents/MLMC-DFM_data/nn_data/homogenization_samples_MLMC-DFM_hom_nn_gen_hom_samples_cl_10_0_overlap_subset"
    #data_dir = "/home/martin/Documents/MLMC-DFM_data/nn_data/homogenization_samples_MLMC-DFM_5LMC-L4_cl_6"
    #data_dir = "/home/martin/Documents/MLMC-DFM_data/nn_data/homogenization_samples_charon"
    data_dir = "/home/martin/Documents/MLMC-DFM_data/nn_data/homogenization_samples_MLMC-DFM_5LMC_L1_cl_v_0_overlap"
    #data_dir = "/home/martin/Documents/MLMC-DFM_data/nn_data/homogenization_samples_MLMC-DFM_hom_nn_rho_5_0_no_fractures"
    data_dir = "/home/martin/Documents/MLMC-DFM_data/nn_data/homogenization_samples_MLMC-DFM_hom_nn_rho_10_0_no_sigma_rast_3"
    #data_dir = "/home/martin/Documents/MLMC-DFM_data/nn_data/homogenization_samples_MLMC-DFM_hom_nn_rho_10_0_no_sigma_step_1_871_rast_3"
    #data_dir = "/home/martin/Documents/MLMC-DFM_data/nn_data/homogenization_samples_MLMC-DFM_hom_nn_rho_10_0_no_sigma_step_4_325_fr_mult_2_5"
    #data_dir = "/home/martin/Documents/MLMC-DFM_data/nn_data/homogenization_samples_MLMC-DFM_hom_nn_rho_10_0_no_sigma_step_1_871_fr_div_10_rast_3"
    #data_dir = "/home/martin/Documents/MLMC-DFM_data/nn_data/MLMC-DFM_general_dataset/test_datasets/homogenization_samples_MLMC-DFM_hom_nn_rho_10_0_no_sigma_mean_min_8"
    data_dir = "/home/martin/Documents/MLMC-DFM_data/nn_data/MLMC-DFM_general_dataset_2/test_datasets/homogenization_samples_MLMC-DFM_hom_nn_rho_10_0_no_sigma_fr_div_0_1"
    config = {#"num_epochs": trials_config["num_epochs"],
              "batch_size_train": 32,
              # "batch_size_test": 250,
              "n_train_samples": study.user_attrs["n_train_samples"],
              "n_test_samples": study.user_attrs["n_test_samples"],
              "train_samples_ratio": 0.8,
              "val_samples_ratio":  0.2,
              "print_batches": 10,
              "init_norm": study.user_attrs["init_norm"],
              "log_input": study.user_attrs["input_log"],
              "normalize_input": study.user_attrs["normalize_input"],
              "log_output": study.user_attrs["output_log"],
              "normalize_output": study.user_attrs["normalize_output"],
              #"output_transform": study.user_attrs["output_transform"],
              "seed": 12345}

    if "output_transform" in study.user_attrs:
        config["output_transform"] = study.user_attrs["output_transform"]

    return preprocess_dataset(config, study.user_attrs, data_dir=data_dir, results_dir=results_dir)


def get_rotation_angles(model, loader, inverse_transform):
    targets_list, predictions_list = [], []
    inv_targets_list, inv_predictions_list = [], []

    for i, test_sample in enumerate(loader):
        inputs, targets = test_sample
        inputs = inputs.float()

        if args.cuda and torch.cuda.is_available():
            inputs = inputs.cuda()
        predictions = model(inputs)

        if len(targets.size()) > 1 and np.sum(targets.size()) / len(targets.size()) != 1:
            targets = torch.squeeze(targets.float())
        targets_np = targets.numpy()
        targets_list.append(targets_np)

        if len(predictions.size()) > 1 and np.sum(predictions.size()) / len(predictions.size()) != 1:
            predictions = torch.squeeze(predictions.float())
        predictions = predictions.cpu()
        predictions_np = predictions.cpu().numpy()

        predictions_list.append(predictions_np)

        inv_targets = targets
        inv_predictions = predictions
        if inverse_transform is not None:
            inv_targets = inverse_transform(torch.reshape(targets, (*targets.shape, 1, 1)))
            inv_predictions = inverse_transform(torch.reshape(predictions, (*predictions.shape, 1, 1)))

            inv_targets = np.reshape(inv_targets, targets.shape)
            inv_predictions = np.reshape(inv_predictions, predictions.shape)

        inv_targets_list.append(inv_targets.numpy())
        inv_predictions_list.append(inv_predictions.numpy())

    all_targets = np.array(targets_list)
    all_predictions = np.array(predictions_list)

    if len(all_targets.shape) == 1:
        all_targets = np.reshape(all_targets, (*all_targets.shape, 1))
        all_predictions = np.reshape(all_predictions, (*all_predictions.shape, 1))

    n_channels = 1 if len(all_targets.shape) == 1 else all_targets.shape[1]

    angles = []
    for i in range(n_channels):
        k_target, k_predict = all_targets[:, i], all_predictions[:, i]
        angle, slope, intercept = get_channel_angles(k_target, k_predict)
        angles.append((angle, slope, intercept))
    return angles


# def find_predictions_transform(k_target, k_predict):
#     print("k target ", k_target)
#     print("k predict ", k_predict)
#     pass

def get_channel_angles(targets, predictions):
    slope, intercept = np.polyfit(targets, predictions, 1)
    angle = np.pi/4 - np.arctan(slope)
    return angle, slope, intercept


def rotate_by_angle(targets, predictions, angles):
    n_channels = targets.shape[0]
    rotated_predictions = []

    for i in range(n_channels):
        angle, shift, intercept = angles[i]
        #print("orig predictions ", predictions[i])
        predictions[i] -= shift
        #print("shifted predictions ", predictions[i])
        #yr = (predictions[i] * np.cos(angle)) + shift
        #print("yr ", yr)
        yr = (targets[i] * np.sin(angle)) + (predictions[i] * np.cos(angle)) + shift
        #print("yr tr pr", yr)
        rotated_predictions.append(yr)
    return torch.from_numpy(np.array(rotated_predictions))


def get_inverse_transform_input(study):
    inverse_transform = None
    print("study.user_attrs", study.user_attrs)

    if "normalize_input" in study.user_attrs and study.user_attrs["normalize_input"]:
        std = 1 / study.user_attrs["input_std"]
        zeros_mean = np.zeros(len(study.user_attrs["input_mean"]))

        print("input_mean ", study.user_attrs["input_mean"])
        print("input_std ", study.user_attrs["input_std"])

        ones_std = np.ones(len(zeros_mean))
        mean = -study.user_attrs["input_mean"]

        transforms_list = [transforms.Normalize(mean=zeros_mean, std=std),
                           transforms.Normalize(mean=mean, std=ones_std)]

        if "input_log" in study.user_attrs and study.user_attrs["input_log"]:
            print("input log to transform list")
            transforms_list.append(transforms.Lambda(exp_data))

        inverse_transform = transforms.Compose(transforms_list)

    if "init_norm" in study.user_attrs and study.user_attrs["init_norm"]:
        print("init norm ")

    return inverse_transform


def get_transform(study, results_dir=None):
    input_transformations = []
    output_transformations = []
    init_transform = []

    data_normalizer = NormalizeData()

    ###########################
    ## Initial normalization ##
    ###########################
    data_init_transform = None
    if "init_norm" in study.user_attrs and study.user_attrs["init_norm"]:
        init_transform.append(transforms.Lambda(init_norm))

    if len(init_transform) > 0:
        data_init_transform = transforms.Compose(init_transform)

    # if "input_transform" in study.user_attrs or "output_transform" in study.user_attrs:
    #     input_transformations, output_transformations = features_transform(config, data_dir, output_file_name,
    #                                                                        input_transformations,
    #                                                                        output_transformations, train_set)

    data_input_transform, data_output_transform = None, None
    # Standardize input
    if "input_log" in study.user_attrs and study.user_attrs["input_log"]:
        if "log_all_input_channels" in study.user_attrs and study.user_attrs["log_all_input_channels"]:
            input_transformations.append(transforms.Lambda(log_all_data))
        else:
            input_transformations.append(transforms.Lambda(log_data))
    if "normalize_input" in study.user_attrs and study.user_attrs["normalize_input"]:
        if "normalize_input_indices" in study.user_attrs:
            data_normalizer.input_indices = study.user_attrs["normalize_input_indices"]
        data_normalizer.input_mean = study.user_attrs["input_mean"]
        data_normalizer.input_std = study.user_attrs["input_std"]
        input_transformations.append(data_normalizer.normalize_input)

    if len(input_transformations) > 0:
        data_input_transform = transforms.Compose(input_transformations)

    # Standardize output
    if "output_log" in study.user_attrs and study.user_attrs["output_log"]:
        output_transformations.append(transforms.Lambda(log_data))
    if "normalize_output" in study.user_attrs and study.user_attrs["normalize_output"]:
        if "normalize_output_indices" in study.user_attrs:
            data_normalizer.output_indices = study.user_attrs["normalize_output_indices"]
        data_normalizer.output_mean = study.user_attrs["output_mean"]
        data_normalizer.output_std = study.user_attrs["output_std"]
        if "output_quantiles" in study.user_attrs:
            data_normalizer.output_quantiles = study.user_attrs["output_quantiles"]
        output_transformations.append(data_normalizer.normalize_output)

    transforms_list = []
    if ("output_transform" in study.user_attrs and len(study.user_attrs["output_transform"]) > 0) \
            or os.path.exists(os.path.join(results_dir, "output_transform.pkl")):
        output_transform = joblib.load(os.path.join(results_dir, "output_transform.pkl"))
        quantile_trf_obj = QuantileTRF()
        quantile_trf_obj.quantile_trfs_out = output_transform
        transforms_list.append(quantile_trf_obj.quantile_inv_transform_out)

    if len(output_transformations) > 0:
        data_output_transform = transforms.Compose(output_transformations)

    return data_init_transform, data_input_transform, data_output_transform


def get_inverse_transform(study, results_dir=None):
    inverse_transform = None
    print("study.user_attrs", study.user_attrs)

    transforms_list = []
    if ("output_transform" in study.user_attrs and len(study.user_attrs["output_transform"]) > 0) \
            or os.path.exists(os.path.join(results_dir, "output_transform.pkl")):
        output_transform = joblib.load(os.path.join(results_dir, "output_transform.pkl"))
        quantile_trf_obj = QuantileTRF()
        quantile_trf_obj.quantile_trfs_out = output_transform
        transforms_list.append(quantile_trf_obj.quantile_inv_transform_out)

    if "normalize_output" in study.user_attrs and study.user_attrs["normalize_output"]:
        std = 1/study.user_attrs["output_std"]
        zeros_mean = np.zeros(len(study.user_attrs["output_mean"]))

        print("output_mean ", study.user_attrs["output_mean"])
        print("output_std ",  study.user_attrs["output_std"])

        ones_std = np.ones(len(zeros_mean))
        mean = -study.user_attrs["output_mean"]

        transforms_list.extend([transforms.Normalize(mean=zeros_mean, std=std),
                            transforms.Normalize(mean=mean, std=ones_std)])

        if "output_log" in study.user_attrs and study.user_attrs["output_log"]:
            print("output log to transform list")
            transforms_list.append(transforms.Lambda(exp_data))

        inverse_transform = transforms.Compose(transforms_list)

    print("inverse transform ", inverse_transform)

    return inverse_transform


def renormalize_data(dataset, study, input=False, output=False):
    print("dataset.input_transform ", dataset.input_transform)

    import copy

    new_dataset = copy.deepcopy(dataset)

    if input:
        transforms_list = []
        if "input_log" in study.user_attrs and study.user_attrs["input_log"]:
            transforms_list.append(transforms.Lambda(log_data))

        input_transform = transforms.Compose(transforms_list)

        new_dataset.input_transform = input_transform #None
        #new_dataset.input_transform = None

        loader = torch.utils.data.DataLoader(new_dataset, shuffle=False)

        input_mean, input_std, output_mean, output_std, _ = get_mean_std(loader)
        print("Test loader, INPUT mean: {}, std: {}, OUTPUT mean: {}, std: {}".format(input_mean, input_std, output_mean, output_std))

        if "normalize_input" in study.user_attrs and study.user_attrs["normalize_input"]:
            transforms_list.append(transforms.Normalize(mean=input_mean, std=input_std))

        input_transform = transforms.Compose(transforms_list)

        new_dataset.input_transform = input_transform
        # new_dataset.input_transform = None

        #loader = torch.utils.data.DataLoader(new_dataset, shuffle=False)

    if output:
        # transforms_list = []
        # if "normalize_output" in study.user_attrs and study.user_attrs["normalize_output"]:
        #     std = 1 / study.user_attrs["output_std"]
        #     zeros_mean = np.zeros(len(study.user_attrs["output_mean"]))
        #     print("output_mean ", study.user_attrs["output_mean"])
        #     print("output_std ", study.user_attrs["output_std"])
        #     ones_std = np.ones(len(zeros_mean))
        #     mean = -study.user_attrs["output_mean"]
        #     transforms_list = [transforms.Normalize(mean=zeros_mean, std=std),
        #                        transforms.Normalize(mean=mean, std=ones_std)]
        #
        # if "output_log" in study.user_attrs and study.user_attrs["output_log"]:
        #     print("output log to transform list")
        #     transforms_list.append(transforms.Lambda(exp_data))
        #
        # inverse_transform = transforms.Compose(transforms_list)

        # if "output_log" in study.user_attrs and study.user_attrs["output_log"]:
        #     transforms_list.append(transforms.Lambda(exp_data))
        # output_transform = transforms.Compose(transforms_list)
        #
        # new_dataset.output_transform = output_transform
        # # new_dataset.input_transform = None

        transforms_list = []
        new_dataset.output_transform = None

        loader = torch.utils.data.DataLoader(new_dataset, shuffle=False)

        input_mean, input_std, output_mean, output_std, _ = get_mean_std(loader)
        print(
            "Test loader, INPUT mean: {}, std: {}, OUTPUT mean: {}, std: {}".format(input_mean, input_std, output_mean,
                                                                                    output_std))

        if "normalize_output" in study.user_attrs and study.user_attrs["normalize_output"]:
            transforms_list.append(transforms.Normalize(mean=output_mean, std=output_std))

        output_transform = transforms.Compose(transforms_list)

        new_dataset.output_transform = output_transform
        # new_dataset.input_transform = None

    loader = torch.utils.data.DataLoader(new_dataset, shuffle=False)

    input_mean, input_std, output_mean, output_std, _ = get_mean_std(loader)
    print("Renormalized loader, INPUT mean: {}, std: {}, OUTPUT mean: {}, std: {}".format(input_mean, input_std, output_mean,
                                                                                  output_std))

    return loader


def plot_target_data(train_loader, validation_loader, test_loader):
    import matplotlib.pyplot as plt
    train_targets = []
    val_targets = []
    test_targets = []

    k_xx_list_input = []
    k_xy_list_input = []
    k_yy_list_input = []
    for i, test_sample in enumerate(train_loader):
        inputs, targets = test_sample
        inputs = np.squeeze(inputs.numpy())
        print("inputs ", inputs)
        if i > 50:
            break
        print("inputs[0].ravel() ", inputs[0].ravel())
        k_xx_list_input.extend(inputs[0].ravel())
        k_xy_list_input.extend(inputs[1].ravel())
        k_yy_list_input.extend(inputs[2].ravel())

    print("kxx shape ", np.array(k_xx_list_input).shape)
    plt.hist(k_xx_list_input, bins=25, color="red", label="input k_xx", density=True)
    #plt.xlim([-0.001, 1])
    plt.legend()
    plt.show()

    plt.hist(k_xy_list_input, bins=60, color="red", label="input k_xy", density=True)
    plt.legend()
    plt.show()

    plt.hist(k_yy_list_input, bins=60, color="red", label="input k_yy", density=True)
    plt.legend()
    plt.show()

    for i, test_sample in enumerate(train_loader):
        inputs, targets = test_sample
        targets_np = np.squeeze(targets.numpy())
        #print("targets_np ", targets_np)
        train_targets.append(targets_np)
        #exit()

    # for i, test_sample in enumerate(validation_loader):
    #     inputs, targets = test_sample
    #     targets_np = np.squeeze(targets.numpy())
    #     val_targets.append(targets_np)
    #
    # for i, test_sample in enumerate(test_loader):
    #     inputs, targets = test_sample
    #     targets_np = np.squeeze(targets.numpy())
    #     test_targets.append(targets_np)

    train_targets = np.array(train_targets)
    #val_targets = np.array(val_targets)
    #test_targets = np.array(test_targets)

    # ########################
    # # Quantile transform  ##
    # ########################
    # print("QUANTILE TRANSFORM")
    # quantile_trf_obj = QuantileTRF()
    # quantile_trfs_out = quantile_transform_fit(train_targets.T,
    #                                            indices=[1],
    #                                            transform_type="QuantileTransform")
    # quantile_trf_obj.quantile_trfs_out = quantile_trfs_out
    #
    # print("train targets shape ", train_targets.shape)
    # trf_train_targets = quantile_trf_obj.quantile_transform_out_real(train_targets.T)
    #
    # print("trf train tragets shape ", trf_train_targets.shape)
    # plt.hist(trf_train_targets[1, :], bins=1000, color="red", label="QuantileTRF train", alpha=0.5, density=True)
    # plt.legend()
    # plt.show()

    # trf_test_targets = quantile_trf_obj.quantile_transform_out_real(test_targets.T)
    #
    # print("trf test tragets shape ", trf_test_targets.shape)
    # print("trf test targets data ", trf_test_targets[1, :100])
    # plt.hist(trf_test_targets[1, :], bins=1000, color="red", label="QuantileTRF test", alpha=0.5, density=True)
    # plt.legend()
    # plt.show()

    n_channels = 3
    for i in range(n_channels):
        print("train targets shape ", train_targets.shape)
        k_train_targets = train_targets[:, i]

        #k_train_targets = k_train_targets[k_train_targets < 0.6]
        #k_val_targets = val_targets[:, i]
        #k_test_targets = test_targets[:, i]
        print("min: {}, max: {}, avg:{} ".format(np.min(k_train_targets),
                                                 np.max(k_train_targets),
                                                 np.mean(k_train_targets)))
        print("k_train_targets[:100] ", k_train_targets[:100])
        print("k_train_targets ", k_train_targets.shape)
        print("k train targets ")

        density = True
        plt.hist(k_train_targets, bins=1000, color="red", label="train, ch: {}".format(i), alpha=0.5, density=density)

        #plt.hist(k_val_targets, bins=60, color="blue", label="val", alpha=0.5, density=density)
        #plt.hist(k_test_targets, bins=60, color="green", label="test", alpha=0.5, density=density)
        #plt.xlabel(xlabel)
        # plt.ylabel("Frequency for relative")
        # if i == 1:
        #     plt.xlim([-1, 1])
        plt.legend()
        #plt.savefig("hist_" + title + ".pdf")
        plt.show()
        # if i == 1:
        #     b = 0.1
        #     # normal_sample = k_train_targets / (b * np.sqrt(2))
        #     # plt.hist(normal_sample, bins=1000, color="red", label="train, ch: {}".format(i), alpha=0.5, density=density)
        #     # plt.legend()
        #     # plt.show()
        #
        #     n_quantiles = 10000
        #     transformer = QuantileTransformer(n_quantiles=n_quantiles, random_state=0, output_distribution="normal")
        #     transform_obj = transformer.fit(k_train_targets.reshape(-1, 1))
        #
        #     transformed_data = transform_obj.transform(k_train_targets.reshape(-1, 1))
        #     plt.hist(transformed_data, bins=1000, color="red", label="train, ch: {}".format(i), alpha=0.5, density=density)
        #     plt.legend()
        #     plt.show()
        #
        #     exit()

        # exit()


def load_models(args, study):
    calibrate = False
    rotate = False
    results_dir = args.results_dir
    model_path = get_saved_model_path(results_dir, study.best_trial)
    print("model path ", model_path)
    train_set, validation_set, test_set = load_dataset(results_dir, study)

    #test_loader = torch.utils.data.DataLoader(test_set, shuffle=False)

    # input_mean, input_std, output_mean, output_std = get_mean_std(train_loader)
    # print("Train loader, input mean: {}, std: {}".format(input_mean, input_std))

    #dataset = joblib.load("/home/martin/Documents/MLMC-DFM/optuna_runs/karolina/cond_conv/exp_2/dataset.pkl")
    #print("len dataset ", len(dataset))
    # n_train_samples = study.user_attrs["n_train_samples"]
    # n_val_samples = study.user_attrs["n_val_samples"]
    #
    # #print("len(dataset) ", len(dataset))
    # # print("n val samples ", n_val_samples)
    #
    # train_val_set = dataset[:(n_train_samples+n_val_samples)]
    # train_set = train_val_set[:-n_val_samples]
    # validation_set = train_val_set[-n_val_samples:]
    # n_test_samples = study.user_attrs["n_test_samples"]
    # test_set = dataset[-n_test_samples:]
    # #test_set = dataset[:study.user_attrs["n_train_samples"]]
    #
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=False)
    validation_loader = torch.utils.data.DataLoader(validation_set, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False)

    print("len(trainset): {}, len(valset): {}, len(testset): {}".format(len(train_set), len(validation_set),
                                                                        len(test_set)))

    # input_mean, input_std, output_mean, output_std, _ = get_mean_std(train_loader)
    # print("Train loader, INPUT mean: {}, std: {}, OUTPUT mean: {}, std: {}".format(input_mean, input_std, output_mean, output_std))
    # #@TODO: skip when validation_loader has empty set
    # input_mean, input_std, output_mean, output_std, _ = get_mean_std(validation_loader)
    # print("Validation loader, INPUT mean: {}, std: {}, OUTPUT mean: {}, std: {}".format(input_mean, input_std, output_mean, output_std))
    # input_mean, input_std, output_mean, output_std, _ = get_mean_std(test_loader)
    # print("Test loader, INPUT mean: {}, std: {}, OUTPUT mean: {}, std: {}".format(input_mean, input_std, output_mean, output_std))

    #test_loader = renormalize_data(test_set, study, output=True, input=True)
    #

    train_set.init_transform = None
    validation_set.init_transform = None
    test_set.init_transform = None
    train_set.input_transform = None
    validation_set.input_transform = None
    test_set.input_transform = None
    train_set.output_transform = None
    validation_set.output_transform = None
    test_set.output_transform = None
    print("train_set.init_transform ", train_set.init_transform)
    print("train_set.input_transform ", train_set.input_transform)
    print("train_set.output_transform ", train_set.output_transform)
    #exit()
    plot_target_data(train_loader, validation_loader, test_loader)
    #exit()

    # train_inputs = []
    # train_outputs = []
    # for data in train_loader:
    #     input, output = data
    #     train_inputs.append(input)
    #     train_outputs.append(output)

    inverse_transform = get_inverse_transform(study, results_dir)
    input_inverse_transform = get_inverse_transform_input(study)

    plot_separate_images = False
    # Disable grad
    with torch.no_grad():
        # Initialize model
        print("model kwargs ", study.best_trial.user_attrs["model_kwargs"])
        model_kwargs = study.best_trial.user_attrs["model_kwargs"]
        # model_kwargs = {'n_conv_layers': 1, 'max_channel': 72, 'pool': 'None', 'pool_size': 0, 'kernel_size': 3,
        #                 'stride': 1, 'pool_stride': 0, 'use_batch_norm': True, 'n_hidden_layers': 1,
        #                 'max_hidden_neurons': 48, 'input_size': 3}

        # if "min_channel" in model_kwargs:
        #     model_kwargs["input_channel"] = 1 #model_kwargs["min_channel"]
        #     model_kwargs["n_output_neurons"] = 1
        #     del model_kwargs["min_channel"]
        model = study.best_trial.user_attrs["model_class"](**model_kwargs)

        # Initialize optimizer
        non_frozen_parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = None
        if len(non_frozen_parameters) > 0:
            optimizer = study.best_trial.user_attrs["optimizer_class"](non_frozen_parameters,
                                                                   **study.best_trial.user_attrs["optimizer_kwargs"])

        print("model path ", model_path)
        if not torch.cuda.is_available():
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(model_path)
        train_loss = checkpoint['train_loss']
        valid_loss = checkpoint['valid_loss']
        #print("checkpoint ", checkpoint)
        #print("convs weight shape ", checkpoint["best_model_state_dict"]['_convs.0.weight'].shape)
        #print("out layer weight shape ", checkpoint["best_model_state_dict"]['_output_layer.weight'].shape)

        print("best val loss: {}".format(np.min(valid_loss)))
        print("best epoch: {}".format(np.argmin(valid_loss)))
        #exit()
        plot_train_valid_loss(train_loss, valid_loss)

        print("checkpoint ", list(checkpoint['best_model_state_dict'].keys()))
        #exit()

        #del checkpoint["best_model_state_dict"]["_output_layer.bias"]
        # del checkpoint["best_model_state_dict"]["_batch_norms_after.0.weight"]
        # del checkpoint["best_model_state_dict"]["_batch_norms_after.0.bias"]
        # del checkpoint["best_model_state_dict"]["_batch_norms_after.1.weight"]
        # del checkpoint["best_model_state_dict"]["_batch_norms_after.1.bias"]

        model.load_state_dict(checkpoint['best_model_state_dict'])
        model.eval()

        if rotate:
            angles = get_rotation_angles(model, validation_loader, inverse_transform)

        if calibrate:
            val_targets, val_predictions, inv_val_targets, inv_val_predictions = [], [], [], []
            val_targets_ch, val_predictions_ch, inv_val_targets_ch, inv_val_predictions_ch = {}, {}, {}, {}

            for i, val_sample in enumerate(validation_loader):
                inputs, targets = val_sample
                inputs = inputs.float()
                if args.cuda and torch.cuda.is_available():
                    inputs = inputs.cuda()

                predictions = model(inputs)

                if len(targets.size()) > 1 and np.sum(targets.size()) / len(targets.size()) != 1:
                    targets = torch.squeeze(targets.float())
                targets_np = targets.numpy()

                if len(predictions.size()) > 1 and np.sum(predictions.size()) / len(predictions.size()) != 1:
                    predictions = torch.squeeze(predictions.float())

                predictions_np = predictions.numpy()

                if inverse_transform is not None:
                    inv_targets = inverse_transform(torch.reshape(targets, (*targets.shape, 1, 1)))
                    inv_predictions = inverse_transform(torch.reshape(predictions, (*predictions.shape, 1, 1)))
                    if len(inv_targets.size()) > 1 and np.sum(inv_targets.size()) / len(inv_targets.size()) != 1:
                        inv_targets = torch.squeeze(inv_targets.float())
                    inv_targets_np = inv_targets.numpy()

                    if len(inv_predictions.size()) > 1 and np.sum(inv_predictions.size()) / len(inv_predictions.size()) != 1:
                        inv_predictions = torch.squeeze(inv_predictions.float())
                    inv_predictions_np = inv_predictions.numpy()

                #print("predictions ", predictions)

                val_predictions.append(predictions_np)
                val_targets.append(targets_np)
                inv_val_predictions.append(inv_predictions_np)
                inv_val_targets.append(inv_targets_np)

                inv_predictions_np = np.reshape(inv_predictions_np, predictions_np.shape)
                inv_targets_np = np.reshape(inv_targets_np, targets_np.shape)

                for ch in range(predictions_np.shape[0]):
                    val_predictions_ch.setdefault(ch, []).append(predictions_np[ch])
                    inv_val_predictions_ch.setdefault(ch, []).append(inv_predictions_np[ch])
                    val_targets_ch.setdefault(ch, []).append(targets_np[ch])
                    inv_val_targets_ch.setdefault(ch, []).append(inv_targets_np[ch])

            val_predictions = np.array(val_predictions)
            val_targets = np.array(val_targets)
            plot_target_prediction(val_predictions, val_targets)
            mse, rmse, nrmse, r2 = get_mse_nrmse_r2(val_targets, val_predictions)

            r2_str = "Validation r2 "
            nrmse_str = "Validation NRMSE"
            for i in range(len(mse)):
                r2_str += " k_{}: {}".format(i, r2[i])
                nrmse_str += " k_{}: {}".format(i, nrmse[i])
            print(r2_str)
            print(nrmse_str)

            #exit()

            inv_val_predictions = np.array(inv_val_predictions)
            inv_val_targets = np.array(inv_val_targets)
            plot_target_prediction(inv_val_predictions, inv_val_targets)
            mse, rmse, nrmse, r2 = get_mse_nrmse_r2(inv_val_targets, inv_val_predictions)
            r2_str = "INV Validation r2 "
            nrmse_str = "INV Validation NRMSE"
            for i in range(len(mse)):
                r2_str += " k_{}: {}".format(i, r2[i])
                nrmse_str += " k_{}: {}".format(i, nrmse[i])
            print(r2_str)
            print(nrmse_str)
            # print("INV Validation r2  k_xx: {}, k_xy: {}, k_yy: {}".format(r2[0], r2[1], r2[2]))
            # print("INV Validation NRMSE  k_xx: {}, k_xy: {}, k_yy: {}".format(nrmse[0], nrmse[1], nrmse[2]))

            #val_predictions_1 = np.array(val_predictions_1)
            #val_targets_1 = np.array(val_targets_1)
            #print("val_predictions_1.shape ", val_predictions_1.shape)
            #print("val_targets_1.shape ", val_targets_1.shape)
            calibrators = []
            inv_calibrators = []
            cal_val_predictions = []
            inv_cal_val_predictions = []
            for ch in range(predictions_np.shape[0]):
                calibrator = get_calibrator()
                calibrators.append(calibrator)
                calibrator_fit(calibrator, val_predictions_ch[ch], val_targets_ch[ch])
                cal_val_predictions.append(torch.tensor(calibrator_predict(calibrator, val_predictions_ch[ch])))

                inv_calibrator = get_calibrator()
                inv_calibrators.append(inv_calibrator)
                calibrator_fit(inv_calibrator, inv_val_predictions_ch[ch], inv_val_targets_ch[ch])
                inv_cal_val_predictions.append(torch.tensor(calibrator_predict(inv_calibrator, inv_val_predictions_ch[ch])))

            # calibrator_0 = get_calibrator()
            # calibrator_fit(calibrator_0, val_predictions_0, val_targets_0)
            # #calibrator_0.fit(val_predictions_0, val_targets_0)
            # cal_val_predictions_0 = torch.tensor(calibrator_predict(calibrator_0, val_predictions_0))
            #
            # calibrator_1 = get_calibrator()
            #
            # # # Fit a logistic regression model to the predicted values
            # # log_reg = LogisticRegression()
            # # log_reg.fit(val_predictions_1.reshape(-1, 1), val_targets_1)
            # #
            # # calibrator_1 = CalibratedClassifierCV(log_reg, cv='prefit')
            # # calibrator_1.fit(val_predictions_1.reshape(-1, 1), val_targets_1)
            #
            # #pred_y = calibrated_clf.predict(model.predict(new_X).reshape(-1, 1))
            #
            # # # Calibrate the neural network predictions using quantile regression
            # # quantiles = [0.01, 0.25, 0.5, 0.75, 0.99]
            # # calibrated_predictions = []
            # # for q in quantiles:
            # #     qr = QuantReg(train_outputs, train_inputs).fit(q=q)
            # #     calibrated_predictions.append(qr.predict())
            # #
            # # # Take the average of the calibrated predictions as the final calibrated prediction
            # # final_predictions = np.mean(calibrated_predictions, axis=0)
            #
            # calibrator_fit(calibrator_1, val_predictions_1, val_targets_1)
            # cal_val_predictions_1 = torch.tensor(calibrator_predict(calibrator_1, val_predictions_1))
            #
            # calibrator_2 =  get_calibrator()
            # calibrator_fit(calibrator_2, val_predictions_2, val_targets_2)
            # cal_val_predictions_2 = torch.tensor(calibrator_predict(calibrator_2, val_predictions_2))


            ###
            # INV
            ###
            # inv_calibrator_0 =  get_calibrator()
            # calibrator_fit(inv_calibrator_0, inv_val_predictions_0, inv_val_targets_0)
            # inv_cal_val_predictions_0 = torch.tensor(calibrator_predict(inv_calibrator_0, inv_val_predictions_0))
            #
            # inv_calibrator_1 = get_calibrator()
            # calibrator_fit(inv_calibrator_1, inv_val_predictions_1, inv_val_targets_1)
            # inv_cal_val_predictions_1 = torch.tensor(calibrator_predict(inv_calibrator_1, inv_val_predictions_1))
            #
            # inv_calibrator_2 = get_calibrator()
            # calibrator_fit(inv_calibrator_2, inv_val_predictions_2, inv_val_targets_2)
            # inv_cal_val_predictions_2 = torch.tensor(calibrator_predict(inv_calibrator_2, inv_val_predictions_2))

            # print("cal val predictions 2 ", cal_val_predictions_2.numpy().shape)
            # exit()
            # cal_val_preds = np.array(cal_val_predictions).T
            # inv_cal_val_preds = np.array(inv_cal_val_predictions).T

            cal_val_preds = np.stack(cal_val_predictions, axis=1)
            inv_cal_val_preds = np.stack(inv_cal_val_predictions, axis=1)

            plot_target_prediction(cal_val_preds, val_targets)
            mse, rmse, nrmse, r2 = get_mse_nrmse_r2(val_targets, cal_val_preds)

            r2_str = "Calibrated Validation r2 "
            nrmse_str = "Calibrated Validation NRMSE"
            for i in range(len(mse)):
                r2_str += " k_{}: {}".format(i, r2[i])
                nrmse_str += " k_{}: {}".format(i, nrmse[i])
            print(r2_str)
            print(nrmse_str)
            #print("Calibrated Validation r2  k_xx: {}, k_xy: {}, k_yy: {}".format(r2[0], r2[1], r2[2]))
            #print("Calibrated Validation NRMSE  k_xx: {}, k_xy: {}, k_yy: {}".format(nrmse[0], nrmse[1], nrmse[2]))

            #inv_cal_val_preds = np.stack((inv_cal_val_predictions_0, inv_cal_val_predictions_1, inv_cal_val_predictions_2), axis=1)
            plot_target_prediction(inv_cal_val_preds, inv_val_targets)
            mse, rmse, nrmse, r2 = get_mse_nrmse_r2(val_targets, cal_val_preds)

            r2_str = "Calibrated INV Validation r2 "
            nrmse_str = "Calibrated INV Validation NRMSE"
            for i in range(len(mse)):
                r2_str += " k_{}: {}".format(i, r2[i])
                nrmse_str += " k_{}: {}".format(i, nrmse[i])
            print(r2_str)
            print(nrmse_str)
            #print("Calibrated INV Validation r2  k_xx: {}, k_xy: {}, k_yy: {}".format(r2[0], r2[1], r2[2]))
            #print("Calibrated INV Validation NRMSE  k_xx: {}, k_xy: {}, k_yy: {}".format(nrmse[0], nrmse[1], nrmse[2]))
            #print("results.shape ", result.shape)
            #exit()

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['best_optimizer_state_dict'])
        epoch = checkpoint['best_epoch']

        print("Best epoch ", epoch)

        print("train loss ", train_loss)
        print("valid loss ", valid_loss)
        print("model training time ", checkpoint["training_time"])

        plot_train_valid_loss(train_loss, valid_loss)

        running_loss, inv_running_loss = 0, 0
        targets_list, predictions_list = [], []
        inv_targets_list, inv_predictions_list = [], []
        pred_evals, tar_evals = [], []
        pred_evecs, tar_evecs = [], []
        square_err_k_xx = []
        square_err_k_xy = []
        square_err_k_yy = []

        nrmse = 0

        # if input_inverse_transform is not None:
        #     import copy
        #     tst_input_transform = copy.deepcopy(test_set.input_transform)
        #     print("test_set.input_transform ", test_set.input_transform)
        #     test_set.init_transform = None
        #     test_set.input_transform = None

        for i, test_sample in enumerate(test_loader): #@TODO: SET TEST LOADER (test_loader)
            inputs, targets = test_sample
            #print("intputs ", inputs)

            inputs = inputs.float()

            if args.cuda and torch.cuda.is_available():
                inputs = inputs.cuda()
            predictions = model(inputs)

            if test_set.init_transform is not None and input_inverse_transform is not None:
                #print("inputs ", inputs)
                #print("test_set._bulk_features_avg ", test_set._bulk_features_avg)
                inv_transformed_inputs = input_inverse_transform(inputs)
                #print("inv tra inputs ", inv_transformed_inputs)

                inv_input_avg = test_set._bulk_features_avg #torch.mean(inv_transformed_inputs)
                #print("inv input avg ", inv_input_avg)
                #exit()

                #inv_transformed_inputs /= inv_input_avg
                #print("inv transformed inputs ", inv_transformed_inputs)
                #print("inv transformed inputs shape", inv_transformed_inputs.shape)

                #inv_transformed_inputs_shape = inv_transformed_inputs.shape
                #inputs = tst_input_transform(torch.squeeze(inv_transformed_inputs))

                #inputs = torch.reshape(inputs, inv_transformed_inputs_shape)
                #print("final inputs ", inputs)
                #print("final inputs shape ", inputs.shape)
                #exit()
            #print("targets size ", targets.size())

            if len(targets.size()) > 1 and np.sum(targets.size())/len(targets.size()) != 1:
                targets = torch.squeeze(targets.float())

            # print("predictions ", predictions)
            # print("targets ", targets)
            #print("targets_np.shape ", targets_np.shape)

            # if calibrate:
            #     predictions[1] = torch.tensor(calibrator_1.predict(targets[1].cpu().numpy()))
            #     print("predictins.shape ", predictions.shape)
            #     #predictions[:, 1] = targets[1] * 0.32867652
            #     #print("calibrated output ", calibrated_outputs)
            #
            #     predictions = torch.squeeze(predictions).cpu()
            #     predictions_np = predictions.cpu().numpy()
            #     predictions_list.append(predictions_np)
            #
            # else:
            if len(predictions.size()) > 1 and np.sum(predictions.size())/len(predictions.size()) != 1:
                #print("squeeze ")
                predictions = torch.squeeze(predictions.float())

            #print("squeeze prediction size ", predictions.size())

            if rotate:
                predictions = rotate_by_angle(targets, predictions, angles)

            predictions = predictions.cpu()
            predictions_np = predictions.cpu().numpy()

            targets_np = targets.numpy()
            targets_list.append(targets_np)
            predictions_list.append(predictions_np)

            # square_err_k_xx.append(targets_np[0])
            # square_err_k_xy.append(targets_np[1])
            # square_err_k_yy.append(targets_np[2])

            #print("targets_lists[0][0] ", targets_list[0][0])
            if "loss_fn_name" in study.best_trial.user_attrs:
                loss_fn = study.best_trial.user_attrs["loss_fn_name"]()
            if "loss_fn" in study.best_trial.user_attrs:
                loss_fn = study.best_trial.user_attrs["loss_fn"]

            # print("predictions.device ", predictions.device)
            # print("targets.device", targets.device)

            #print("predictions.shape ", predictions.shape)
            #print("targets.shape ", targets.shape)

            loss = loss_fn(predictions, targets)
            running_loss += loss

            inv_targets = targets
            inv_predictions = predictions
            if inverse_transform is not None:
                inv_targets = inverse_transform(torch.reshape(targets, (*targets.shape, 1, 1)))
                inv_predictions = inverse_transform(torch.reshape(predictions, (*predictions.shape, 1, 1)))

                if test_set.init_transform is not None and input_inverse_transform is not None:
                    # print("inv targets ", inv_targets)
                    # print("inv prediction ", inv_predictions)
                    # print("inv input avg ", inv_input_avg)
                    inv_predictions *= inv_input_avg
                    inv_targets *= inv_input_avg

                    # print("mult inv targets ", inv_targets)
                    # print("mult inv prediction ", inv_predictions)
                    #
                    #
                    # print("inv predictins mult ", inv_predictions)
                    # exit()

                inv_targets = np.reshape(inv_targets, targets.shape)
                inv_predictions = np.reshape(inv_predictions, predictions.shape)

                # if len(inv_targets.size()) > 1 and np.sum(inv_targets.size())/len(inv_targets.size()) != 1:
                #     inv_targets = torch.squeeze(inv_targets)
                #
                # if len(inv_predictions.size()) > 1 and np.sum(inv_predictions.size())/len(inv_predictions.size()) != 1:
                #     inv_predictions = torch.squeeze(inv_predictions)

            inv_running_loss += loss_fn(inv_predictions, inv_targets)

            inv_targets_list.append(inv_targets.numpy())
            inv_predictions_list.append(inv_predictions.numpy())


            # pred_eval, pred_evec = get_eigendecomp(inv_predictions.numpy())
            # tar_eval, tar_evec = get_eigendecomp(inv_targets.numpy())
            #
            # pred_evals.append(pred_eval)
            # pred_evecs.append(pred_evec)
            #
            # tar_evals.append(tar_eval)
            # tar_evecs.append(tar_evec)

            # if i % 50 == 9:
            #     plot_tensors(inv_predictions.numpy(), inv_targets.numpy(), label="test_sample_{}_cal".format(i),
            #              plot_separate_images=plot_separate_images)

            # if i % 10 == 9:
            #     plot_tensors(inv_predictions.numpy(), inv_targets.numpy(), label="test_sample_{}".format(i),
            #                  plot_separate_images=plot_separate_images)

        if calibrate:
            #calibs = [calibrator_0, calibrator_1, calibrator_2]
            predictions_list = calibrate_data(predictions_list, calibrators)
            #inv_calibs = [inv_calibrator_0, inv_calibrator_1, inv_calibrator_2]
            inv_predictions_list = calibrate_data(inv_predictions_list, inv_calibrators)

        mse, rmse, nrmse, r2 = get_mse_nrmse_r2(targets_list, predictions_list)
        inv_mse, inv_rmse, inv_nrmse, inv_r2 = get_mse_nrmse_r2(inv_targets_list, inv_predictions_list)

        test_loss = running_loss / (i + 1)
        inv_test_loss = inv_running_loss / (i + 1)

        plot_target_prediction(np.array(targets_list), np.array(predictions_list), "preprocessed_")
        plot_target_prediction(np.array(inv_targets_list), np.array(inv_predictions_list))

        print("epochs: {}, train loss: {}, valid loss: {}, test loss: {}, inv test loss: {}".format(epoch,
                                                                                                    train_loss,
                                                                                                    valid_loss,
                                                                                                    test_loss,
                                                                                                    inv_test_loss))
        mse_str, inv_mse_str = "MSE", "Original data MSE"
        r2_str, inv_r2_str = "R2", "Original data R2"
        rmse_str, inv_rmse_str = "RMSE", "Original data RMSE"
        nrmse_str, inv_nrmse_str = "NRMSE", "Original data NRMSE"
        for i in range(len(mse)):
            mse_str += " k_{}: {}".format(i, mse[i])
            r2_str += " k_{}: {}".format(i, r2[i])
            rmse_str += " k_{}: {}".format(i, rmse[i])
            nrmse_str += " k_{}: {}".format(i, nrmse[i])

            inv_mse_str += " k_{}: {}".format(i, inv_mse[i])
            inv_r2_str += " k_{}: {}".format(i, inv_r2[i])
            inv_rmse_str += " k_{}: {}".format(i, inv_rmse[i])
            inv_nrmse_str += " k_{}: {}".format(i, inv_nrmse[i])

            # print("MSE k_xx: {}, k_xy: {}, k_yy: {}".format(mse[0], mse[1], mse[2]))
            # print("R2 k_xx: {}, k_xy: {}, k_yy: {}".format(r2[0], r2[1], r2[2]))
            # print("RMSE  k_xx: {}, k_xy: {}, k_yy: {}".format(rmse[0], rmse[1], rmse[2]))
            # print("NRMSE  k_xx: {}, k_xy: {}, k_yy: {}".format(nrmse[0], nrmse[1], nrmse[2]))

            # print("Original data MSE k_xx: {}, k_xy: {}, k_yy: {}".format(inv_mse[0], inv_mse[1], inv_mse[2]))
            # print("Original data R2 k_xx: {}, k_xy: {}, k_yy: {}".format(inv_r2[0], inv_r2[1], inv_r2[2]))
            # print("Original data RMSE k_xx: {}, k_xy: {}, k_yy: {}".format(inv_rmse[0], inv_rmse[1], inv_rmse[2]))
            # print("Original data NRMSE  k_xx: {}, k_xy: {}, k_yy: {}".format(inv_nrmse[0], inv_nrmse[1], inv_nrmse[2]))

        print(mse_str)
        print(r2_str)
        print(rmse_str)
        print(nrmse_str)

        print(inv_mse_str)
        print(inv_r2_str)
        print(inv_rmse_str)
        print(inv_nrmse_str)


def load_study(results_dir):
    #print("os.path.join(results_dir, study.pkl) ", os.path.join(results_dir, "study.pkl"))
    # path_to_file = os.path.join(results_dir, "pickle_study.pkl")
    # with open(path_to_file, 'rb') as f:
    #     data = pickle.load(f)
    # print("data ", data)
    # exit()
    # # model = pickle.load(path_to_file)
    # model = pickle.load(open(path_to_file, "rb"))
    study = joblib.load(os.path.join(results_dir, "study.pkl"))

    # with open(os.path.join(results_dir, "pickle_study.pkl"), "wb") as f:
    #     pickle.dump(study, f)
    # exit()
    #study = pickle.load(os.path.join(results_dir, "study.pkl"))
    print("Best trial until now:")
    print(" Value: ", study.best_trial.value)
    print(" Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    return study

def compare_trials(study):
    df = study.trials_dataframe()
    print("df ", df)

    df.to_csv("trials.csv")
    # Import pandas package
    import pandas as pd

    df_duration = df.sort_values("duration")

    print("df_duration ", df_duration)
    # iterating the columns
    for col in df.columns:
        print(col)
    exit()
    fastest = None
    for trial in study.trials:
        print("datetime start ", trial["datetime_start"])
        print("trial ", trial)

    exit()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', help='results directory')
    parser.add_argument("-c", "--cuda", default=False, action='store_true', help="use cuda")
    args = parser.parse_args(sys.argv[1:])

    study = load_study(args.results_dir)

    #@TODO: RM ASAP
    print("study attrs ", study.user_attrs)
    #study.user_attrs["output_log"] = True
    #study.set_user_attr("output_log", True)
    print("study attrs ", study.user_attrs)
    #compare_trials(study)

    load_models(args, study)


