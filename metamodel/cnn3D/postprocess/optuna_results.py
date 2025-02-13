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
from metamodel.cnn3D.datasets.dfm3d_dataset import DFM3DDataset
#from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from metamodel.cnn.visualization.visualize_tensor import plot_tensors
from metamodel.cnn.visualization.visualize_data import plot_target_prediction, plot_train_valid_loss
from metamodel.cnn.models.auxiliary_functions import exp_data, get_eigendecomp, get_mse_nrmse_r2, get_mean_std, log_data, exp_data,\
    quantile_transform_fit, QuantileTRF, NormalizeData, log_all_data, init_norm, get_mse_nrmse_r2_eigh_3D, log10_data, log10_all_data, power_10_all_data, power_10_data, CorrelatedOutputLoss
from metamodel.cnn.models.train_pure_cnn_optuna import prepare_dataset
#from sklearn.isotonic import IsotonicRegression
#from statsmodels.regression.quantile_regression import QuantReg
#from sklearn.linear_model import LogisticRegression
#from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import QuantileTransformer, RobustScaler

#os.environ["CUDA_VISIBLE_DEVICES"]=""


def preprocess_dataset(config, user_attrs, data_dir, results_dir=None):
    print("user attrs ", user_attrs)
    # output_file_name = "output_tensor.npy"
    # if "vel_avg" in config and config["vel_avg"]:
    #     output_file_name = "output_vel_avg.npy"

    # ===================================
    # Get mean and std for each channel
    # ===================================
    input_mean, output_mean, input_std, output_std = 0, 0, 1, 1
    data_normalizer = NormalizeData()

    # n_train_samples = None
    # if "n_train_samples" in config and config["n_train_samples"] is not None:
    #     n_train_samples = config["n_train_samples"]

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

    print("data dir ", data_dir)
    print("results dir ", results_dir)

    # Standardize output
    if config["log_output"]:
        output_transformations.append(transforms.Lambda(log_data))
    elif config["log10_output"]:
        output_transformations.append(transforms.Lambda(log10_data))
    elif config["log10_all_output"]:
        output_transformations.append(transforms.Lambda(log10_all_data))
    elif config["log_all_output"]:
        output_transformations.append(transforms.Lambda(log_all_data))
    if config["normalize_output"]:
        if "normalize_output_indices" in config:
            data_normalizer.input_indices = config["normalize_output_indices"]

        print("output mean ", user_attrs["output_mean"])
        print("output std ",  user_attrs["output_std"])
        data_normalizer.output_mean = user_attrs["output_mean"]
        data_normalizer.output_std = user_attrs["output_std"]
        output_transformations.append(data_normalizer.normalize_output)

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
    train_set = DFM3DDataset(zarr_path=data_dir,
                            init_transform=data_init_transform,
                            input_transform=data_input_transform,
                            output_transform=data_output_transform,
                            input_channels=config["input_channels"] if "input_channels" in config else None,
                            output_channels=config["output_channels"] if "output_channels" in config else None,
                            fractures_sep=config["fractures_sep"] if "fractures_sep" in config else False,
                            cross_section=config["cross_section"] if "cross_section" in config else False,
                            init_norm_use_all_features=config[
                                "init_norm_use_all_features"] if "init_norm_use_all_features" in config else False,
                            mode="train", train_size=0, val_size=0, test_size=config["n_test_samples"])

    validation_set = DFM3DDataset(zarr_path=data_dir,
                             init_transform=data_init_transform,
                             input_transform=data_input_transform,
                             output_transform=data_output_transform,
                             input_channels=config["input_channels"] if "input_channels" in config else None,
                             output_channels=config["output_channels"] if "output_channels" in config else None,
                             fractures_sep=config["fractures_sep"] if "fractures_sep" in config else False,
                             cross_section=config["cross_section"] if "cross_section" in config else False,
                             init_norm_use_all_features=config[
                                 "init_norm_use_all_features"] if "init_norm_use_all_features" in config else False,
                             mode="val", train_size=0, val_size=0, test_size=config["n_test_samples"])

    test_set = DFM3DDataset(zarr_path=data_dir,
                         init_transform=data_init_transform,
                         input_transform=data_input_transform,
                         output_transform=data_output_transform,
                         input_channels=config["input_channels"] if "input_channels" in config else None,
                         output_channels=config["output_channels"] if "output_channels" in config else None,
                         fractures_sep=config["fractures_sep"] if "fractures_sep" in config else False,
                         cross_section=config["cross_section"] if "cross_section" in config else False,
                         init_norm_use_all_features=config["init_norm_use_all_features"] if "init_norm_use_all_features" in config else False,
                         mode="test", train_size=0, val_size=0, test_size=config["n_test_samples"])

    #dataset.shuffle(config["seed"])
    #print("len dataset ", len(dataset))

    # if n_train_samples is None:
    #     n_train_samples = int(len(dataset) * config["train_samples_ratio"])
    #
    # n_train_samples = np.min([n_train_samples, int(len(dataset) * config["train_samples_ratio"])])
    #
    # train_val_set = dataset[:n_train_samples]
    # train_set = train_val_set[:-int(n_train_samples * config["val_samples_ratio"])]
    # validation_set = train_val_set[-int(n_train_samples * config["val_samples_ratio"]):]
    #
    # if "n_test_samples" in config and config["n_test_samples"] is not None:
    #     n_test_samples = config["n_test_samples"]
    #     test_set = dataset[-n_test_samples:]
    # else:
    #     test_set = dataset[n_train_samples:]

    print("LEN train set: {}, val set: {}, test set: {}".format(len(train_set), len(validation_set), len(test_set)))

    return train_set, validation_set, test_set

    # print("len(trainset): {}, len(valset): {}, len(testset): {}".format(len(train_set), len(validation_set),
    #                                                                     len(test_set)))


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
    # train_set = joblib.load(os.path.join(results_dir, "train_dataset.pkl"))
    # val_set = joblib.load(os.path.join(results_dir, "val_dataset.pkl"))
    # test_set = joblib.load(os.path.join(results_dir, "test_dataset.pkl"))
    #
    # # if os.path.exists(os.path.join(results_dir, "output_transform.pkl")) or os.path.exists(os.path.join(results_dir, "input_transform.pkl")):
    # #     quantile_trf_obj = QuantileTRF()
    # #     if os.path.exists(os.path.join(results_dir, "output_transform.pkl")):
    # #         out_trans_obj = joblib.load(os.path.join(results_dir, "output_transform.pkl"))
    # #         quantile_trf_obj.quantile_trfs_out = out_trans_obj
    # #         print("out trans obj ", out_trans_obj)
    # #     if os.path.exists(os.path.join(results_dir, "input_transform.pkl")):
    # #         in_trans_obj = joblib.load(os.path.join(results_dir, "input_transform.pkl"))
    # #         quantile_trf_obj.quantile_trfs_in = in_trans_obj
    #
    # return train_set, val_set, test_set

    #data_dir = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_3/test_data/MLMC-DFM_3D_n_voxels_64/samples_data.zarr"
    data_dir = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_3/train_data/dataset_2/sub_dataset_2_1000.zarr"
    data_dir = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_3/cond_nearest_interp/dataset_2/train_data/sub_dataset_2_1000.zarr"

    #data_dir = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_3/cond_nearest_interp/test_data/samples_data.zarr"

    data_dir = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_3/cond_nearest_interp/test_data/samples_data.zarr"

    data_dir = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_3/cond_nearest_interp/test_data/MLMC-DFM_3D_n_voxels_64_n/samples_data.zarr"

    data_dir = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_5/cond_nearest_interp/test_data/MLMC-DFM_3D_n_voxels_64_500_frac/samples_data.zarr"

    #data_dir = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_5/cond_nearest_interp/test_data/samples_data.zarr"

    #data_dir = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_5/cond_nearest_interp/test_data/MLMC-DFM_3D_n_voxels_64_save_bulk_avg/samples_data.zarr"

    data_dir = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_5/cond_nearest_interp/test_data/MLMC-DFM_3D_n_voxels_64_save_bulk_avg_n_frac_2500/samples_data.zarr"

    data_dir = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_3/cond_nearest_interp/dataset_2_other_method/sub_dataset_2_2500.zarr"

    data_dir = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_3/cond_nearest_interp/test_data/MLMC-DFM_3D_n_voxels_64_save_bulk_avg/samples_data.zarr"

    data_dir = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_5/cond_nearest_interp/test_data/fractures_diag_cond/MLMC-DFM_3D_n_voxels_64_save_bulk_avg/samples_data.zarr"

    #data_dir = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_3/cond_nearest_interp/train_data/dataset_2_save_bulk_avg_n_frac_2500_cl_0_10_25_seed_merge/validation_samples.zarr"

    #data_dir = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_5/cond_nearest_interp/test_data/MLMC-DFM_3D_n_voxels_64_save_bulk_avg_disp_0/samples_data.zarr"
    #data_dir = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_5/cond_nearest_interp/test_data/MLMC-DFM_3D_n_voxels_64_hom_samples/merged.zarr"

    config = {#"num_epochs": trials_config["num_epochs"],
              "batch_size_train": 32,
              # "batch_size_test": 250,
              "n_train_samples": 5,#study.user_attrs["n_train_samples"],
              "n_test_samples": 3500, #2500, #study.user_attrs["n_test_samples"],
              "train_samples_ratio": 0.1,
              "val_samples_ratio":  0.2,
              "print_batches": 10,
              "init_norm": study.user_attrs["init_norm"],
              "log_input": study.user_attrs["input_log"],
              "normalize_input": study.user_attrs["normalize_input"],
              "log_output": study.user_attrs["output_log"],
              # "log10_all_output": study.user_attrs["log10_all_output"],
              # "log_all_output": study.user_attrs["log10_all_output"],
              "normalize_output": study.user_attrs["normalize_output"],
              "init_norm_use_all_features":  study.user_attrs["init_norm_use_all_features"] if "init_norm_use_all_features" in study.user_attrs else False,
              #"output_transform": study.user_attrs["output_transform"],
              "cross_section": False,
              "seed": 12345}

    #@TODO: rm ASAP
    #config["output_channels"] = [0,1,2]

    if "output_channels" in study.user_attrs:
        config["output_channels"] = study.user_attrs["output_channels"]

    if "input_channels" in study.user_attrs:
        config["input_channels"] = study.user_attrs["input_channels"]

    if "output_transform" in study.user_attrs:
        config["output_transform"] = study.user_attrs["output_transform"]

    if "log10_output" in study.user_attrs:
        config["log10_output"] = study.user_attrs["log10_output"]
    if "log10_all_output" in study.user_attrs:
        config["log10_all_output"] = study.user_attrs["log10_all_output"]
    if "log_all_output" in study.user_attrs:
        config["log_all_output"] = study.user_attrs["log_all_output"]

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

        # print("mean shape ", mean.shape)
        # print("std shape ", std.shape)
        # print("zeros mean shape ", zeros_mean.shape)
        # print("ones std shape ", ones_std.shape)

        zeros_mean = zeros_mean.reshape(mean.shape)
        ones_std = ones_std.reshape(std.shape)

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
        elif "log10_output" in study.user_attrs and study.user_attrs["log10_output"]:
            print("log10_output to transform list")
            transforms_list.append(transforms.Lambda(power_10_data()))
        elif "log10_all_output" in study.user_attrs and study.user_attrs["log10_all_output"]:
            print("log10_all_output to transform list")
            #transforms_list.append(transforms.Lambda(exp_data))
            transforms_list.append(transforms.Lambda(power_10_all_data))
        elif "log_all_output" in study.user_attrs and study.user_attrs["log_all_output"]:
            print("log_all_output to transform list")
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
    test_targets = []

    # k_xx_list_input = []
    # k_xy_list_input = []
    # k_yy_list_input = []
    # for i, test_sample in enumerate(test_loader):
    #     inputs, targets = test_sample
    #     inputs = np.squeeze(inputs.numpy())
    #     print("inputs ", inputs)
    #     if i > 50:
    #         break
    #     print("inputs[0].ravel() ", inputs[0].ravel())
    #     k_xx_list_input.extend(inputs[0].ravel())
    #     k_xy_list_input.extend(inputs[1].ravel())
    #     k_yy_list_input.extend(inputs[2].ravel())
    #
    # print("kxx shape ", np.array(k_xx_list_input).shape)
    # plt.hist(k_xx_list_input, bins=25, color="red", label="input k_xx", density=True)
    # #plt.xlim([-0.001, 1])
    # plt.legend()
    # plt.show()
    #
    # plt.hist(k_xy_list_input, bins=60, color="red", label="input k_xy", density=True)
    # plt.legend()
    # plt.show()
    #
    # plt.hist(k_yy_list_input, bins=60, color="red", label="input k_yy", density=True)
    # plt.legend()
    # plt.show()

    for i, test_sample in enumerate(test_loader): #@TODO: use train_loader again
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

    n_channels = 6
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
        plt.hist(k_train_targets, bins=60, color="red", label="train, ch: {}".format(i), alpha=0.5, density=density)

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


# Function to visualize feature maps
# Function to register a hook to capture feature maps
def get_feature_maps(model, layer):
    feature_maps = []

    def hook(module, input, output):
        feature_maps.append(output)

    print("dict([*model.named_modules()]) ", dict([*model.named_modules()]))
    #dict([*model.named_modules()])[layer_name]  # Access the layer by name
    layer.register_forward_hook(hook)

    return feature_maps

def visualize_feature_maps(model, data_loader, num_images=5):
    import matplotlib.pyplot as plt

    layer = model._convs[3]

    # Get feature maps
    feature_maps = get_feature_maps(model, layer)

    feature_index = 1

    # Pass some sample inputs through the model
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            if i >= num_images:
                break
            _ = model(inputs)

            # Get the last captured feature maps
            fmap = feature_maps[-1][0]  # First image in the batch

            print("fmap shape ", fmap.shape)

            # Select the specific feature map
            feature_map = fmap[feature_index].cpu().numpy()

            # Prepare coordinates for 3D plotting
            depth, height, width = feature_map.shape
            x = np.arange(depth)
            y = np.arange(height)
            z = np.arange(width)
            x, y, z = np.meshgrid(x, y, z, indexing='ij')

            # Create a 3D plot
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x.flatten(), y.flatten(), z.flatten(), c=feature_map.flatten(), cmap='coolwarm', alpha=0.75)
            ax.set_xlabel('Depth')
            ax.set_ylabel('Height')
            ax.set_zlabel('Width')
            ax.set_title(f'3D Feature Map Visualization from Layer: {layer}, Feature: {feature_index}')
            plt.show()

            from mayavi import mlab
            # Create a Mayavi figure
            mlab.figure(size=(800, 800), bgcolor=(1, 1, 1))

            # Create a 3D volume visualization
            src = mlab.pipeline.scalar_field(feature_map)
            mlab.pipeline.volume(src, vmin=feature_map.min(), vmax=feature_map.max())
            mlab.title(f'3D Feature Map from Layer: {layer}, Feature: {feature_index}')
            mlab.show()

            # # Plot feature maps
            # num_features = fmap.shape[0]  # Number of feature maps
            # num_cols = 8  # Number of columns for visualization
            # num_rows = (num_features + num_cols - 1) // num_cols  # Rows needed
            #
            # plt.figure(figsize=(15, 15))
            # for j in range(num_features):
            #     plt.subplot(num_rows, num_cols, j + 1)
            #     plt.imshow(fmap[j].cpu().numpy(), cmap='viridis')
            #     plt.axis('off')
            # plt.suptitle(f'Feature Maps from Layer: {layer}', fontsize=16)
            # plt.show()


def load_models(args, study):
    calibrate = False
    rotate = False
    results_dir = args.results_dir
    model_path = get_saved_model_path(results_dir, study.best_trial)
    #model_path = "/home/martin/Documents/MLMC-DFM/optuna_runs/karolina/MLMC-DFM_general_dataset_2/pooling/mase_loss/fr_div_0_1_exp_rho_2_5_3/seed_36974/trial_0_losses_model_cnn_net"
    #model_path = "/home/martin/Documents/MLMC-DFM/optuna_runs/3D_cnn/lumi/cond_frac_1_3/cond_nearest_interp/dataset_2_save_bulk_avg_n_frac_2500_cl_0_10_25_seed_merge_new/init_norm/exp_12_14/seed_12345/trial_0_losses_model_cnn_net_best_59"
    print("model path ", model_path)
    print("model kwargs ", study.best_trial.user_attrs["model_kwargs"])

    train_set, validation_set, test_set = load_dataset(results_dir, study)

    #@TODO: RM ASAP
    # test_set = test_set[:100]

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

    #train_set.init_transform = None
    #validation_set.init_transform = None
    #test_set.init_transform = None
    # train_set.input_transform = None
    # validation_set.input_transform = None
    # test_set.input_transform = None
    # train_set.output_transform = None
    # validation_set.output_transform = None
    # test_set.output_transform = None
    # print("train_set.init_transform ", train_set.init_transform)
    # print("train_set.input_transform ", train_set.input_transform)
    # print("train_set.output_transform ", train_set.output_transform)
    # #exit()
    #plot_target_data(train_loader, validation_loader, test_loader)
    #exit()

    # train_inputs = []
    # train_outputs = []
    # for data in train_loader:
    #     input, output = data
    #     train_inputs.append(input)
    #     train_outputs.append(output)

    inverse_transform = get_inverse_transform(study, results_dir)
    input_inverse_transform = get_inverse_transform_input(study)

    #exit()

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

        #model_path = "/home/martin/Documents/MLMC-DFM/optuna_runs/metacentrum/MLMC-DFM_general_dataset_2/pooling/eigh_loss/fr_div_0_1_exp_6/seed_36974/trial_0_losses_model_cnn_net"
        if not torch.cuda.is_available():
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(model_path)
        train_loss = checkpoint['train_loss']
        valid_loss = checkpoint['valid_loss']
        #print("checkpoint ", checkpoint)
        #print("convs weight shape ", checkpoint["best_model_state_dict"]['_convs.0.weight'].shape)
        #print("out layer weight shape ", checkpoint["best_model_state_dict"]['_output_layer.weight'].shape)

        print("valid loss ", valid_loss)

        print("best val loss: {}".format(np.min(valid_loss)))
        print("best epoch: {}".format(np.argmin(valid_loss)))
        #exit()
        #plot_train_valid_loss(train_loss, valid_loss)

        #visualize_feature_maps(model, test_loader)

        print("checkpoint ", list(checkpoint['best_model_state_dict'].keys()))
        #print("_output_layer.bias ", checkpoint['best_model_state_dict']['_output_layer.bias'])

        #del checkpoint["best_model_state_dict"]["_output_layer.bias"]
        # del checkpoint["best_model_state_dict"]["_batch_norms_after.0.weight"]
        # del checkpoint["best_model_state_dict"]["_batch_norms_after.0.bias"]
        # del checkpoint["best_model_state_dict"]["_batch_norms_after.1.weight"]
        # del checkpoint["best_model_state_dict"]["_batch_norms_after.1.bias"]

        model.load_state_dict(checkpoint['best_model_state_dict'])
        model.eval()

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


        covariance_matrix = CorrelatedOutputLoss.calculate_cov(test_loader)
        print("covariance matrix ", covariance_matrix)

        inv_covariance_matrix = np.linalg.inv(covariance_matrix.cpu())
        print("inv_covariance_matrix ", inv_covariance_matrix)

        np.save("covariance_matrix ", covariance_matrix.cpu())

        n_wrong = 0
        wrong_targets = []
        wrong_predictions = []
        for i, test_sample in enumerate(test_loader): #@TODO: SET TEST LOADER (test_loader)
            inputs, targets = test_sample
            #print("targets ", targets)
            #print("intputs ", inputs.shape)

            inputs = inputs.float()

            #print("input inverse transform ", input_inverse_transform)

            if args.cuda and torch.cuda.is_available():
                inputs = inputs.cuda()
            predictions = model(inputs)



            #print("predictions ", predictions.shape)

            if test_set.init_transform is not None and input_inverse_transform is not None:
                #print("inputs ", inputs)
                #print("test_set._bulk_features_avg ", test_set._bulk_features_avg)
                #print("inputs to inverse transform shape ", inputs.shape)
                inv_transformed_inputs = input_inverse_transform(inputs)
                #print("inv tra inputs ", inv_transformed_inputs)

                inv_input_avg = test_set._bulk_features_avg #torch.mean(inv_transformed_inputs)
                #inv_input_avg = train_set._bulk_features_avg  # torch.mean(inv_transformed_inputs)
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

            #print("targets ", targets)
            #print("predictions ", predictions)
            #print("targets_np.shape ", targets_np.shape)

            # else:
            if len(predictions.size()) > 1 and np.sum(predictions.size())/len(predictions.size()) != 1:
                #print("squeeze ")
                predictions = torch.squeeze(predictions.float())

            #print("squeeze prediction size ", predictions.size())

            predictions = predictions.cpu()
            predictions_np = predictions.cpu().numpy()

            # print("predictions np shape ", predictions_np.shape)
            # predictions_np[0] += 0.1
            # predictions_np[2] += 0.1

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

            #loss = loss_fn(predictions, targets)
            #running_loss += loss

            inv_targets = targets
            inv_predictions = predictions
            if inverse_transform is not None:
                #print("tragets shape ", targets.shape)

                try:
                    inv_targets = inverse_transform(torch.reshape(targets, (*targets.shape, 1, 1)))
                    inv_predictions = inverse_transform(torch.reshape(predictions, (*predictions.shape, 1, 1)))
                except:
                    print("continue")
                    continue

                # print("inv targets ", inv_targets)
                # print("inv predictions", inv_predictions)

                if test_set.init_transform is not None and input_inverse_transform is not None:
                    # print("inv targets ", inv_targets)
                    # print("inv prediction ", inv_predictions)
                    #print("inv input avg ", inv_input_avg)
                    inv_predictions *= inv_input_avg
                    inv_targets *= inv_input_avg

                    #print("mult inv targets ", inv_targets)
                    #print("mult inv prediction ", inv_predictions)
                    # #
                    #
                    # print("inv predictins mult ", inv_predictions)
                    # exit()

                inv_targets = np.reshape(inv_targets, targets.shape)
                inv_predictions = np.reshape(inv_predictions, predictions.shape)

                # if np.abs(np.log(inv_targets[0]) - np.log(inv_predictions[0])) > 0.145:
                #     n_wrong += 1
                #     wrong_targets.append(inv_targets.numpy())
                #     wrong_predictions.append(inv_predictions.numpy())
                #     print("np.log(inv_targets): {}, np.log(inv_predictions): {}".format(np.log(inv_targets), np.log(inv_predictions)))
                #
                #     continue

                # print("inv targets init norm ", inv_targets)
                # print("inv predictions init norm", inv_predictions)

                # if np.abs(inv_predictions[1]) > inv_predictions[0] or np.abs(inv_predictions[1]) > inv_predictions[2]:
                #     off_diag = (inv_predictions[0] + inv_predictions[2]) / 2 / 10
                #
                #     if inv_predictions[1] < 0:
                #         off_diag = -off_diag
                #
                #     inv_predictions[1] = off_diag
                #     #print("adjusted inv predictions ", inv_predictions)


                # if len(inv_targets.size()) > 1 and np.sum(inv_targets.size())/len(inv_targets.size()) != 1:
                #     inv_targets = torch.squeeze(inv_targets)
                #
                # if len(inv_predictions.size()) > 1 and np.sum(inv_predictions.size())/len(inv_predictions.size()) != 1:
                #     inv_predictions = torch.squeeze(inv_predictions)

            #inv_running_loss += loss_fn(inv_predictions, inv_targets)

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


        inv_targets_arr = np.array(inv_targets_list)
        inv_predictions_arr = np.array(inv_predictions_list)

        print("inv targets arr shape", inv_targets_arr.shape)

        #inv_predictions_arr /= 11.11
        #inv_predictions_arr *= 9.642726938
        #inv_predictions_arr *= 1.1
        #inv_predictions_arr /= 0.9333

        predictions_list = np.array(predictions_list)
        #predictions_list += 0.1

        #inv_targets_arr, inv_predictions_arr = _remove_outliers_iqr(inv_targets_arr, inv_predictions_arr)

        np.savez("inv_targets_arr", data=inv_targets_arr)
        np.savez("inv_predictions_arr", data=inv_predictions_arr)

        np.savez("targets_list", data=targets_list)
        np.savez("predictions_list", data=predictions_list)
        exit()

        mse, rmse, nrmse, r2 = get_mse_nrmse_r2(targets_list, predictions_list)
        inv_mse, inv_rmse, inv_nrmse, inv_r2 = get_mse_nrmse_r2(inv_targets_arr, inv_predictions_arr)

        test_loss = running_loss / (i + 1)
        inv_test_loss = inv_running_loss / (i + 1)


        print("epochs: {}, train loss: {}, valid loss: {}, test loss: {}, inv test loss: {}".format(epoch,
                                                                                                    train_loss,
                                                                                                    valid_loss,
                                                                                                    test_loss,
                                                                                                inv_test_loss))

        titles = ['k_xx', 'k_yy', 'k_zz', 'k_yz', 'k_xz', 'k_xy']
        x_labels = [r'$log(k_{xx})$', r'$log(k_{yy})$', r'$log(k_{zz})$', r'$k_{yz}$', r'$k_{xz}$', r'$k_{xy}$']


        mse_str, inv_mse_str = "MSE", "Original data MSE"
        r2_str, inv_r2_str = "R2", "Original data R2"
        rmse_str, inv_rmse_str = "RMSE", "Original data RMSE"
        nrmse_str, inv_nrmse_str = "NRMSE", "Original data NRMSE"
        for i in range(len(mse)):
            mse_str += " {}: {}".format(titles[i], mse[i])
            r2_str += " {}: {}".format(titles[i], r2[i])
            rmse_str += " {}: {}".format(titles[i], rmse[i])
            nrmse_str += " {}: {}".format(titles[i], nrmse[i])

            inv_mse_str += " {}: {}".format(titles[i], inv_mse[i])
            inv_r2_str += " {}: {}".format(titles[i], inv_r2[i])
            inv_rmse_str += " {}: {}".format(titles[i], inv_rmse[i])
            inv_nrmse_str += " {}: {}".format(titles[i], inv_nrmse[i])

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

        print("mean R2: {}, NRMSE: {}".format(np.mean(r2), np.mean(nrmse)))

        print(inv_mse_str)
        print(inv_r2_str)
        print(inv_rmse_str)
        print(inv_nrmse_str)

        print("Original data mean R2: {}, NRMSE: {}".format(np.mean(inv_r2), np.mean(inv_nrmse)))

        import copy
        log_inv_targets_arr = copy.deepcopy(inv_targets_arr)
        log_inv_predictions_arr = copy.deepcopy(inv_predictions_arr)

        log_inv_targets_arr[:, 0] = np.log10(log_inv_targets_arr[:, 0])
        log_inv_targets_arr[:, 2] = np.log10(log_inv_targets_arr[:, 2])

        log_inv_predictions_arr[:, 0] = np.log10(log_inv_predictions_arr[:, 0])
        log_inv_predictions_arr[:, 2] = np.log10(log_inv_predictions_arr[:, 2])

        log_inv_mse, log_inv_rmse, log_inv_nrmse, log_inv_r2 = get_mse_nrmse_r2(log_inv_targets_arr, log_inv_predictions_arr)

        mse_str, inv_mse_str = "MSE", "LOG Original data MSE"
        r2_str, inv_r2_str = "R2", "LOG Original data R2"
        rmse_str, inv_rmse_str = "RMSE", "LOG Original data RMSE"
        nrmse_str, inv_nrmse_str = "NRMSE", "LOG Original data NRMSE"
        for i in range(len(mse)):
            inv_mse_str += " {}: {}".format(titles[i], log_inv_mse[i])
            inv_r2_str += " {}: {}".format(titles[i], log_inv_r2[i])
            inv_rmse_str += " {}: {}".format(titles[i], log_inv_rmse[i])
            inv_nrmse_str += " {}: {}".format(titles[i], log_inv_nrmse[i])

        print(inv_mse_str)
        print(inv_r2_str)
        print(inv_rmse_str)
        print(inv_nrmse_str)

        get_mse_nrmse_r2_eigh_3D(targets_list, predictions_list)
        print("ORIGINAL DATA")
        get_mse_nrmse_r2_eigh_3D(inv_targets_arr, inv_predictions_arr)

        print("log_inv_r2 ", log_inv_r2)
        print("log_inv_nrmse ", log_inv_nrmse)

        print("mean log_inv_r2 ", np.mean(log_inv_r2))
        print("mean log_inv_nrmse ", np.mean(log_inv_nrmse))


        plot_target_prediction(np.array(targets_list), np.array(predictions_list), "preprocessed_", x_labels=x_labels, titles=titles)
        plot_target_prediction(inv_targets_arr, inv_predictions_arr, x_labels=x_labels, titles=titles)

        #np.save("inv_tragets_arr_fr_div_10", inv_targets_arr)
        #np.save("inv_predictions_arr_fr_div_10", inv_predictions_arr)

        # inv_targets_arr[:, 0] = np.log10(inv_targets_arr[:, 0])
        # inv_targets_arr[:, 2] = np.log10(inv_targets_arr[:, 2])
        #
        # inv_predictions_arr[:, 0] = np.log10(inv_predictions_arr[:, 0])
        # inv_predictions_arr[:, 2] = np.log10(inv_predictions_arr[:, 2])

        # wrong_targets = np.array(wrong_targets)
        # wrong_predictions = np.array(wrong_predictions)
        # wrong_targets[:, 0] = np.log10(wrong_targets[:, 0])
        # wrong_targets[:, 2] = np.log10(wrong_targets[:, 2])
        #
        # wrong_predictions[:, 0] = np.log10(wrong_predictions[:, 0])
        # wrong_predictions[:, 2] = np.log10(wrong_predictions[:, 2])





        plot_target_prediction(log_inv_targets_arr, log_inv_predictions_arr, title_prefix="log_orig_", r2=log_inv_r2, nrmse=log_inv_nrmse,
                               x_labels=x_labels, titles=titles)

        # plot_target_prediction(wrong_targets, wrong_predictions, title_prefix="wrong_log_orig_", r2=log_inv_r2,
        #                        nrmse=log_inv_nrmse,
        #                        x_labels=[r'$log(k_{xx})$', r'$k_{xy}$', r'$log(k_{yy})$'])

        ######
        ## main peak fr div 0.1
        ######
        #print("inv_targets_arr[inv_targets_arr[:, 0] > -3.8]", inv_targets_arr[inv_targets_arr[:, 0] > -3.8])
        #print(" inv_predictions_arr[inv_targets_arr[:, 0] > -3.8]",  inv_predictions_arr[inv_targets_arr[:, 0] > -3.8])


        # log_inv_mse, log_inv_rmse, log_inv_nrmse, log_inv_r2 = get_mse_nrmse_r2(inv_targets_arr[inv_targets_arr[:, 0] > -3.8],
        #                                                                         inv_predictions_arr[inv_targets_arr[:, 0] > -3.8])

        log_inv_mse, log_inv_rmse, log_inv_nrmse, log_inv_r2 = get_mse_nrmse_r2(log_inv_targets_arr, log_inv_predictions_arr)

        print("log_inv_r2 main peak", log_inv_r2)
        print("log_inv_nrmse main peak", log_inv_nrmse)


def _remove_outliers_iqr(targets, predictions, iqr_multiplier=1.5):
    print("targets.shape ", targets.shape)
    all_indices = []
    for i_ch in range(targets.shape[1]):
        t_data = targets[:, i_ch]
        #p_data = predictions[:, i_ch]
        q1 = np.percentile(t_data, 5)
        q3 = np.percentile(t_data, 95)
        #q1 = np.percentile(t_data, 25)
        #q3 = np.percentile(t_data, 75)
        iqr = q3 - q1
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr
        indices = (t_data >= lower_bound) & (t_data <= upper_bound)
        #filtered_t_data = t_data[indices]
        #filtered_p_data = p_data[indices]
        all_indices.append(indices)

    indices = [index for index, (elem1, elem2, elem3) in enumerate(zip(*all_indices)) if elem1 == elem2 == elem3]

    all_targets = []
    all_predictions = []
    for i_ch in range(targets.shape[1]):
        t_data = targets[:, i_ch]
        p_data = predictions[:, i_ch]

        targets_s = t_data[indices]
        predictions_s = p_data[indices]

        all_targets.append(targets_s)
        all_predictions.append(predictions_s)

    all_targets = np.array(all_targets).T
    all_predictions = np.array(all_predictions).T

    print("all targets ", all_targets.shape)
    print("all predictions ", all_predictions.shape)

    return all_targets, all_predictions


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

    print("torch.cuda.is_available() ", torch.cuda.is_available())
    #print(torch.zeros(1).cuda())


    study = load_study(args.results_dir)

    #@TODO: RM ASAP
    print("study attrs ", study.user_attrs)
    #study.user_attrs["output_log"] = True
    #study.set_user_attr("output_log", True)
    print("study attrs ", study.user_attrs)
    #compare_trials(study)

    load_models(args, study)


