import os
import joblib
import copy
import torch
import torch.nn as nn
import yaml
import numpy as np
import torchvision.transforms as transforms
from metamodel.cnn3D.datasets.dfm3d_dataset import DFM3DDataset
from metamodel.cnn.models.auxiliary_functions import get_mean_std, log_data, exp_data,\
    quantile_transform_fit, QuantileTRF, NormalizeData, log_all_data, init_norm, log10_data, log10_all_data


def load_trials_config(path_to_config):
    with open(path_to_config, "r") as f:
        trials_config = yaml.load(f, Loader=yaml.FullLoader)
    return trials_config


def train_one_epoch(model, optimizer, train_loader, config, loss_fn=nn.MSELoss(), use_cuda=True):
    """
    Train NN
    :param model: neural network model to train
    :param optimizer: optimizer for updating model parameters
    :param train_loader: DataLoader providing training data batches
    :param config: configuration object/dict (not used in current code)
    :param loss_fn: loss function to compute error (default: MSELoss)
    :param use_cuda: whether to use GPU if available
    :return: average training loss for the epoch
    """
    running_loss = 0.  # accumulator for total loss over batches

    for i, data in enumerate(train_loader):
        inputs, targets = data  # unpack batch inputs and targets

        # Move data to GPU if available and requested
        if torch.cuda.is_available() and use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # Ensure inputs and targets are floats
        inputs = inputs.float()
        targets = targets.float()

        optimizer.zero_grad()  # reset gradients before backpropagation

        outputs = torch.squeeze(model(inputs))  # forward pass and squeeze output
        loss = loss_fn(outputs, targets)  # compute loss between prediction and target
        loss.backward()  # backpropagate gradients

        optimizer.step()  # update model parameters

        running_loss += loss.item()  # accumulate scalar loss value

    # Compute average loss over all batches
    train_loss = running_loss / (i + 1)
    return train_loss


def validate(model, validation_loader, loss_fn=nn.MSELoss(), acc_fn=nn.MSELoss(), use_cuda=False):
    """
    Validate model
    :param model: neural network model to evaluate
    :param validation_loader: DataLoader providing validation data batches
    :param loss_fn: loss function to compute error (default: MSELoss)
    :param acc_fn: accuracy function or metric (default: MSELoss used as placeholder)
    :param use_cuda: whether to use GPU if available
    :return: average validation loss and accuracy over the validation dataset
    """
    running_vloss = 0.0  # accumulator for total validation loss
    running_vacc = 0  # accumulator for total accuracy metric

    with torch.no_grad():  # disable gradient computation during validation
        for i, vdata in enumerate(validation_loader):
            vinputs, vtargets = vdata  # unpack batch inputs and targets

            # Move data to GPU if available and requested
            if torch.cuda.is_available() and use_cuda:
                vinputs = vinputs.cuda()
                vtargets = vtargets.cuda()

            # Ensure inputs and targets are floats
            vinputs = vinputs.float()
            vtargets = vtargets.float()

            voutputs = torch.squeeze(model(vinputs))  # forward pass and squeeze output
            vloss = loss_fn(voutputs, vtargets)  # compute validation loss
            running_vloss += vloss.item()  # accumulate validation loss

            vacc = acc_fn(voutputs, vtargets)  # compute accuracy metric
            running_vacc += vacc.item()  # accumulate accuracy

        # Compute average loss and accuracy over all validation batches
        avg_vloss = running_vloss / (i + 1)
        avg_vacc = running_vacc / (i + 1)

    return avg_vloss, avg_vacc


def features_transform(config, data_dir, input_transform_list, output_transform_list, dataset_for_transform=None):
    #################################
    ## Data for Quantile Transform ##
    #################################
    # Not applied in 3D case so far

    quantile_trf_obj = QuantileTRF()  # Initialize quantile transformer object

    # If no dataset provided, create one using DFM3DDataset with config options
    if dataset_for_transform is None:
        dataset_for_transform = DFM3DDataset(
            zarr_path=data_dir,
            fractures_sep=config["fractures_sep"] if "fractures_sep" in config else False,
            cross_section=config["cross_section"] if "cross_section" in config else False
        )

    input_data = np.array([])  # Empty array to accumulate input data for fitting quantile transform
    output_data = np.array([])  # Empty array to accumulate output data for fitting quantile transform

    n_data_input = 1000000  # Maximum number of input data points to use for fitting
    n_data_output = 300000  # Maximum number of output data points to use for fitting

    # Check if input or output transform is specified in config
    if "input_transform" in config or "output_transform" in config:
        # Loop over the dataset to gather data for quantile transform fitting
        for index, data in enumerate(dataset_for_transform):
            input, output = data

            # Reshape input to 2D array: (features, flattened spatial dimensions)
            input = np.reshape(input, (input.shape[0], input.shape[-1] * input.shape[-2]))

            # Accumulate input data until reaching max data points
            if input.shape[-1] * input.shape[-2] * index < n_data_input:
                if len(input_data) == 0:
                    input_data = input
                else:
                    input_data = np.concatenate([input_data, input], axis=1)

            # Accumulate output data until reaching max data points
            if output_data.shape[-1] < n_data_output:
                output = np.reshape(output, (output.shape[0], 1))  # reshape output to 2D
                if len(output_data) == 0:
                    output_data = output
                else:
                    output_data = np.concatenate([output_data, output], axis=1)

    # If input transform specified, fit quantile transformer and append to input_transform_list
    if "input_transform" in config and len(config["input_transform"]) > 0:
        quantile_trfs = quantile_transform_fit(
            input_data,
            indices=config["input_transform"]["indices"],
            transform_type=config["input_transform"]["type"]
        )
        # Save the fitted transformer to disk
        joblib.dump(quantile_trfs, os.path.join(config["output_dir"], "input_transform.pkl"))

        # Store the transformer object for later use
        quantile_trf_obj.quantile_trfs_in = quantile_trfs

        # Append the quantile transform function to the input_transform_list
        input_transform_list.append(quantile_trf_obj.quantile_transform_in)

    # If output transform specified, fit quantile transformer and append to output_transform_list
    if "output_transform" in config and len(config["output_transform"]) > 0:
        quantile_trfs_out = quantile_transform_fit(
            output_data,
            indices=config["output_transform"]["indices"],
            transform_type=config["output_transform"]["type"]
        )
        # Save the fitted transformer to disk
        joblib.dump(quantile_trfs_out, os.path.join(config["output_dir"], "output_transform.pkl"))

        # Store the transformer object for later use
        quantile_trf_obj.quantile_trfs_out = quantile_trfs_out

        # Append the quantile transform function to the output_transform_list
        output_transform_list.append(quantile_trf_obj.quantile_transform_out)

    # Return the updated lists of transforms
    return input_transform_list, output_transform_list


def _append_dataset(dataset_1, dataset_2):
    # Append file paths from dataset_2 to dataset_1 for all file categories
    dataset_1._bulk_file_paths.extend(dataset_2._bulk_file_paths)
    dataset_1._fracture_file_paths.extend(dataset_2._fracture_file_paths)
    dataset_1._cross_section_file_paths.extend(dataset_2._cross_section_file_paths)
    dataset_1._output_file_paths.extend(dataset_2._output_file_paths)


def prepare_sub_datasets(study, config, data_dir, serialize_path=None):
    # Initialize variables to hold combined train, validation, and test datasets
    complete_train_set, complete_val_set, complete_test_set = None, None, None

    # Loop through each sub-dataset configuration defined in config
    for key, dset_config in config["sub_datasets"].items():
        # Create a deep copy of the main config to customize for this sub-dataset
        prepare_dset_config = copy.deepcopy(config)

        # Override config parameters with sub-dataset specific settings
        prepare_dset_config["log_input"] = dset_config["log_input"]
        if "init_norm" in dset_config:
            prepare_dset_config["init_norm"] = dset_config["init_norm"]
        if "input_transform" in dset_config:
            prepare_dset_config["input_transform"] = dset_config["input_transform"]
        if "output_transform" in dset_config:
            prepare_dset_config["output_transform"] = dset_config["output_transform"]

        prepare_dset_config["normalize_input"] = dset_config["normalize_input"]
        prepare_dset_config["log_output"] = dset_config["log_output"]
        if "log10_output" in dset_config:
            prepare_dset_config["log10_output"] = dset_config["log10_output"]
        if "log_all_output" in dset_config:
            prepare_dset_config["log_all_output"] = dset_config["log_all_output"]
        if "log10_all_output" in dset_config:
            prepare_dset_config["log10_all_output"] = dset_config["log10_all_output"]
        prepare_dset_config["normalize_output"] = dset_config["normalize_output"]

        # Number of samples and ratios for training, validation, and testing
        prepare_dset_config["n_train_samples"] = dset_config["n_train_samples"]
        prepare_dset_config["n_val_samples"] = dset_config["n_val_samples"]
        prepare_dset_config["n_test_samples"] = dset_config["n_test_samples"]
        prepare_dset_config["val_samples_ratio"] = dset_config["val_samples_ratio"]

        print("prepare_dset_config ", prepare_dset_config)

        # Prepare datasets for this sub-dataset configuration
        sub_train_set, sub_val_set, sub_test_set = prepare_dataset(study, prepare_dset_config,
                                                                   dset_config['dataset_path'])

        # If this is the first sub-dataset, assign directly
        if complete_train_set is None:
            complete_train_set = sub_train_set
            complete_val_set = sub_val_set
            complete_test_set = sub_test_set
        else:
            # Otherwise, append current sub-dataset to the complete datasets
            _append_dataset(complete_train_set, sub_train_set)
            if len(sub_val_set) > 0:
                _append_dataset(complete_val_set, sub_val_set)
            _append_dataset(complete_test_set, sub_test_set)

    # After processing all sub-datasets, prepare transforms using the combined training set
    data_init_transform, data_input_transform, data_output_transform = prepare_dataset(
        study, config, data_dir, train_dataset=complete_train_set)

    # Assign transforms to the complete datasets
    complete_train_set.init_transform = data_init_transform
    complete_train_set.input_transform = data_input_transform
    complete_train_set.output_transform = data_output_transform
    if len(complete_val_set) > 0:
        complete_val_set.init_transform = data_init_transform
        complete_val_set.input_transform = data_input_transform
        complete_val_set.output_transform = data_output_transform
    complete_test_set.init_transform = data_init_transform
    complete_test_set.input_transform = data_input_transform
    complete_test_set.output_transform = data_output_transform

    # Create a combined dataset containing train, val, and test data
    dataset = copy.deepcopy(complete_train_set)
    if len(complete_val_set) > 0:
        _append_dataset(dataset, complete_val_set)
    _append_dataset(dataset, complete_test_set)

    # Optionally serialize the combined dataset to disk
    if serialize_path is not None:
        joblib.dump(dataset, os.path.join(serialize_path, "dataset.pkl"))

    # If study object provided, record dataset sizes as user attributes
    if study is not None:
        study.set_user_attr("n_train_samples", len(complete_train_set))
        study.set_user_attr("n_val_samples", len(complete_val_set))
        study.set_user_attr("n_test_samples", len(complete_test_set))

    # Return the separate complete train, val, and test datasets
    return complete_train_set, complete_val_set, complete_test_set


def _split_dataset(dataset_config, config, n_train_samples, n_val_samples, n_test_samples):
    # Instantiate the full dataset
    dataset = DFM3DDataset(**dataset_config)

    # If no explicit train sample count, compute based on ratio in config
    if n_train_samples is None:
        n_train_samples = int(len(dataset) * config["train_samples_ratio"])
        n_train_val_samples = np.min([n_train_samples, int(len(dataset) * config["train_samples_ratio"])])

    # If no explicit validation sample count, compute based on ratio or set zero if ratio is 0
    if n_val_samples is None:
        if config["val_samples_ratio"] == 0.0:
            n_train_samples = n_train_val_samples
            n_val_samples = 0
        else:
            n_val_samples = int(n_train_val_samples * config["val_samples_ratio"])
            n_train_samples = n_train_val_samples - n_val_samples

    # If no explicit test sample count, compute as remainder of dataset
    if n_test_samples is None:
        n_test_samples = len(dataset) - n_train_val_samples

    # Create train, validation, and test dataset splits with specified sizes and modes
    train_set = DFM3DDataset(**dataset_config, mode="train", train_size=n_train_samples, val_size=n_val_samples,
                             test_size=n_test_samples)
    validation_set = DFM3DDataset(**dataset_config, mode="val", train_size=n_train_samples, val_size=n_val_samples,
                                  test_size=n_test_samples)
    test_set = DFM3DDataset(**dataset_config, mode="test", train_size=n_train_samples, val_size=n_val_samples,
                            test_size=n_test_samples)

    return train_set, validation_set, test_set


def prepare_dataset(study, config, data_dir, serialize_path=None, train_dataset=None):
    # ===================================
    # Get mean and std for each channel
    # ===================================
    input_mean, output_mean, input_std, output_std = 0, 0, 1, 1
    output_quantiles = []
    data_normalizer = NormalizeData()

    # ==============================
    # Determine dataset sample sizes
    # ==============================
    n_train_samples = None
    if "n_train_samples" in config and config["n_train_samples"] is not None:
        n_train_samples = config["n_train_samples"]
    n_val_samples = None
    if "n_val_samples" in config and config["n_val_samples"] is not None:
        n_val_samples = config["n_val_samples"]
    n_test_samples = None
    if "n_test_samples" in config and config["n_test_samples"] is not None:
        n_test_samples = config["n_test_samples"]

    print("n train samples: {}, n val samples: {}, n test samples: {}".format(n_train_samples, n_val_samples, n_test_samples))

    # ===========================
    # Initialize transformation lists
    # ===========================
    init_transform = []
    input_transform_list = []
    output_transform_list = []

    ###########################
    ## Initial normalization ##
    ###########################
    if config["init_norm"]:
        # Apply custom initial normalization transform
        init_transform.append(transforms.Lambda(init_norm))

    ####################
    ## Log transforms ##
    ####################
    if config["log_input"]:
        # Log-transform all input channels or only specific ones
        if "log_all_input_channels" in config and config["log_all_input_channels"]:
            input_transform_list.append(transforms.Lambda(log_all_data))
        else:
            input_transform_list.append(transforms.Lambda(log_data))

    # Apply logarithmic transform to output depending on config flags
    if config["log_output"]:
        output_transform_list.append(transforms.Lambda(log_data))
    elif "log10_output" in config and  config["log10_output"]:
        output_transform_list.append(transforms.Lambda(log10_data))
    elif "log_all_output" in config and config["log_all_output"]:
        output_transform_list.append(transforms.Lambda(log_all_data))
    elif "log10_all_output" in config and config["log10_all_output"]:
        output_transform_list.append(transforms.Lambda(log10_all_data))

    ########################
    ## Quantile Transform ##
    ########################
    # Apply feature-based transforms (e.g., quantile scaling)
    input_transform_list, output_transform_list = features_transform(config,
                                                                     data_dir,
                                                                     input_transform_list,
                                                                     output_transform_list)

    # Compose transforms for input and output
    input_transform = transforms.Compose(input_transform_list)
    output_transform = transforms.Compose(output_transform_list)

    # Compose init_transform if any were specified
    if len(init_transform) > 0:
        init_transform = transforms.Compose(init_transform)
    else:
        init_transform = None

    # ====================================
    # Compute and apply normalization stats
    # ====================================
    if config["normalize_input"] or config["normalize_output"]:
        if train_dataset is not None:
            # Use provided training dataset
            train_set = train_dataset
            train_set.init_transform = init_transform
            train_set.input_transform = input_transform
            train_set.output_transform = output_transform
        else:
            # Create training dataset from scratch with provided transforms
            train_set = DFM3DDataset(
                zarr_path=data_dir,
                input_transform=input_transform,
                output_transform=output_transform,
                init_transform=init_transform,
                init_norm_use_all_features=config.get("init_norm_use_all_features", False),
                input_channels=config.get("input_channels", None),
                output_channels=config.get("output_channels", None),
                fractures_sep=config.get("fractures_sep", False),
                cross_section=config.get("cross_section", False),
                mode="train",
                train_size=n_train_samples,
                val_size=n_val_samples,
                test_size=n_test_samples
            )

        print("len(train_set) for get mean std", len(train_set))

        # Use DataLoader to iterate over train set and compute stats
        train_loader_mean_std = torch.utils.data.DataLoader(
            train_set,
            batch_size=config["batch_size_train"],
            shuffle=False
        )

        # Optional: interquartile range scaling
        iqr = config.get("output_iqr_scale", [])

        # Compute input/output means and stds over training data
        input_mean, input_std, output_mean, output_std, output_quantiles = get_mean_std(
            train_loader_mean_std,
            output_iqr=iqr,
            mean_dims=[0, 2, 3, 4]  # Apply mean/std over spatial dims + batch
        )

        # Reshape input mean and std to match input tensor dimensions
        input_mean = input_mean.reshape(input_mean.shape[0], 1, 1, 1)
        input_std = input_std.reshape(input_std.shape[0], 1, 1, 1)

        print("input mean: {}, std:{}, output mean: {}, std: {}".format(input_mean, input_std, output_mean, output_std))
        print("output quantiles: {}".format(output_quantiles))

    # =======================
    # data transforms
    # =======================
    input_transformations = []  # List to store input transformations
    output_transformations = []  # List to store output transformations
    init_transform = []  # List to store initial transformations (e.g., initial normalization)

    ###########################
    ## Initial normalization ##
    ###########################
    data_init_transform = None
    if config["init_norm"]:
        # Add user-defined initial normalization if specified in config
        init_transform.append(transforms.Lambda(init_norm))

    # Compose initial transformations if any are specified
    if len(init_transform) > 0:
        data_init_transform = transforms.Compose(init_transform)

    data_input_transform, data_output_transform = None, None

    # ===============================
    # Input transformations section
    # ===============================
    # Apply log transformation to input if enabled
    if config["log_input"]:
        if "log_all_input_channels" in config and config["log_all_input_channels"]:
            input_transformations.append(transforms.Lambda(log_all_data))
        else:
            input_transformations.append(transforms.Lambda(log_data))

    # Apply normalization to input if enabled
    if config["normalize_input"]:
        if "normalize_input_indices" in config:
            data_normalizer.input_indices = config["normalize_input_indices"]
        data_normalizer.input_mean = input_mean
        data_normalizer.input_std = input_std
        input_transformations.append(data_normalizer.normalize_input)

    # Compose all input transformations
    if len(input_transformations) > 0:
        data_input_transform = transforms.Compose(input_transformations)

    # ================================
    # Output transformations section
    # ================================
    # Apply log or log10 transformation to output depending on config
    if config["log_output"]:
        output_transformations.append(transforms.Lambda(log_data))
    elif "log10_output" in config and config["log10_output"]:
        output_transformations.append(transforms.Lambda(log10_data))
    elif "log_all_output" in config and config["log_all_output"]:
        output_transformations.append(transforms.Lambda(log_all_data))
    elif "log10_all_output" in config and config["log10_all_output"]:
        output_transformations.append(transforms.Lambda(log10_all_data))

    # Apply normalization to output if enabled
    if config["normalize_output"]:
        if "normalize_output_indices" in config:
            data_normalizer.input_indices = config["normalize_output_indices"]
        data_normalizer.output_mean = output_mean
        data_normalizer.output_std = output_std
        data_normalizer.output_quantiles = output_quantiles
        output_transformations.append(data_normalizer.normalize_output)

    # Compose all output transformations
    if len(output_transformations) > 0:
        data_output_transform = transforms.Compose(output_transformations)

    # =============================================
    # If no existing train_dataset, build datasets
    # =============================================
    if train_dataset is None:
        # Build dataset configuration dictionary
        dataset_config = {"zarr_path": data_dir,
                          "input_transform": data_input_transform,
                          "output_transform": data_output_transform,
                          "init_transform": data_init_transform,
                          "init_norm_use_all_features": config[
                              "init_norm_use_all_features"] if "init_norm_use_all_features" in config else False,
                          "input_channels": config["input_channels"] if "input_channels" in config else None,
                          "output_channels": config["output_channels"] if "output_channels" in config else None,
                          "fractures_sep": config["fractures_sep"] if "fractures_sep" in config else False,
                          "cross_section": config["cross_section"] if "cross_section" in config else False}

        # Split dataset into train, val, test
        train_set, validation_set, test_set = _split_dataset(dataset_config, config, n_train_samples=n_train_samples,
                                                             n_val_samples=n_val_samples, n_test_samples=n_test_samples)

    else:
        # Set transform attributes on existing train_dataset
        train_dataset.init_transform = data_init_transform
        train_dataset.input_transform = data_input_transform
        train_dataset.output_transform = data_output_transform

    # =================================================
    # If transform configs are provided, refine them
    # =================================================
    if "input_transform" in config or "output_transform" in config:
        if train_dataset is not None:
            train_set = train_dataset

        # Update transformations with additional config-based feature transforms
        input_transformations, output_transformations = features_transform(config, data_dir,
                                                                           input_transformations,
                                                                           output_transformations, train_set)

        # Re-compose transformations after possible updates
        if len(output_transformations) > 0:
            data_output_transform = transforms.Compose(output_transformations)
        if len(input_transformations) > 0:
            data_input_transform = transforms.Compose(input_transformations)

        # Rebuild dataset configuration and perform split again
        dataset_config = {"zarr_path": data_dir,
                          "input_transform": data_input_transform,
                          "output_transform": data_output_transform,
                          "init_transform": data_init_transform,
                          "init_norm_use_all_features": config[
                              "init_norm_use_all_features"] if "init_norm_use_all_features" in config else False,
                          "input_channels": config["input_channels"] if "input_channels" in config else None,
                          "output_channels": config["output_channels"] if "output_channels" in config else None,
                          "fractures_sep": config["fractures_sep"] if "fractures_sep" in config else False,
                          "cross_section": config["cross_section"] if "cross_section" in config else False
                          }

        train_set, validation_set, test_set = _split_dataset(dataset_config, config, n_train_samples)

    # ========================================================
    # Save configuration to the study for reproducibility
    # ========================================================
    if study is not None:
        if train_dataset is None:
            study.set_user_attr("n_train_samples", len(train_set))
            study.set_user_attr("n_val_samples", len(validation_set))
            study.set_user_attr("n_test_samples", len(test_set))

        if "normalize_input_indices" in config:
            study.set_user_attr("normalize_input_indices", config["normalize_input_indices"])

        if "normalize_output_indices" in config:
            study.set_user_attr("normalize_output_indices", config["normalize_output_indices"])

        if "log_all_input_channels" in config:
            study.set_user_attr("log_all_input_channels", config["log_all_input_channels"])

        if "output_transform" in config:
            study.set_user_attr("output_transform", config["output_transform"])

        if "input_transform" in config:
            study.set_user_attr("input_transform", config["input_transform"])

        if "init_norm_use_all_features" in config:
            study.set_user_attr("init_norm_use_all_features", config["init_norm_use_all_features"])

        study.set_user_attr("init_norm", config["init_norm"])
        study.set_user_attr("normalize_input", config["normalize_input"])
        study.set_user_attr("normalize_output", config["normalize_output"])

        study.set_user_attr("input_log", config["log_input"])
        study.set_user_attr("input_mean", input_mean)
        study.set_user_attr("input_std", input_std)

        study.set_user_attr("output_log", config["log_output"])
        if "log_all_output" in config:
            study.set_user_attr("log_all_output", config["log_all_output"])
        if "log10_all_output" in config:
            study.set_user_attr("log10_all_output", config["log10_all_output"])
        if "log10_output" in config:
            study.set_user_attr("log10_output", config["log10_output"])

        study.set_user_attr("output_mean", output_mean)
        study.set_user_attr("output_std", output_std)
        study.set_user_attr("output_quantiles", output_quantiles)

        if "input_channels" in config:
            study.set_user_attr("input_channels", config["input_channels"])

        if "output_channels" in config:
            study.set_user_attr("output_channels", config["output_channels"])

    # =========================
    # Return or serialize data
    # =========================
    if train_dataset is not None:
        # If working with existing dataset, return transforms
        return data_init_transform, data_input_transform, data_output_transform

    # Serialize datasets if path is provided
    if serialize_path is not None:
        joblib.dump(train_set, os.path.join(serialize_path, "train_dataset.pkl"))
        joblib.dump(validation_set, os.path.join(serialize_path, "val_dataset.pkl"))
        joblib.dump(test_set, os.path.join(serialize_path, "test_dataset.pkl"))

    # Return train/val/test splits
    return train_set, validation_set, test_set