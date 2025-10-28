import os
import sys
import argparse
import joblib
import torch
import numpy as np
import torchvision.transforms as transforms
from metamodel.cnn3D.datasets.dfm3d_dataset import DFM3DDataset
from metamodel.cnn.visualization.visualize_data import plot_target_prediction, plot_train_valid_loss
from metamodel.cnn.models.auxiliary_functions import get_mse_nrmse_r2, get_mean_std, log_data, exp_data, QuantileTRF, \
    NormalizeData, log_all_data, init_norm, get_mse_nrmse_r2_eigh_3D, log10_data, log10_all_data, power_10_all_data, power_10_data

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


    print("LEN train set: {}, val set: {}, test set: {}".format(len(train_set), len(validation_set), len(test_set)))

    return train_set, validation_set, test_set

def get_saved_model_path(results_dir, best_trial):
    model_path = 'trial_{}_losses_model_{}'.format(best_trial.number, best_trial.user_attrs["model_name"])
    return os.path.join(results_dir, model_path)


def load_dataset(results_dir, data_zarr, study):
    """
    Load and preprocess dataset using configuration derived from an Optuna study.

    :param results_dir: Path to the directory where results (e.g., normalization stats) will be saved.
    :param data_zarr: Path to the input data in Zarr format.
    :param study: Optuna study object containing user-defined preprocessing and training parameters.
    :return: Preprocessed training inputs, training targets, validation inputs, and validation targets.
    """

    # Configuration dictionary for dataset preprocessing and training
    config = {
        "batch_size_train": 32,                     # Batch size for training
        "n_train_samples": 5,                       # Number of training samples (absolute)
        "n_test_samples": 3500,                     # Number of test samples (absolute)
        "train_samples_ratio": 0.1,                 # Ratio of full dataset used for training
        "val_samples_ratio": 0.2,                   # Ratio of training data used for validation
        "print_batches": 10,                        # Frequency of batch-level logging

        # Input normalization options
        "init_norm": study.user_attrs["init_norm"],                     # Method for computing initial normalization (e.g., mean/std)
        "log_input": study.user_attrs["input_log"],                     # Apply log transform to inputs
        "normalize_input": study.user_attrs["normalize_input"],         # Normalize input data
        "init_norm_use_all_features": study.user_attrs.get(
            "init_norm_use_all_features", False),                       # Whether to normalize across all input features jointly

        # Output normalization options
        "log_output": study.user_attrs["output_log"],                   # Apply log transform to outputs
        "normalize_output": study.user_attrs["normalize_output"],       # Normalize output data
        "seed": 12345                                                   # Seed for reproducibility
    }

    # Optional configurations set if present in the study's user attributes
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

    # Call the preprocessing pipeline with the assembled configuration
    return preprocess_dataset(
        config=config,
        user_attrs=study.user_attrs,
        data_dir=data_zarr,
        results_dir=results_dir
    )


def get_inverse_transform_input(study):
    """
    Constructs the inverse transformation pipeline for input data based on the normalization
    and transformation settings stored in the Optuna study's user attributes.

    :param study: Optuna study object that contains user-defined preprocessing parameters
                  such as input normalization and log transformation settings.
    :return: A composed torchvision transform that approximately inverts input normalization/logging.
    """
    inverse_transform = None

    print("study.user_attrs", study.user_attrs)

    # Check if normalization was applied to input data
    if "normalize_input" in study.user_attrs and study.user_attrs["normalize_input"]:
        # Compute inverse of standard deviation
        std = 1 / study.user_attrs["input_std"]
        zeros_mean = np.zeros(len(study.user_attrs["input_mean"]))

        print("input_mean ", study.user_attrs["input_mean"])
        print("input_std ", study.user_attrs["input_std"])

        # Create placeholders for identity normalization
        ones_std = np.ones(len(zeros_mean))
        mean = -study.user_attrs["input_mean"]

        # Reshape to match expected input shape for normalization
        zeros_mean = zeros_mean.reshape(mean.shape)
        ones_std = ones_std.reshape(std.shape)

        # First normalization reverses standard scaling, second adds back the mean
        transforms_list = [
            transforms.Normalize(mean=zeros_mean, std=std),     # Undo division by std
            transforms.Normalize(mean=mean, std=ones_std)       # Undo subtraction of mean
        ]

        # If log transformation was applied to input, add exponential to reverse it
        if "input_log" in study.user_attrs and study.user_attrs["input_log"]:
            print("input log to transform list")
            transforms_list.append(transforms.Lambda(exp_data))  # Apply exp to undo log

        # Compose all inverse transformations
        inverse_transform = transforms.Compose(transforms_list)

    # Placeholder print if initial normalization was used
    if "init_norm" in study.user_attrs and study.user_attrs["init_norm"]:
        print("init norm ")

    return inverse_transform


def get_transform(study, results_dir=None):
    # Lists to store composed transformations for input, output, and initial normalization
    input_transformations = []
    output_transformations = []
    init_transform = []

    # Normalizer object for applying custom normalization
    data_normalizer = NormalizeData()

    ###########################
    ## Initial normalization ##
    ###########################

    data_init_transform = None
    # Apply initial normalization if enabled in the study configuration
    if "init_norm" in study.user_attrs and study.user_attrs["init_norm"]:
        init_transform.append(transforms.Lambda(init_norm))

    # Compose initial normalization transform if any were added
    if len(init_transform) > 0:
        data_init_transform = transforms.Compose(init_transform)

    data_input_transform, data_output_transform = None, None

    # -------------------------
    # Standardize input
    # -------------------------

    # Apply log transform to input if specified
    if "input_log" in study.user_attrs and study.user_attrs["input_log"]:
        if "log_all_input_channels" in study.user_attrs and study.user_attrs["log_all_input_channels"]:
            input_transformations.append(transforms.Lambda(log_all_data))  # Log transform all channels
        else:
            input_transformations.append(transforms.Lambda(log_data))  # Log transform selective channels

    # Apply normalization to input if specified
    if "normalize_input" in study.user_attrs and study.user_attrs["normalize_input"]:
        if "normalize_input_indices" in study.user_attrs:
            data_normalizer.input_indices = study.user_attrs["normalize_input_indices"]
        data_normalizer.input_mean = study.user_attrs["input_mean"]
        data_normalizer.input_std = study.user_attrs["input_std"]
        input_transformations.append(data_normalizer.normalize_input)

    # Compose input transformations if any were added
    if len(input_transformations) > 0:
        data_input_transform = transforms.Compose(input_transformations)

    # -------------------------
    # Standardize output
    # -------------------------

    # Apply log transform to output if specified
    if "output_log" in study.user_attrs and study.user_attrs["output_log"]:
        output_transformations.append(transforms.Lambda(log_data))

    # Apply normalization to output if specified
    if "normalize_output" in study.user_attrs and study.user_attrs["normalize_output"]:
        if "normalize_output_indices" in study.user_attrs:
            data_normalizer.output_indices = study.user_attrs["normalize_output_indices"]
        data_normalizer.output_mean = study.user_attrs["output_mean"]
        data_normalizer.output_std = study.user_attrs["output_std"]
        if "output_quantiles" in study.user_attrs:
            data_normalizer.output_quantiles = study.user_attrs["output_quantiles"]
        output_transformations.append(data_normalizer.normalize_output)

    # -------------------------
    # Optional quantile transformation for inverse mapping
    # -------------------------

    transforms_list = []
    if ("output_transform" in study.user_attrs and len(study.user_attrs["output_transform"]) > 0) \
            or os.path.exists(os.path.join(results_dir, "output_transform.pkl")):
        output_transform = joblib.load(os.path.join(results_dir, "output_transform.pkl"))
        quantile_trf_obj = QuantileTRF()
        quantile_trf_obj.quantile_trfs_out = output_transform
        transforms_list.append(quantile_trf_obj.quantile_inv_transform_out)

    # Compose output transformations if any were added
    if len(output_transformations) > 0:
        data_output_transform = transforms.Compose(output_transformations)

    # Return all transformation pipelines
    return data_init_transform, data_input_transform, data_output_transform


def get_inverse_transform(study, results_dir=None):
    """
    Constructs and returns an inverse transformation pipeline for model outputs.
    This is used to reverse the normalization, log-transform, or quantile transformation
    applied during training or preprocessing.

    Parameters:
    -----------
    study : optuna.Study or similar object
        Study object containing user-defined attributes (user_attrs) specifying
        which output transformations were applied and need to be reversed.

    results_dir : str or None, optional
        Directory path where transformation objects like quantile transforms
        (e.g., 'output_transform.pkl') are stored.

    Returns:
    --------
    inverse_transform : torchvision.transforms.Compose or None
        Composed transformation object that reverses all applied output transformations.
    """

    inverse_transform = None
    print("study.user_attrs", study.user_attrs)

    transforms_list = []

    #############################
    # Invert quantile transform #
    #############################
    if ("output_transform" in study.user_attrs and len(study.user_attrs["output_transform"]) > 0) \
            or os.path.exists(os.path.join(results_dir, "output_transform.pkl")):
        output_transform = joblib.load(os.path.join(results_dir, "output_transform.pkl"))
        quantile_trf_obj = QuantileTRF()
        quantile_trf_obj.quantile_trfs_out = output_transform
        transforms_list.append(quantile_trf_obj.quantile_inv_transform_out)

    ###############################
    # Invert standard normalization
    ###############################
    if "normalize_output" in study.user_attrs and study.user_attrs["normalize_output"]:
        std = 1 / study.user_attrs["output_std"]
        zeros_mean = np.zeros(len(study.user_attrs["output_mean"]))

        print("output_mean ", study.user_attrs["output_mean"])
        print("output_std ",  study.user_attrs["output_std"])

        ones_std = np.ones(len(zeros_mean))
        mean = -study.user_attrs["output_mean"]

        # First, scale back by std, then shift by mean
        transforms_list.extend([
            transforms.Normalize(mean=zeros_mean, std=std),
            transforms.Normalize(mean=mean, std=ones_std)
        ])

        #############################
        # Invert log transformations
        #############################
        if "output_log" in study.user_attrs and study.user_attrs["output_log"]:
            print("output log to transform list")
            transforms_list.append(transforms.Lambda(exp_data))
        elif "log10_output" in study.user_attrs and study.user_attrs["log10_output"]:
            print("log10_output to transform list")
            transforms_list.append(transforms.Lambda(power_10_data()))
        elif "log10_all_output" in study.user_attrs and study.user_attrs["log10_all_output"]:
            print("log10_all_output to transform list")
            transforms_list.append(transforms.Lambda(power_10_all_data))
        elif "log_all_output" in study.user_attrs and study.user_attrs["log_all_output"]:
            print("log_all_output to transform list")
            transforms_list.append(transforms.Lambda(exp_data))

    inverse_transform = transforms.Compose(transforms_list)
    return inverse_transform


def load_models(args, study):
    # Extract directory paths and model path from arguments and study
    results_dir = args.results_dir
    data_zarr = args.data_zarr
    model_path = get_saved_model_path(results_dir, study.best_trial)

    # Load datasets for training, validation, and testing
    train_set, validation_set, test_set = load_dataset(results_dir, data_zarr, study)

    # Create PyTorch DataLoader objects for the datasets without shuffling
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=False)
    validation_loader = torch.utils.data.DataLoader(validation_set, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False)

    # Print lengths of datasets
    print("len(trainset): {}, len(valset): {}, len(testset): {}".format(len(train_set), len(validation_set),
                                                                        len(test_set)))

    # Get inverse transformations for output and input data
    inverse_transform = get_inverse_transform(study, results_dir)
    input_inverse_transform = get_inverse_transform_input(study)

    # Disable gradient computations for inference/loading
    with torch.no_grad():
        # Print model initialization parameters
        print("model kwargs ", study.best_trial.user_attrs["model_kwargs"])
        model_kwargs = study.best_trial.user_attrs["model_kwargs"]
        # Instantiate model class with saved kwargs
        model = study.best_trial.user_attrs["model_class"](**model_kwargs)

        # Collect parameters that require gradients for optimizer initialization
        non_frozen_parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = None
        # Initialize optimizer only if there are trainable parameters
        if len(non_frozen_parameters) > 0:
            optimizer = study.best_trial.user_attrs["optimizer_class"](non_frozen_parameters,
                                                                   **study.best_trial.user_attrs["optimizer_kwargs"])

        # Load checkpoint to CPU or GPU based on availability
        if not torch.cuda.is_available():
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(model_path)

        # Extract training and validation loss histories from checkpoint
        train_loss = checkpoint['train_loss']
        valid_loss = checkpoint['valid_loss']

        # Print best validation loss and corresponding epoch
        print("best val loss: {}".format(np.min(valid_loss)))
        #print("best epoch: {}".format(np.argmin(valid_loss)))

        # Load the best model state from checkpoint and set model to evaluation mode
        model.load_state_dict(checkpoint['best_model_state_dict'])
        model.eval()

        # Load optimizer state if optimizer was initialized
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['best_optimizer_state_dict'])
        epoch = checkpoint['best_epoch']

        # Print best epoch and losses
        print("Best epoch ", epoch)
        print("train loss ", train_loss)
        print("valid loss ", valid_loss)
        print("model training time ", checkpoint["training_time"])

        # Plot training and validation loss curves
        plot_train_valid_loss(train_loss, valid_loss)

        # Initialize accumulators and lists for metrics and predictions
        running_loss, inv_running_loss = 0, 0
        targets_list, predictions_list = [], []
        inv_targets_list, inv_predictions_list = [], []

        # Iterate over test dataset batches
        for i, test_sample in enumerate(test_loader):
            inputs, targets = test_sample
            inputs = inputs.float()
            # Move inputs to GPU if available and requested
            if args.cuda and torch.cuda.is_available():
                inputs = inputs.cuda()
            # Generate predictions from model
            predictions = model(inputs)

            # Handle inverse transformation for inputs if applicable
            if test_set.init_transform is not None and input_inverse_transform is not None:
                inv_input_avg = test_set._bulk_features_avg

            # Squeeze targets and predictions if they have extra singleton dimensions
            if len(targets.size()) > 1 and np.sum(targets.size())/len(targets.size()) != 1:
                targets = torch.squeeze(targets.float())
            if len(predictions.size()) > 1 and np.sum(predictions.size())/len(predictions.size()) != 1:
                predictions = torch.squeeze(predictions.float())

            # Move predictions to CPU and convert to numpy arrays
            predictions = predictions.cpu()
            predictions_np = predictions.cpu().numpy()

            targets_np = targets.numpy()
            # Append current batch targets and predictions to lists
            targets_list.append(targets_np)
            predictions_list.append(predictions_np)

            inv_targets = targets
            inv_predictions = predictions
            # Apply inverse transform to targets and predictions if defined
            if inverse_transform is not None:

                try:
                    inv_targets = inverse_transform(torch.reshape(targets, (*targets.shape, 1, 1)))
                    inv_predictions = inverse_transform(torch.reshape(predictions, (*predictions.shape, 1, 1)))
                except:
                    # If inverse transform fails, skip this batch
                    print("continue")
                    continue

                # Apply input inverse transform scaling if applicable
                if test_set.init_transform is not None and input_inverse_transform is not None:
                    inv_predictions *= inv_input_avg
                    inv_targets *= inv_input_avg

                # Reshape inverse transformed tensors back to original shape
                inv_targets = np.reshape(inv_targets, targets.shape)
                inv_predictions = np.reshape(inv_predictions, predictions.shape)

            # Append inverse transformed targets and predictions to lists
            inv_targets_list.append(inv_targets.numpy())
            inv_predictions_list.append(inv_predictions.numpy())

        # Convert lists to numpy arrays for further processing
        inv_targets_arr = np.array(inv_targets_list)
        inv_predictions_arr = np.array(inv_predictions_list)
        predictions_list = np.array(predictions_list)

        # Save arrays to disk
        np.savez("inv_targets_arr", data=inv_targets_arr)
        np.savez("inv_predictions_arr", data=inv_predictions_arr)
        np.savez("targets_list", data=targets_list)
        np.savez("predictions_list", data=predictions_list)

        # Calculate MSE, RMSE, NRMSE, R2 metrics for predictions and inverse transformed predictions
        mse, rmse, nrmse, r2 = get_mse_nrmse_r2(targets_list, predictions_list)
        inv_mse, inv_rmse, inv_nrmse, inv_r2 = get_mse_nrmse_r2(inv_targets_arr, inv_predictions_arr)

        # Compute average test loss values
        test_loss = running_loss / (i + 1)
        inv_test_loss = inv_running_loss / (i + 1)

        # Print summary of losses
        print("epochs: {}, train loss: {}, valid loss: {}, test loss: {}, inv test loss: {}".format(epoch,
                                                                                                    train_loss,
                                                                                                    valid_loss,
                                                                                                    test_loss,
                                                                                                inv_test_loss))

        # Define titles and labels for plotting results for each output variable
        titles = ['k_xx', 'k_yy', 'k_zz', 'k_yz', 'k_xz', 'k_xy']
        x_labels = [r'$log(k_{xx})$', r'$log(k_{yy})$', r'$log(k_{zz})$', r'$k_{yz}$', r'$k_{xz}$', r'$k_{xy}$']

        # Prepare strings to print metric values
        mse_str, inv_mse_str = "MSE", "Original data MSE"
        r2_str, inv_r2_str = "R2", "Original data R2"
        rmse_str, inv_rmse_str = "RMSE", "Original data RMSE"
        nrmse_str, inv_nrmse_str = "NRMSE", "Original data NRMSE"
        # Append metric values for each output variable
        for i in range(len(mse)):
            mse_str += " {}: {}".format(titles[i], mse[i])
            r2_str += " {}: {}".format(titles[i], r2[i])
            rmse_str += " {}: {}".format(titles[i], rmse[i])
            nrmse_str += " {}: {}".format(titles[i], nrmse[i])

            inv_mse_str += " {}: {}".format(titles[i], inv_mse[i])
            inv_r2_str += " {}: {}".format(titles[i], inv_r2[i])
            inv_rmse_str += " {}: {}".format(titles[i], inv_rmse[i])
            inv_nrmse_str += " {}: {}".format(titles[i], inv_nrmse[i])

        # Print computed metrics for preprocessed and original data
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
        # Create deep copies for log-scale transformations
        log_inv_targets_arr = copy.deepcopy(inv_targets_arr)
        log_inv_predictions_arr = copy.deepcopy(inv_predictions_arr)

        # Apply log10 transform to selected channels (0 and 2) representing peaks or main values
        log_inv_targets_arr[:, 0] = np.log10(log_inv_targets_arr[:, 0])
        log_inv_targets_arr[:, 2] = np.log10(log_inv_targets_arr[:, 2])

        log_inv_predictions_arr[:, 0] = np.log10(log_inv_predictions_arr[:, 0])
        log_inv_predictions_arr[:, 2] = np.log10(log_inv_predictions_arr[:, 2])

        # Compute metrics on log-transformed data
        log_inv_mse, log_inv_rmse, log_inv_nrmse, log_inv_r2 = get_mse_nrmse_r2(log_inv_targets_arr, log_inv_predictions_arr)

        # Prepare strings for log-transformed metrics
        mse_str, inv_mse_str = "MSE", "LOG Original data MSE"
        r2_str, inv_r2_str = "R2", "LOG Original data R2"
        rmse_str, inv_rmse_str = "RMSE", "LOG Original data RMSE"
        nrmse_str, inv_nrmse_str = "NRMSE", "LOG Original data NRMSE"
        # Append log metric values for each output variable
        for i in range(len(mse)):
            inv_mse_str += " {}: {}".format(titles[i], log_inv_mse[i])
            inv_r2_str += " {}: {}".format(titles[i], log_inv_r2[i])
            inv_rmse_str += " {}: {}".format(titles[i], log_inv_rmse[i])
            inv_nrmse_str += " {}: {}".format(titles[i], log_inv_nrmse[i])

        # Print log-transformed metrics
        print(inv_mse_str)
        print(inv_r2_str)
        print(inv_rmse_str)
        print(inv_nrmse_str)

        # Evaluate and print metrics for eigenvalues of 3D tensors for both original and inverse data
        get_mse_nrmse_r2_eigh_3D(targets_list, predictions_list)
        print("ORIGINAL DATA")
        get_mse_nrmse_r2_eigh_3D(inv_targets_arr, inv_predictions_arr)

        # Print detailed log-transformed metric arrays and their means
        print("log_inv_r2 ", log_inv_r2)
        print("log_inv_nrmse ", log_inv_nrmse)

        print("mean log_inv_r2 ", np.mean(log_inv_r2))
        print("mean log_inv_nrmse ", np.mean(log_inv_nrmse))

        # Plot predictions vs targets for preprocessed, original, and log original data
        plot_target_prediction(np.array(targets_list), np.array(predictions_list), "preprocessed_", x_labels=x_labels, titles=titles)
        plot_target_prediction(inv_targets_arr, inv_predictions_arr, x_labels=x_labels, titles=titles)

        plot_target_prediction(log_inv_targets_arr, log_inv_predictions_arr, title_prefix="log_orig_", r2=log_inv_r2, nrmse=log_inv_nrmse,
                               x_labels=x_labels, titles=titles)

        # Recalculate log metrics (redundant but performed again)
        log_inv_mse, log_inv_rmse, log_inv_nrmse, log_inv_r2 = get_mse_nrmse_r2(log_inv_targets_arr, log_inv_predictions_arr)

        # Print main peak metrics for log-transformed data
        print("log_inv_r2 main peak", log_inv_r2)
        print("log_inv_nrmse main peak", log_inv_nrmse)


def load_study(results_dir):
    # Load the Optuna study object from a pickle file in the results directory
    study = joblib.load(os.path.join(results_dir, "study.pkl"))

    # Print summary info about the best trial found so far
    print("Best trial until now:")
    print(" Value: ", study.best_trial.value)
    print(" Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    # Return the loaded study object
    return study


if __name__ == "__main__":
    # Setup command line argument parser
    parser = argparse.ArgumentParser()
    # Argument for directory where results are saved
    parser.add_argument('results_dir', help='results directory')
    # Argument for the data zarr file/directory
    parser.add_argument('data_zarr', help='data_zarr')
    # Optional flag to enable CUDA (GPU) usage
    parser.add_argument("-c", "--cuda", default=False, action='store_true', help="use cuda")

    # Parse the command line arguments (excluding script name)
    args = parser.parse_args(sys.argv[1:])

    # Load the Optuna study from results directory
    study = load_study(args.results_dir)

    # Load models and run evaluation using the loaded study and arguments
    load_models(args, study)
