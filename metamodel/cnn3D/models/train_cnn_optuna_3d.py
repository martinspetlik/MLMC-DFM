import os
import shutil
import sys
import argparse
import joblib
import torch
import torch.nn.functional as F
import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler, BruteForceSampler
import time
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
from metamodel.cnn3D.models.net_optuna import Net
from metamodel.cnn.models.auxiliary_functions import check_shapes, get_loss_fn
from metamodel.cnn3D.models.train_pure_cnn_optuna_3d import train_one_epoch, prepare_sub_datasets, prepare_dataset, validate, load_trials_config

def objective(trial, trials_config, train_loader, validation_loader, load_existing=False):
    best_vloss = 1_000_000.  # Initialize best validation loss to a large number
    best_epoch = 0  # Track the epoch with the best validation loss
    save_model_best_epoch = 0  # Epoch at which the best model is saved

    # Suggest hyperparameters from the search space defined in trials_config
    max_channel = trial.suggest_categorical("max_channel", trials_config["max_channel"])
    n_conv_layers = trial.suggest_categorical("n_conv_layers", trials_config["n_conv_layers"])
    kernel_size = trial.suggest_categorical("kernel_size", trials_config["kernel_size"])
    stride = trial.suggest_categorical("stride", trials_config["stride"])
    # pool = trial.suggest_categorical("pool", trials_config["pool"])
    pool_size = trial.suggest_categorical("pool_size", trials_config["pool_size"])
    pool_stride = trial.suggest_categorical("pool_stride", trials_config["pool_stride"])
    lr = trial.suggest_categorical("lr", trials_config["lr"])
    use_batch_norm = trial.suggest_categorical("use_batch_norm", trials_config["use_batch_norm"])
    max_hidden_neurons = trial.suggest_categorical("max_hidden_neurons", trials_config["max_hidden_neurons"])
    n_hidden_layers = trial.suggest_categorical("n_hidden_layers", trials_config["n_hidden_layers"])

    # Set training batch size in config
    batch_size_train = trial.suggest_categorical("batch_size_train", trials_config["batch_size_train"])
    config["batch_size_train"] = batch_size_train

    # Choose loss function, default to MSE
    loss_function = ["MSE", []]
    if "loss_function" in trials_config:
        loss_function = trial.suggest_categorical("loss_function", trials_config["loss_function"])

    # Get loss function object
    loss_fn = get_loss_fn(loss_function)
    loss_fn_validation = None

    pool_indices = None
    if "pool_indices" in trials_config:
        pool_indices = trial.suggest_categorical("pool_indices", trials_config["pool_indices"])

    padding = 0  # Default padding
    if "padding" in trials_config:
        padding = trial.suggest_categorical("padding", trials_config["padding"])

    activation_before_pool = False  # Default flag
    if "activation_before_pool" in trials_config:
        activation_before_pool = trial.suggest_categorical("activation_before_pool",
                                                           trials_config["activation_before_pool"])
    # CNN dropout options
    if "use_cnn_dropout" in trials_config:
        use_cnn_dropout = trial.suggest_categorical("use_cnn_dropout", trials_config["use_cnn_dropout"])

    # FC dropout options
    if "use_fc_dropout" in trials_config:
        use_fc_dropout = trial.suggest_categorical("use_fc_dropout", trials_config["use_fc_dropout"])

    # Dropout index and ratio selections
    if "cnn_dropout_indices" in trials_config:
        cnn_dropout_indices = trial.suggest_categorical("cnn_dropout_indices", trials_config["cnn_dropout_indices"])

    if "fc_dropout_indices" in trials_config:
        fc_dropout_indices = trial.suggest_categorical("fc_dropout_indices", trials_config["fc_dropout_indices"])

    if "bias_reduction_layer_indices" in trials_config:
        bias_reduction_layer_indices = trial.suggest_categorical("bias_reduction_layer_indices",
                                                                 trials_config["bias_reduction_layer_indices"])

    if "cnn_dropout_ratios" in trials_config:
        cnn_dropout_ratios = trial.suggest_categorical("cnn_dropout_ratios", trials_config["cnn_dropout_ratios"])

    if "fc_dropout_ratios" in trials_config:
        fc_dropout_ratios = trial.suggest_categorical("fc_dropout_ratios", trials_config["fc_dropout_ratios"])

    # If sample sizes are specified, override the default dataset loading
    if "n_train_samples" in trials_config and trials_config["n_train_samples"] is not None:
        n_train_samples = trial.suggest_categorical("n_train_samples", trials_config["n_train_samples"])
        config["n_train_samples"] = n_train_samples

        if "n_val_samples" in trials_config and trials_config["n_val_samples"] is not None:
            config["n_val_samples"] = trial.suggest_categorical("n_val_samples", trials_config["n_val_samples"])

        if "n_test_samples" in trials_config and trials_config["n_test_samples"] is not None:
            config["n_test_samples"] = trial.suggest_categorical("n_test_samples", trials_config["n_test_samples"])

        # Prepare datasets (subsets or full)
        if "sub_datasets" in config and len(config["sub_datasets"]) > 0:
            train_set, validation_set, test_set = prepare_sub_datasets(study, config, data_dir=data_dir,
                                                                       serialize_path=output_dir)
        else:
            train_set, validation_set, test_set = prepare_dataset(study, config, data_dir=data_dir,
                                                                  serialize_path=output_dir)

        print("len(trainset): {}, len(valset): {}, len(testset): {}".format(len(train_set), len(validation_set),
                                                                            len(test_set)))

        # Initialize DataLoaders
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size_train"], shuffle=True)

        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=config["batch_size_train"],
                                                        shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=config["batch_size_test"], shuffle=False)

    # If using correlated output loss, set covariance matrices
    print("loss function ", loss_function)
    if loss_function[0] == "CorrelatedOutputLoss":
        print("loss fn set cov")
        loss_fn.set_cov(train_loader)

        loss_fn_validation = get_loss_fn(loss_function)
        loss_fn_validation.set_cov(validation_loader)

        np.save("cov_mat_validation_set", loss_fn_validation.covariance_matrix.cpu())
        np.save("cov_mat_training_set", loss_fn.covariance_matrix.cpu())

        print("loss_fn_validation ", loss_fn_validation)

    # Optimizer choice
    optimizer_name = "Adam"
    if "optimizer_name" in trials_config:
        optimizer_name = trial.suggest_categorical("optimizer_name", trials_config["optimizer_name"])

    # L2 regularization
    L2_penalty = 0
    if "L2_penalty" in trials_config:
        L2_penalty = trial.suggest_categorical("L2_penalty", trials_config["L2_penalty"])

    # CNN activation function
    cnn_activation_name = "relu"
    if "cnn_activation_name" in trials_config:
        cnn_activation_name = trial.suggest_categorical("cnn_activation_name",
                                                        trials_config["cnn_activation_name"])
    cnn_activation = getattr(F, cnn_activation_name)

    # Hidden layer activation function
    hidden_activation_name = "relu"
    if "hidden_activation_name" in trials_config:
        hidden_activation_name = trial.suggest_categorical("hidden_activation_name",
                                                           trials_config["hidden_activation_name"])
    hidden_activation = getattr(F, hidden_activation_name)

    # Check whether CNN configuration is valid and get flattened input size
    flag, input_size = check_shapes(n_conv_layers, kernel_size, stride, pool_size, pool_stride, pool_indices,
                                    input_size=trials_config["input_size"], padding=padding)

    # Use global pooling if configured
    if "global_pool" in trials_config:
        global_pool = trial.suggest_categorical("global_pool", trials_config["global_pool"])

    # If shape check fails, exit trial early
    if flag == -1:
        print("flag: {}".format(flag))
        return best_vloss

    # Initialize model arguments dictionary with hyperparameters and configurations
    model_kwargs = {"n_conv_layers": n_conv_layers,
                    "max_channel": max_channel,
                    "activation_before_pool": activation_before_pool,
                    "pool_size": pool_size,
                    "kernel_size": kernel_size,
                    "stride": stride,
                    "pool_stride": pool_stride,
                    "use_batch_norm": use_batch_norm,
                    "n_hidden_layers": n_hidden_layers,
                    "max_hidden_neurons": max_hidden_neurons,
                    "hidden_activation": hidden_activation,
                    "cnn_activation": cnn_activation,
                    "global_pool": global_pool if "global_pool" in trials_config else None,
                    "input_size": trials_config["input_size"],
                    "output_bias": trials_config["output_bias"] if "output_bias" in trials_config else False,
                    "pool_indices": pool_indices if "pool_indices" in trials_config else {},
                    "use_cnn_dropout": use_cnn_dropout if "use_cnn_dropout" in trials_config else False,
                    "use_fc_dropout": use_fc_dropout if "use_fc_dropout" in trials_config else False,
                    "cnn_dropout_indices": cnn_dropout_indices if "cnn_dropout_indices" in trials_config else [],
                    "fc_dropout_indices": fc_dropout_indices if "fc_dropout_indices" in trials_config else [],
                    "cnn_dropout_ratios": cnn_dropout_ratios if "cnn_dropout_ratios" in trials_config else [],
                    "fc_dropout_ratios": fc_dropout_ratios if "fc_dropout_ratios" in trials_config else [],
                    "padding": padding,
                    }

    # Optionally add bias reduction layer indices if specified
    if "bias_reduction_layer_indices" in trials_config:
        model_kwargs["bias_reduction_layer_indices"] = bias_reduction_layer_indices

    # Set input channels count if available in config
    if "input_channels" in trials_config:
        model_kwargs["input_channel"] = len(trials_config["input_channels"])

    # Set output neurons count if output channels specified
    if "output_channels" in trials_config:
        model_kwargs["n_output_neurons"] = len(trials_config["output_channels"])

    # Determine the model class name, default is "Net"
    model_class_name = "Net"
    if "model_class_name" in trials_config:
        model_class_name = trials_config["model_class_name"]

    # Load model file path if provided in config
    if "model_file_path" in trials_config:
        model_file_path = trials_config["model_file_path"]

    # Select the model class based on the class name string
    if model_class_name == "Net":
        model_class = Net
    else:
        # Raise error if an unsupported model class is requested
        raise NotImplementedError("model class name {} not supported ".format(model_class_name))

    # Instantiate the model and move it to the specified device (CPU/GPU)
    model = model_class(**model_kwargs).to(device)

    # Prepare optimizer keyword arguments with learning rate and weight decay
    optimizer_kwargs = {"lr": lr, "weight_decay": L2_penalty}

    # Filter parameters that require gradient updates (non-frozen)
    non_frozen_parameters = [p for p in model.parameters() if p.requires_grad]

    optimizer = None
    # Initialize optimizer only if there are parameters to optimize
    if len(non_frozen_parameters) > 0:
        optimizer = getattr(optim, optimizer_name)(params=non_frozen_parameters, **optimizer_kwargs)

    # Store model and optimizer info as user attributes for trial tracking
    trial.set_user_attr("model_class", model.__class__)
    trial.set_user_attr("optimizer_class", optimizer.__class__)
    trial.set_user_attr("model_name", model._name)
    trial.set_user_attr("model_kwargs", model_kwargs)
    trial.set_user_attr("optimizer_kwargs", optimizer_kwargs)
    trial.set_user_attr("loss_fn", loss_fn)
    trial.set_user_attr("trials_config", trials_config)

    # Initialize learning rate scheduler to None
    scheduler = None

    # Determine if training is enabled in config, default True
    train = trials_config["train"] if "train" in trials_config else True

    # Setup learning rate scheduler if configured and optimizer exists
    if "scheduler" in trials_config and optimizer is not None:
        # Suggest scheduler type via trial hyperparameter tuning
        trial_scheduler = trial.suggest_categorical("scheduler", trials_config["scheduler"])
        # Use ReduceLROnPlateau scheduler if specified
        if "class" in trial_scheduler:
            if trial_scheduler["class"] == "ReduceLROnPlateau":
                scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                           patience=trial_scheduler["patience"],
                                                           factor=trial_scheduler["factor"])
            else:
                # Otherwise use StepLR scheduler with given step size and decay gamma
                scheduler = lr_scheduler.StepLR(optimizer, step_size=trial_scheduler["step_size"],
                                                gamma=trial_scheduler["gamma"])

    # If loading an existing model checkpoint is requested
    if load_existing:
        # Ensure the model file exists
        if not os.path.exists(model_file_path):
            raise FileNotFoundError("Model file not found at path: {}".format(model_file_path))

        # Load checkpoint from file
        checkpoint = torch.load(model_file_path)
        # Load model state dict from checkpoint
        model.load_state_dict(checkpoint['best_model_state_dict'])
        # Load optimizer state dict from checkpoint
        optimizer.load_state_dict(checkpoint['best_optimizer_state_dict'])
        # Load scheduler state dict from checkpoint
        scheduler.load_state_dict(checkpoint['best_scheduler_state_dict'])
        # Load the epoch at which best model was saved
        save_model_best_epoch = checkpoint['best_epoch']
        # Initialize best validation loss from checkpoint history
        best_vloss = np.min(checkpoint['valid_loss'])

    # Initialize timing for training duration
    start_time = time.time()
    # Initialize lists to store training and validation losses per epoch
    avg_loss_list = []
    avg_vloss_list = []
    # Initialize average losses with best validation loss (for comparison)
    avg_vloss, avg_loss = best_vloss, best_vloss

    # Initialize placeholders for model, optimizer, and scheduler state dicts
    model_state_dict = {}
    optimizer_state_dict = {}
    scheduler_state_dict = {}

    # Define base path for saving model checkpoints related to this trial
    model_path = 'trial_{}_losses_model_{}'.format(trial.number, model._name)

    # Main training loop over epochs
    for epoch in range(config["num_epochs"]):
        # Adjust epoch count if resuming from saved checkpoint
        epoch = save_model_best_epoch + epoch
        # Run training if enabled
        if train:
            model.train(True)  # Set model to training mode
            # Train one epoch and get average training loss
            avg_loss = train_one_epoch(model, optimizer, train_loader, config, loss_fn=loss_fn, use_cuda=use_cuda)

        # Switch model to evaluation mode for validation
        model.train(False)
        if len(validation_set) == 0:
            # If no validation set, use training loss as validation loss
            avg_vloss = avg_loss
            avg_vacc = 0
        else:
            # Use separate validation loss function if provided, else use training loss function
            if loss_fn_validation is None:
                loss_fn_validation = loss_fn
            # Validate model and get average validation loss and accuracy
            avg_vloss, avg_vacc = validate(model, validation_loader, loss_fn=loss_fn_validation, use_cuda=use_cuda)

        # Step the scheduler if it exists, using validation loss for ReduceLROnPlateau
        if scheduler is not None:
            scheduler.step(avg_vloss)
            print("scheduler lr: {}".format(scheduler._last_lr))

        # Append losses for tracking
        avg_loss_list.append(avg_loss)
        avg_vloss_list.append(avg_vloss)

        # Print epoch statistics for loss and accuracy
        print("epoch: {}, LOSS train: {}, val: {}, ACC val: {}".format(epoch, avg_loss, avg_vloss, avg_vacc))

        # Check if current validation loss is the best so far
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            best_epoch = epoch

            # Save current best model parameters
            model_state_dict = model.state_dict()
            if train:
                optimizer_state_dict = optimizer.state_dict()

            # Compose path for saving this best model checkpoint
            model_path_epoch = os.path.join(output_dir, model_path + "_best_{}".format(epoch))

            # Save scheduler state dict as well
            scheduler_state_dict = scheduler.state_dict()

            # Save checkpoint containing best model, optimizer, scheduler, losses, and training time
            torch.save({
                'best_epoch': best_epoch,
                'best_model_state_dict': model_state_dict,
                'best_optimizer_state_dict': optimizer_state_dict,
                'best_scheduler_state_dict': scheduler_state_dict,
                'train_loss': avg_loss_list,
                'valid_loss': avg_vloss_list,
                'training_time': time.time() - start_time,
            }, model_path_epoch)

        # Report intermediate validation loss to Optuna for pruning decisions
        trial.report(avg_vloss, epoch)
        # Pruning code is commented out but could be enabled:
        # if trial.should_prune():
        #     raise optuna.exceptions.TrialPruned()

    # Construct final model save path after training completes
    model_path = os.path.join(output_dir, model_path)

    # Save final model checkpoint with best model and training stats
    torch.save({
        'best_epoch': best_epoch,
        'best_model_state_dict': model_state_dict,
        'best_optimizer_state_dict': optimizer_state_dict,
        'best_scheduler_state_dict': scheduler_state_dict,
        'train_loss': avg_loss_list,
        'valid_loss': avg_vloss_list,
        'training_time': time.time() - start_time,
    }, model_path)

    # Return the best validation loss obtained during training
    return best_vloss


if __name__ == '__main__':
    # Create argument parser for command line inputs
    parser = argparse.ArgumentParser()
    # Add positional argument for trials configuration file path
    parser.add_argument('trials_config_path', help='Path tp trials config')
    # Add positional argument for data directory
    parser.add_argument('data_dir', help='Data directory')
    # Add positional argument for output directory
    parser.add_argument('output_dir', help='Output directory')
    # Add optional flag to enable CUDA usage
    parser.add_argument("-c", "--cuda", default=False, action='store_true', help="use cuda")
    # Add optional flag to append models instead of overwriting
    parser.add_argument("-a", "--append", default=False, action='store_true', help="append models")

    # Parse command line arguments
    args = parser.parse_args(sys.argv[1:])

    # Assign parsed arguments to variables
    data_dir = args.data_dir
    output_dir = args.output_dir
    # Load trials configuration from specified file path
    trials_config = load_trials_config(args.trials_config_path)
    use_cuda = args.cuda

    # Build config dictionary by extracting parameters from trials_config or setting defaults
    config = {"num_epochs": trials_config["num_epochs"],
              "batch_size_train": trials_config["batch_size_train"],
              "batch_size_test": trials_config["batch_size_test"] if "batch_size_test" in trials_config else 250,
              "n_train_samples": trials_config["n_train_samples"] if "n_train_samples" in trials_config else None,
              "n_test_samples": trials_config["n_test_samples"] if "n_test_samples" in trials_config else None,
              "train_samples_ratio": trials_config["train_samples_ratio"] if "train_samples_ratio" in trials_config else 0.8,
              "val_samples_ratio": trials_config["val_samples_ratio"] if "val_samples_ratio" in trials_config else 0.2,
              "print_batches": 10,
              "init_norm": trials_config["init_norm"] if "init_norm" in trials_config else False,
              "init_norm_use_all_features": trials_config["init_norm_use_all_features"] if "init_norm_use_all_features" in trials_config else False,
              "log_all_input_channels": trials_config["log_all_input_channels"] if "log_all_input_channels" in trials_config else False,
              "log_input": trials_config["log_input"] if "log_input" in trials_config else True,
              "normalize_input": trials_config["normalize_input"] if "normalize_input" in trials_config else True,
              "log_output": trials_config["log_output"] if "log_output" in trials_config else False,
              "log10_output": trials_config["log10_output"] if "log10_output" in trials_config else False,
              "log_all_output": trials_config["log_all_output"] if "log_all_output" in trials_config else False,
              "log10_all_output": trials_config["log10_all_output"] if "log10_all_output" in trials_config else False,
              "normalize_output": trials_config["normalize_output"] if "normalize_output" in trials_config else True,
              "input_channels": trials_config["input_channels"] if "input_channels" in trials_config else None,
              "output_channels": trials_config["output_channels"] if "output_channels" in trials_config else None,
              "fractures_sep": trials_config["fractures_sep"] if "fractures_sep" in trials_config else False,
              "cross_section": trials_config["cross_section"] if "cross_section" in trials_config else False,
              "seed": trials_config["random_seed"] if "random_seed" in trials_config else 12345,
              "output_dir": output_dir,
              "sub_datasets": trials_config["sub_datasets"] if "sub_datasets" in trials_config else {}
              }

    # Add optional transformations to config if present in trials_config
    if "input_transform" in trials_config:
        config["input_transform"] = trials_config["input_transform"]
    if "output_transform" in trials_config:
        config["output_transform"] = trials_config["output_transform"]
    if "output_iqr_scale" in trials_config:
        config["output_iqr_scale"] = trials_config["output_iqr_scale"]
    if "normalize_input_indices" in trials_config:
        config["normalize_input_indices"] = trials_config["normalize_input_indices"]
    if "normalize_output_indices" in trials_config:
        config["normalize_output_indices"] = trials_config["normalize_output_indices"]

    # Extract number of trials for Optuna optimization
    num_trials = trials_config["num_trials"]

    # Set device to CUDA if available and requested, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    print("device ", device)
    print("config seed ", config["seed"])

    # Make experiment runs repeatable by fixing random seed
    random_seed = trials_config["random_seed"]
    torch.backends.cudnn.enabled = False  # Disable cuDNN nondeterministic algorithms
    torch.manual_seed(random_seed)
    # Append seed info to output directory path
    output_dir = os.path.join(output_dir, "seed_{}".format(random_seed))
    # Remove existing output directory unless appending models
    if os.path.exists(output_dir) and not args.append:
        shutil.rmtree(output_dir)
    # Create output directory if not appending
    if not args.append:
        os.mkdir(output_dir)
    # Check output directory existence when appending
    elif not os.path.exists(output_dir):
        raise NotADirectoryError("output dir {} not exists".format(output_dir))

    # Initialize Optuna sampler with seed
    sampler = TPESampler(seed=random_seed)
    # Override sampler if specified in config (e.g., BruteForceSampler)
    if "sampler_class" in trials_config:
        if trials_config["sampler_class"] == "BruteForceSampler":
            sampler = BruteForceSampler(seed=random_seed)

    # Determine whether to load existing study based on append flag
    load_existing = args.append
    # Create Optuna study object for hyperparameter optimization
    study = optuna.create_study(sampler=sampler, direction="minimize")

    # ================================
    # Datasets and data loaders
    # ================================
    train_loader, validation_loader = None, None
    # Placeholder for dataset preparation (commented out)
    # This would normally prepare train, validation, and test datasets and data loaders

    # Define objective function wrapper for Optuna
    def obj_func(trial):
        return objective(trial, trials_config, train_loader, validation_loader, load_existing=load_existing)

    # Run optimization for specified number of trials
    study.optimize(obj_func, n_trials=num_trials)

    # ================================
    # Results
    # ================================
    # Retrieve pruned and completed trials for reporting
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # Print statistics about the study
    print("\nStudy statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    # Retrieve best trial information
    trial = study.best_trial
    print("Best trial:")
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Save trials results to CSV excluding some columns and pruning states
    df = study.trials_dataframe().drop(['datetime_start', 'datetime_complete', 'duration'], axis=1)  # Exclude columns
    df = df.loc[df['state'] == 'COMPLETE']        # Keep only complete trials
    df = df.drop('state', axis=1)                 # Remove state column
    df = df.sort_values('value')                  # Sort by trial value (objective metric)
    df.to_csv(os.path.join(output_dir, 'optuna_results.csv'), index=False)  # Write results to CSV file

    # Print overall results in a dataframe format
    print("\nOverall Results (ordered by accuracy):\n {}".format(df))

    # Attempt to compute and display most important hyperparameters
    try:
        most_important_parameters = optuna.importance.get_param_importances(study, target=None)

        # Print hyperparameter importance ranking
        print('\nMost important hyperparameters:')
        for key, value in most_important_parameters.items():
            print('  {}:{}{:.2f}%'.format(key, (15-len(key))*' ', value*100))
    except Exception as e:
        # Print any exceptions encountered during importance calculation
        print(str(e))

    # Serialize and save the Optuna study object for later use
    joblib.dump(study, os.path.join(output_dir, "study.pkl"))

