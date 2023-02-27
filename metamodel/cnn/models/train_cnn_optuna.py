import os
import sys
import argparse
import logging
import joblib
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler
import time
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from metamodel.cnn.models.trials.net_optuna import Net
from metamodel.cnn.datasets.dfm_dataset import DFMDataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from metamodel.cnn.models.auxiliary_functions import get_mean_std, log_data
from metamodel.cnn.models.train_pure_cnn_optuna import train_one_epoch, prepare_dataset, validate, load_trials_config


def objective(trial, trials_config, train_loader, validation_loader):
    best_vloss = 1_000_000.

    loss_fn_name = nn.MSELoss
    max_channel = trial.suggest_categorical("max_channel", trials_config["max_channel"])
    n_conv_layers = trial.suggest_categorical("n_conv_layers", trials_config["n_conv_layers"])
    kernel_size = trial.suggest_categorical("kernel_size", trials_config["kernel_size"])
    stride = trial.suggest_categorical("stride", trials_config["stride"])
    pool = trial.suggest_categorical("pool", trials_config["pool"])
    pool_size = trial.suggest_categorical("pool_size", trials_config["pool_size"])
    pool_stride = trial.suggest_categorical("pool_stride", trials_config["pool_stride"])
    lr = trial.suggest_categorical("lr", trials_config["lr"])
    use_batch_norm = trial.suggest_categorical("use_batch_norm", trials_config["use_batch_norm"])
    max_hidden_neurons = trial.suggest_categorical("max_hidden_neurons", trials_config["max_hidden_neurons"])
    n_hidden_layers = trial.suggest_categorical("n_hidden_layers", trials_config["n_hidden_layers"])

    optimizer_name = "Adam"  # trial.suggest_categorical("optimizer", ["Adam"])
    hidden_activation = F.relu

    # Initilize model
    model_kwargs = {"n_conv_layers": n_conv_layers,
                    "max_channel": max_channel,
                    "pool": pool,
                    "pool_size": pool_size,
                    "kernel_size": kernel_size,
                    "stride": stride,
                    "pool_stride": pool_stride,
                    "use_batch_norm": use_batch_norm,
                    "n_hidden_layers": n_hidden_layers,
                    "max_hidden_neurons": max_hidden_neurons,
                    "hidden_activation": hidden_activation
                    }

    model = Net(trial, **model_kwargs).to(device)

    # Initialize optimizer
    optimizer_kwargs = {"lr": lr}
    optimizer = getattr(optim, optimizer_name)(params=model.parameters(), **optimizer_kwargs)

    trial.set_user_attr("model_class", model.__class__)
    trial.set_user_attr("optimizer_class", optimizer.__class__)
    trial.set_user_attr("model_name", model._name)
    trial.set_user_attr("model_kwargs", model_kwargs)
    trial.set_user_attr("optimizer_kwargs", optimizer_kwargs)
    trial.set_user_attr("loss_fn_name", loss_fn_name)

    # Training of the model
    start_time = time.time()
    avg_loss_list = []
    avg_vloss_list = []
    avg_vloss = best_vloss
    best_epoch = 0
    model_state_dict = []
    optimizer_state_dict = []

    model_path = 'trial_{}_losses_model_{}'.format(trial.number, model._name)
    if os.path.exists(model_path):
        return avg_vloss

    for epoch in range(config["num_epochs"]):
        try:
            model.train(True)
            avg_loss = train_one_epoch(model, optimizer, train_loader, config, loss_fn=loss_fn_name(), use_cuda=use_cuda)  # Train the model
            model.train(False)
            avg_vloss = validate(model, validation_loader, loss_fn=loss_fn_name(), use_cuda=use_cuda)   # Evaluate the model

            avg_loss_list.append(avg_loss)
            avg_vloss_list.append(avg_vloss)

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                best_epoch = epoch

                model_state_dict = model.state_dict()
                optimizer_state_dict = optimizer.state_dict()

            # For pruning (stops trial early if not promising)
            trial.report(avg_vloss, epoch)
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        except Exception as e:
            print(str(e))
            return avg_vloss

    for key, value in trial.params.items():
        model_path += "_{}_{}".format(key, value)
    model_path = os.path.join(output_dir, model_path)

    torch.save({
        'best_epoch': best_epoch,
        'best_model_state_dict': model_state_dict,
        'best_optimizer_state_dict': optimizer_state_dict,
        'train_loss': avg_loss_list,
        'valid_loss': avg_vloss_list,
        'training_time': time.time() - start_time,
    }, model_path)

    return avg_vloss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('trials_config_path', help='Path tp trials config')
    parser.add_argument('data_dir', help='Data directory')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument("-c", "--cuda", default=False, action='store_true', help="use cuda")
    parser.add_argument("-a", "--append", default=False, action='store_true', help="append models")

    args = parser.parse_args(sys.argv[1:])

    data_dir = args.data_dir
    output_dir = args.output_dir
    trials_config = load_trials_config(args.trials_config_path)

    use_cuda = args.cuda

    config = {"num_epochs": 5,
              "batch_size_train": 25,
              "batch_size_test": 250,
              "train_samples_ratio": 0.8,
              "val_samples_ratio": 0.2,
              "print_batches": 10,
              "log_input": True,
              "normalize_input": True,
              "log_output": False,
              "normalize_output": True}

    # Optuna params
    num_trials = 2

    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    print("device ", device)

    # Make runs repeatable
    random_seed = 12345
    torch.backends.cudnn.enabled = False  # Disable cuDNN use of nondeterministic algorithms
    torch.manual_seed(random_seed)
    output_dir = os.path.join(output_dir, "seed_{}".format(random_seed))
    if os.path.exists(output_dir) and not args.append:
        raise IsADirectoryError("Results output dir {} already exists".format(output_dir))
    if not args.append:
        os.mkdir(output_dir)
    elif not os.path.exists(output_dir):
        raise NotADirectoryError("output dir {} not exists".format(output_dir))

    study = optuna.create_study(sampler=TPESampler(seed=random_seed), direction="minimize")

    # ================================
    # Datasets and data loaders
    # ================================
    train_set, validation_set, test_set = prepare_dataset(study, config, data_dir=data_dir, serialize_path=output_dir)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size_train"], shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=config["batch_size_test"], shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config["batch_size_test"], shuffle=False)

    def obj_func(trial):
        return objective(trial, trials_config, train_loader, validation_loader)



    study.optimize(obj_func, n_trials=num_trials)


    # ================================
    # Results
    # ================================
    # Find number of pruned and completed trials
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # Display the study statistics
    print("\nStudy statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    trial = study.best_trial
    print("Best trial:")
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Save results to csv file
    df = study.trials_dataframe().drop(['datetime_start', 'datetime_complete', 'duration'], axis=1)  # Exclude columns
    df = df.loc[df['state'] == 'COMPLETE']        # Keep only results that did not prune
    df = df.drop('state', axis=1)                 # Exclude state column
    df = df.sort_values('value')                  # Sort based on accuracy
    df.to_csv(os.path.join(output_dir, 'optuna_results.csv'), index=False)  # Save to csv file

    # Display results in a dataframe
    print("\nOverall Results (ordered by accuracy):\n {}".format(df))

    # Find the most important hyperparameters
    most_important_parameters = optuna.importance.get_param_importances(study, target=None)

    # Display the most important hyperparameters
    print('\nMost important hyperparameters:')
    for key, value in most_important_parameters.items():
        print('  {}:{}{:.2f}%'.format(key, (15-len(key))*' ', value*100))

    # serialize optuna study object
    joblib.dump(study, os.path.join(output_dir, "study.pkl"))
