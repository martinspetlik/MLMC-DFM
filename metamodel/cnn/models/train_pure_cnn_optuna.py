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
from metamodel.cnn.models.trials.net_optuna_2 import Net
from metamodel.cnn.datasets.dfm_dataset import DFMDataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from metamodel.cnn.models.auxiliary_functions import get_mean_std, log_data


def train_one_epoch(model, optimizer, train_loader, config, loss_fn=nn.MSELoss(), use_cuda=True):
    """
    Train NN
    :param model:
    :param optimizer:
    :param loss_fn:
    :return:
    """
    running_loss = 0.

    for i, data in enumerate(train_loader):
        inputs, targets = data

        if torch.cuda.is_available() and use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        inputs = inputs.float()
        targets = targets.float()

        optimizer.zero_grad()

        outputs = torch.squeeze(model(inputs))

        loss = loss_fn(outputs, targets)
        loss.backward()

        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

    train_loss = running_loss / config["print_batches"]
    return train_loss


def validate(model, validation_loader, loss_fn=nn.MSELoss(), use_cuda=False):
    """
    Validate model
    :param model:
    :param loss_fn:
    :return:
    """
    running_vloss = 0.0
    for i, vdata in enumerate(validation_loader):
        vinputs, vtargets = vdata

        if torch.cuda.is_available() and use_cuda:
            vinputs = vinputs.cuda()
            vtargets = vtargets.cuda()

        vinputs = vinputs.float()
        vtargets = vtargets.float()

        voutputs = torch.squeeze(model(vinputs))
        vloss = loss_fn(voutputs, vtargets)
        running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)

    return avg_vloss


def objective(trial, train_loader, validation_loader):
    best_vloss = 1_000_000.
    # Settings
    max_channel = 3 #trial.suggest_categorical("max_channel",[3, 32, 64, 128])
    kernel_size = 3 #trial.suggest_int("kernel_size", 3)
    #stride = trial.suggest_int("stride", 2, 3)
    stride = 2
    #pool = trial.suggest_categorical("pool", [None, "max", "avg"])
    # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    lr = trial.suggest_categorical("lr", [1e-3]) #trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    use_batch_norm = True
    loss_fn_name = nn.MSELoss
    # max_channel = trial.suggest_int("max_channel", 3)
    # kernel_size = trial.suggest_int("kernel_size", 3)
    # stride = trial.suggest_int("stride", 2)
    pool = trial.suggest_categorical("pool", [None])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
    #lr = trial.suggest_float("lr", 1e-3)

    # Initilize model
    model_kwargs = {"pool": pool,
                    "max_channel": max_channel,
                    "kernel_size": kernel_size,
                    "stride": stride,
                    "use_batch_norm": use_batch_norm}
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
    for epoch in range(config["num_epochs"]):
        avg_loss = train_one_epoch(model, optimizer, train_loader, config, loss_fn=loss_fn_name(), use_cuda=use_cuda)  # Train the model
        avg_vloss = validate(model, validation_loader, loss_fn=loss_fn_name(), use_cuda=use_cuda)   # Evaluate the model

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(model._name, epoch)
            trial.set_user_attr("epoch", epoch)
            for key, value in trial.params.items():
                model_path += "_{}_{}".format(key, value)
            model_path = os.path.join(output_dir, model_path)

            torch.save({
                'epoch': int(epoch) + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_loss,
                'valid_loss': avg_vloss,
                'training_time': time.time() - start_time,
            }, model_path)

        # For pruning (stops trial early if not promising)
        trial.report(avg_vloss, epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return avg_vloss


def prepare_dataset(study, config, data_dir, serialize_path=None):
    # ===================================
    # Get mean and std for each channel
    # ===================================
    input_transform, output_transform = None, None
    input_mean, output_mean, input_std, output_std = 0, 0, 1, 1
    if config["log_input"]:
        input_transform = transforms.Compose([transforms.Lambda(log_data)])
    if config["log_output"]:
        output_transform = transforms.Compose([transforms.Lambda(log_data)])

    if config["normalize_input"]:
        dataset_for_mean_std = DFMDataset(data_dir=data_dir,
                                          input_transform=input_transform,
                                          output_transform=output_transform,
                                          two_dim=True)

        n_train_samples = int(len(dataset_for_mean_std) * config["train_samples_ratio"])

        train_val_set = dataset_for_mean_std[:n_train_samples]
        train_set = train_val_set[int(n_train_samples * config["val_samples_ratio"]):]

        train_loader_mean_std = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size_train"], shuffle=False)
        input_mean, input_std, output_mean, output_std = get_mean_std(train_loader_mean_std)

    # =======================
    # data transforms
    # =======================
    input_transformations = []
    output_transformations = []
    data_input_transform, data_output_transform = None, None
    # Standardize input
    if config["log_input"]:
        input_transformations.append(transforms.Lambda(log_data))
    if config["normalize_input"]:
        input_transformations.append(transforms.Normalize(mean=input_mean, std=input_std))

    if len(input_transformations) > 0:
        data_input_transform = transforms.Compose(input_transformations)

    # Standardize output
    if config["log_output"]:
        output_transformations.append(transforms.Lambda(log_data))
    if config["normalize_output"]:
        output_transformations.append(transforms.Normalize(mean=output_mean, std=output_std))

    if len(output_transformations) > 0:
        data_output_transform = transforms.Compose(output_transformations)

    # ============================
    # Datasets and data loaders
    # ============================
    dataset = DFMDataset(data_dir=data_dir, input_transform=data_input_transform,
                         output_transform=data_output_transform)

    n_train_samples = int(len(dataset) * config["train_samples_ratio"])

    train_val_set = dataset[:n_train_samples]
    validation_set = train_val_set[:int(n_train_samples * config["val_samples_ratio"])]
    train_set = train_val_set[int(n_train_samples * config["val_samples_ratio"]):]
    test_set = dataset[n_train_samples:]
    print("len(trainset): {}, len(valset): {}, len(testset): {}".format(len(train_set), len(validation_set),
                                                                        len(test_set)))

    study.set_user_attr("n_train_samples", len(train_set))
    study.set_user_attr("n_val_samples", len(validation_set))
    study.set_user_attr("n_test_samples", len(test_set))

    study.set_user_attr("normalize_input", config["normalize_input"])
    study.set_user_attr("normalize_output", config["normalize_output"])

    study.set_user_attr("input_log", config["log_input"])
    study.set_user_attr("input_mean", input_mean)
    study.set_user_attr("input_std", input_std)

    study.set_user_attr("output_log", config["log_output"])
    study.set_user_attr("output_mean", output_mean)
    study.set_user_attr("output_std", output_std)

    if serialize_path is not None:
        joblib.dump(dataset, os.path.join(serialize_path, "dataset.pkl"))

    return train_set, validation_set, test_set


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='Data directory')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument("-c", "--cuda", default=False, action='store_true', help="use cuda")
    args = parser.parse_args(sys.argv[1:])

    data_dir = args.data_dir
    output_dir = args.output_dir
    use_cuda = args.cuda
    config = {"num_epochs": 1,
              "batch_size_train": 25,
              "batch_size_test": 250,
              "train_samples_ratio": 0.8,
              "val_samples_ratio": 0.2,
              "print_batches": 10,
              "log_input": False,
              "normalize_input": True,
              "log_output": False,
              "normalize_output": False}

    # Ooptuna params
    num_trials = 2#100

    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    print("device ", device)

    # Make runs repeatable
    random_seed = 12345
    torch.backends.cudnn.enabled = False  # Disable cuDNN use of nondeterministic algorithms
    torch.manual_seed(random_seed)
    output_dir = os.path.join(output_dir, "seed_{}".format(random_seed))
    if os.path.exists(output_dir):
        raise IsADirectoryError("Results output dir {} already exists".format(output_dir))
    os.mkdir(output_dir)

    study = optuna.create_study(sampler=TPESampler(seed=random_seed), direction="maximize")

    # ================================
    # Datasets and data loaders
    # ================================
    train_set, validation_set, test_set = prepare_dataset(study, config, data_dir=data_dir, serialize_path=output_dir)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size_train"], shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=config["batch_size_test"], shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config["batch_size_test"], shuffle=False)


    def obj_func(trial):
        return objective(trial, train_loader, validation_loader)



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
    df.to_csv('optuna_results.csv', index=False)  # Save to csv file

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
