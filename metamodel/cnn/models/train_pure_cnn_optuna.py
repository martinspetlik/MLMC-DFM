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
import yaml
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from metamodel.cnn.models.trials.net_optuna_2 import Net
from metamodel.cnn.datasets.dfm_dataset import DFMDataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from metamodel.cnn.models.auxiliary_functions import get_mean_std, log_data
#from metamodel.cnn.visualization.visualize_data import plot_samples


def get_trained_layers(trials_config, model_kwargs):
    trained_layers_dir = trials_config["trained_layers_dir"]

    all_convs = nn.ModuleList()
    all_fcls = nn.ModuleList()
    all_batch_norms = nn.ModuleList()
    for layer_dir in trained_layers_dir:
        executed_study = joblib.load(os.path.join(layer_dir, "study.pkl"))
        model_path = os.path.join(layer_dir, "best_trial")

        executed_model_kwargs = executed_study.best_trial.user_attrs["model_kwargs"]
        model = executed_study.best_trial.user_attrs["model_class"](**executed_model_kwargs)

        print("executed model kwargs ", executed_model_kwargs)

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['best_model_state_dict'])
        # model.out_channels = 3
        model.eval()

        model._convs.stride = model_kwargs["stride"]

        # print("model.parameters() ", model.parameters)
        # for param in model.parameters():
        #     print(param.data)

        # for param in model.parameters():
        #     param.requires_grad = False

        for cnv in model._convs:
            cnv.requires_grad_(False)
            for param in cnv.parameters():
                param.requires_grad = False

            all_convs.append(cnv)

        for bnorm in model._batch_norms:
            bnorm.requires_grad_(False)
            for param in bnorm.parameters():
                param.requires_grad = False

            all_batch_norms.append(bnorm)

        for fcl in model._fcls:
            fcl.requires_grad_(False)
            for param in fcl.parameters():
                param.requires_grad = False

            all_fcls.append(fcl)

            # all_convs.append(model._convs)
            # all_batch_norms.append(model._batch_norms)
            # all_fcls.append(model._fcls)

    model_kwargs["convs"] = all_convs
    model_kwargs["batch_norms"] = all_batch_norms
    model_kwargs["fcls"] = all_fcls

    n_pretrained_layers = len(all_convs)
    return model_kwargs, n_pretrained_layers


def load_trials_config(path_to_config):
    with open(path_to_config, "r") as f:
        trials_config = yaml.load(f, Loader=yaml.FullLoader)
    return trials_config


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

    train_loss = running_loss / (i + 1)
    return train_loss


def validate(model, validation_loader, loss_fn=nn.MSELoss(), use_cuda=False):
    """
    Validate model
    :param model:
    :param loss_fn:
    :return:
    """
    running_vloss = 0.0
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vtargets = vdata

            if torch.cuda.is_available() and use_cuda:
                vinputs = vinputs.cuda()
                vtargets = vtargets.cuda()

            vinputs = vinputs.float()
            vtargets = vtargets.float()

            voutputs = torch.squeeze(model(vinputs))
            #print("voutputs.shape ", voutputs.shape)
            #print("vtargets.shape ", vtargets.shape)
            vloss = loss_fn(voutputs, vtargets)
            running_vloss += vloss.item()

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

    n_train_samples = None
    if "n_train_samples" in config and config["n_train_samples"] is not None:
        n_train_samples = config["n_train_samples"]
    if config["log_input"]:
        input_transform = transforms.Compose([transforms.Lambda(log_data)])
    if config["log_output"]:
        output_transform = transforms.Compose([transforms.Lambda(log_data)])

    if config["normalize_input"]:
        dataset_for_mean_std = DFMDataset(data_dir=data_dir,
                                          input_transform=input_transform,
                                          output_transform=output_transform,
                                          two_dim=True,
                                          input_channels=config["input_channels"] if "input_channels" in config else None,
                                          output_channels=config["output_channels"] if "output_channels" in config else None
                                          )
        dataset_for_mean_std.shuffle(seed=config["seed"])

        if n_train_samples is None:
            n_train_samples = int(len(dataset_for_mean_std) * config["train_samples_ratio"])

        train_val_set = dataset_for_mean_std[:n_train_samples]
        train_set = train_val_set[:-int(n_train_samples * config["val_samples_ratio"])]

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
                         output_transform=data_output_transform,
                         input_channels=config["input_channels"] if "input_channels" in config else None,
                         output_channels=config["output_channels"] if "output_channels" in config else None
                         )
    dataset.shuffle(config["seed"])

    if n_train_samples is None:
        n_train_samples = int(len(dataset) * config["train_samples_ratio"])

    train_val_set = dataset[:n_train_samples]
    train_set = train_val_set[:-int(n_train_samples * config["val_samples_ratio"])]
    validation_set = train_val_set[-int(n_train_samples * config["val_samples_ratio"]):]

    if "n_test_samples" in config and config["n_test_samples"] is not None:
        n_test_samples = config["n_test_samples"]
        test_set = dataset[-n_test_samples:]
    else:
        test_set = dataset[n_train_samples:]


    print("len(trainset): {}, len(valset): {}, len(testset): {}".format(len(train_set), len(validation_set),
                                                                        len(test_set)))


    # Save data to numpy array
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
    # from npy_append_array import NpyAppendArray
    # filename_input = 'input.npy'
    # filename_output = 'output.npy'
    # for input, output in train_loader:
    #     input = input.numpy()
    #     output = output.numpy()
    #
    #     with NpyAppendArray(filename_input) as npaa_in:
    #         npaa_in.append(input)
    #
    #     with NpyAppendArray(filename_output) as npaa_out:
    #         npaa_out.append(output)

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


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('data_dir', help='Data directory')
#     parser.add_argument('output_dir', help='Output directory')
#     parser.add_argument("-c", "--cuda", default=False, action='store_true', help="use cuda")
#     args = parser.parse_args(sys.argv[1:])
#
#     data_dir = args.data_dir
#     output_dir = args.output_dir
#     use_cuda = args.cuda
#     config = {"num_epochs": 10,
#               "batch_size_train": 25,
#               "batch_size_test": 250,
#               "train_samples_ratio": 0.8,
#               "val_samples_ratio": 0.2,
#               "print_batches": 10,
#               "log_input": True,
#               "normalize_input": True,
#               "log_output": False,
#               "normalize_output": True}
#
#     # Ooptuna params
#     num_trials = 2#100
#
#     device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
#     print("device ", device)
#
#     # Make runs repeatable
#     random_seed = 12345
#     torch.backends.cudnn.enabled = False  # Disable cuDNN use of nondeterministic algorithms
#     torch.manual_seed(random_seed)
#     output_dir = os.path.join(output_dir, "seed_{}".format(random_seed))
#     if os.path.exists(output_dir):
#         raise IsADirectoryError("Results output dir {} already exists".format(output_dir))
#     os.mkdir(output_dir)
#
#     study = optuna.create_study(sampler=TPESampler(seed=random_seed), direction="minimize")
#
#     # ================================
#     # Datasets and data loaders
#     # ================================
#     train_set, validation_set, test_set = prepare_dataset(study, config, data_dir=data_dir, serialize_path=output_dir)
#     train_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size_train"], shuffle=True)
#
#     validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=config["batch_size_test"], shuffle=False)
#     test_loader = torch.utils.data.DataLoader(test_set, batch_size=config["batch_size_test"], shuffle=False)
#
#
#     def obj_func(trial):
#         return objective(trial, train_loader, validation_loader)
#
#     study.optimize(obj_func, n_trials=num_trials)
#
#     # ================================
#     # Results
#     # ================================
#     # Find number of pruned and completed trials
#     pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
#     complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
#
#     # Display the study statistics
#     print("\nStudy statistics: ")
#     print("  Number of finished trials: ", len(study.trials))
#     print("  Number of pruned trials: ", len(pruned_trials))
#     print("  Number of complete trials: ", len(complete_trials))
#
#     trial = study.best_trial
#     print("Best trial:")
#     print("  Value: ", trial.value)
#     print("  Params: ")
#     for key, value in trial.params.items():
#         print("    {}: {}".format(key, value))
#
#     # Save results to csv file
#     df = study.trials_dataframe().drop(['datetime_start', 'datetime_complete', 'duration'], axis=1)  # Exclude columns
#     df = df.loc[df['state'] == 'COMPLETE']        # Keep only results that did not prune
#     df = df.drop('state', axis=1)                 # Exclude state column
#     df = df.sort_values('value')                  # Sort based on accuracy
#     df.to_csv('optuna_results.csv', index=False)  # Save to csv file
#
#     # Display results in a dataframe
#     print("\nOverall Results (ordered by accuracy):\n {}".format(df))
#
#     # Find the most important hyperparameters
#     most_important_parameters = optuna.importance.get_param_importances(study, target=None)
#
#     # Display the most important hyperparameters
#     print('\nMost important hyperparameters:')
#     for key, value in most_important_parameters.items():
#         print('  {}:{}{:.2f}%'.format(key, (15-len(key))*' ', value*100))
#
#     # serialize optuna study object
#     joblib.dump(study, os.path.join(output_dir, "study.pkl"))
