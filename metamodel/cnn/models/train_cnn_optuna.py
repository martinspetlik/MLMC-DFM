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
from optuna.samplers import TPESampler, BruteForceSampler
import time
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from metamodel.cnn.models.trials.net_optuna import Net
from metamodel.cnn.datasets.dfm_dataset import DFMDataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from metamodel.cnn.models.auxiliary_functions import get_mean_std, log_data, check_shapes
from metamodel.cnn.models.train_pure_cnn_optuna import train_one_epoch, prepare_dataset, validate, load_trials_config
from metamodel.cnn.visualization.visualize_data import plot_samples


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

    #if "batch_size_train" in trials_config and isinstance(config["batch_size_train"], (list, np.ndarray)):
    batch_size_train = trial.suggest_categorical("batch_size_train", trials_config["batch_size_train"])
    config["batch_size_train"] = batch_size_train

    if "pool_indices" in trials_config:
        pool_indices = trial.suggest_categorical("pool_indices", trials_config["pool_indices"])

    if "use_cnn_dropout" in trials_config:
        use_cnn_dropout = trial.suggest_categorical("use_cnn_dropout", trials_config["use_cnn_dropout"])

    if "use_fc_dropout" in trials_config:
        use_fc_dropout = trial.suggest_categorical("use_fc_dropout", trials_config["use_fc_dropout"])

    if "cnn_dropout_indices" in trials_config:
        cnn_dropout_indices = trial.suggest_categorical("cnn_dropout_indices", trials_config["cnn_dropout_indices"])

    if "fc_dropout_indices" in trials_config:
        fc_dropout_indices = trial.suggest_categorical("fc_dropout_indices", trials_config["fc_dropout_indices"])
        #config["fc_dropout_indices"] = trials_config["fc_dropout_indices"]

    if "cnn_dropout_ratios" in trials_config:
        cnn_dropout_ratios = trial.suggest_categorical("cnn_dropout_ratios", trials_config["cnn_dropout_ratios"])
        #config["cnn_dropout_ratios"] = trials_config["cnn_dropout_ratios"]

    if "fc_dropout_ratios" in trials_config:
        fc_dropout_ratios = trial.suggest_categorical("fc_dropout_ratios", trials_config["fc_dropout_ratios"])
        #config["fc_dropout_ratios"] = trials_config["fc_dropout_ratios"]

    if "n_train_samples" in trials_config and trials_config["n_train_samples"] is not None:
        n_train_samples = trial.suggest_categorical("n_train_samples", trials_config["n_train_samples"])
        config["n_train_samples"] = n_train_samples

        if "n_test_samples" in trials_config and trials_config["n_test_samples"] is not None:
            config["n_test_samples"] = trials_config["n_test_samples"]

        train_set, validation_set, test_set = prepare_dataset(study, config, data_dir=data_dir,
                                                              serialize_path=output_dir)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size_train"], shuffle=True)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=config["batch_size_test"],
                                                        shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=config["batch_size_test"], shuffle=False)

    # plot_samples(train_loader, n_samples=25)
    # exit()

    optimizer_name = "Adam"
    if "optimizer_name" in trials_config:
        optimizer_name = trial.suggest_categorical("optimizer_name", trials_config["optimizer_name"])

    L2_penalty = 0
    if "L2_penalty" in trials_config:
        L2_penalty = trial.suggest_categorical("L2_penalty", trials_config["L2_penalty"])

    hidden_activation_name = "relu"
    if "hidden_activation_name" in trials_config:
        hidden_activation_name = trial.suggest_categorical("hidden_activation_name",
                                                           trials_config["hidden_activation_name"])

    hidden_activation = getattr(F, hidden_activation_name)

    # print("hidden activation ", hidden_activation)
    # exit()

    #hidden_activation = F.relu

    #print("kernel_size: {}, stride: {}, pool_size: {}, pool_stride: {}".format(kernel_size, stride, pool_size, pool_stride))

    flag, input_size = check_shapes(n_conv_layers, kernel_size, stride, pool_size, pool_stride,
                                    input_size=trials_config["input_size"])

    # print("max channels ", max_channel)
    # print("flag: {} flatten x: {}".format(flag, input_size * input_size * max_channel))

    if flag == -1:
        print("flag: {}".format(flag))
        return best_vloss

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
                    "hidden_activation": hidden_activation,
                    "input_size": trials_config["input_size"],
                    "output_bias": trials_config["output_bias"] if "output_bias" in trials_config else False,
                    "pool_indices": pool_indices if "pool_indices" in trials_config else [],
                    "use_cnn_dropout": use_cnn_dropout if "use_cnn_dropout" in trials_config else False,
                    "use_fc_dropout": use_fc_dropout if "use_fc_dropout" in trials_config else False,
                    "cnn_dropout_indices": cnn_dropout_indices if "cnn_dropout_indices" in trials_config else [],
                    "fc_dropout_indices": fc_dropout_indices if "fc_dropout_indices" in trials_config else [],
                    "cnn_dropout_ratios": cnn_dropout_indices if "cnn_dropout_ratios" in trials_config else [],
                    "fc_dropout_ratios": fc_dropout_indices if "fc_dropout_ratios" in trials_config else []
                    }

    # print("config ", trials_config)
    # if "conv_layer_obj" in trials_config:
    #     #@TODO: refactor ASAP
    #     layer_kwargs = {'n_conv_layers': 1, 'max_channel': 72, 'pool': 'None', 'pool_size': 0, 'kernel_size': 3,
    #                     'stride': 1, 'pool_stride': 0, 'use_batch_norm': True, 'n_hidden_layers': 1,
    #                     'max_hidden_neurons': 48, 'input_size': 3}
    #     model = Net(**layer_kwargs)
    #
    #     #print("model ", model)
    #
    #     # Initialize optimizer
    #     #optimizer = study.best_trial.user_attrs["optimizer_class"](model.parameters(),
    #     #                                                           **study.best_trial.user_attrs["optimizer_kwargs"])
    #
    #     checkpoint = torch.load(trials_config["conv_layer_obj"])
    #     #train_loss = checkpoint['train_loss']
    #     #valid_loss = checkpoint['valid_loss']
    #
    #     model.load_state_dict(checkpoint['best_model_state_dict'])
    #     model.out_channels = 3
    #     model.eval()
    #
    #     model_kwargs["conv_layer_obj"] = [model, model] #trials_config["conv_layer_obj"]
    #print("model_kwargs ", model_kwargs)
    # print("trial trial")
    # return np.random.uniform(0, 1)

    if "input_channels" in trials_config:
        model_kwargs["input_channel"] = len(trials_config["input_channels"])
    if "output_channels" in trials_config:
        model_kwargs["n_output_neurons"] = len(trials_config["output_channels"])

    model = Net(trial, **model_kwargs).to(device)

    # print("model._convs ", model._convs)
    # print("moodel._hidden_layers ", model._hidden_layers)
    # print("moodel._output_layer ", model._output_layer)

    # Initialize optimizer
    optimizer_kwargs = {"lr": lr, "weight_decay": L2_penalty}
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
    # print("model path ", model_path)
    # if os.path.exists(model_path):
    #     print("model pat hexists")
    #     return avg_vloss

    scheduler = None
    train = trials_config["train"] if "train" in trials_config else True

    if "scheduler" in trials_config:
        print("scheduler in config ")
        if "class" in trials_config["scheduler"]:
            if trials_config["scheduler"]["class"] == "ReduceLROnPlateau":
                scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                           patience=trials_config["scheduler"]["patience"],
                                                           factor=trials_config["scheduler"]["factor"])
            else:
                scheduler = lr_scheduler.StepLR(optimizer, step_size=trials_config["scheduler"]["step_size"],
                                            gamma=trials_config["scheduler"]["gamma"])

    for epoch in range(config["num_epochs"]):
        #try:
        if train:
            model.train(True)
            avg_loss = train_one_epoch(model, optimizer, train_loader, config, loss_fn=loss_fn_name(), use_cuda=use_cuda)  # Train the model

        model.train(False)
        avg_vloss = validate(model, validation_loader, loss_fn=loss_fn_name(), use_cuda=use_cuda)   # Evaluate the model

        if scheduler is not None:
            scheduler.step(avg_vloss)
            print("scheduler lr: {}".format(scheduler._last_lr))

        avg_loss_list.append(avg_loss)
        avg_vloss_list.append(avg_vloss)

        print("epoch: {}, loss train: {}, val: {}".format(epoch, avg_loss, avg_vloss))

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            best_epoch = epoch

            model_state_dict = model.state_dict()
            optimizer_state_dict = optimizer.state_dict()

        # For pruning (stops trial early if not promising)
        trial.report(avg_vloss, epoch)
        # # Handle pruning based on the intermediate value.
        # if trial.should_prune():
        #     raise optuna.exceptions.TrialPruned()
        # except Exception as e:
        #     print(str(e))
        #     return avg_vloss

    #for key, value in trial.params.items():
    #    model_path += "_{}_{}".format(key, value)

    model_path = os.path.join(output_dir, model_path)

    torch.save({
        'best_epoch': best_epoch,
        'best_model_state_dict': model_state_dict,
        'best_optimizer_state_dict': optimizer_state_dict,
        'train_loss': avg_loss_list,
        'valid_loss': avg_vloss_list,
        'training_time': time.time() - start_time,
    }, model_path)

    return best_vloss


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

    config = {"num_epochs": trials_config["num_epochs"],
              "batch_size_train": trials_config["batch_size_train"],
              "batch_size_test": 250,
              "n_train_samples": trials_config["n_train_samples"] if "n_train_samples" in trials_config else None,
              "n_test_samples": trials_config["n_test_samples"] if "n_test_samples" in trials_config else None,
              "train_samples_ratio": trials_config["train_samples_ratio"] if "train_samples_ratio" in trials_config else 0.8,
              "val_samples_ratio": trials_config["val_samples_ratio"] if "val_samples_ratio" in trials_config else 0.2,
              "print_batches": 10,
              "log_input": trials_config["log_input"] if "log_input" in trials_config else True,
              "normalize_input": trials_config["normalize_input"] if "normalize_input" in trials_config else True,
              "log_output": trials_config["log_output"] if "log_output" in trials_config else False,
              "normalize_output": trials_config["normalize_output"] if "normalize_output" in trials_config else True,
              "input_channels": trials_config["input_channels"] if "input_channels" in trials_config else None,
              "output_channels": trials_config["output_channels"] if "output_channels" in trials_config else None
              }

    # Optuna params
    num_trials = trials_config["num_trials"]

    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    print("device ", device)

    # Make runs repeatable
    random_seed = trials_config["random_seed"]
    torch.backends.cudnn.enabled = False  # Disable cuDNN use of nondeterministic algorithms
    torch.manual_seed(random_seed)
    output_dir = os.path.join(output_dir, "seed_{}".format(random_seed))
    if os.path.exists(output_dir) and not args.append:
        raise IsADirectoryError("Results output dir {} already exists".format(output_dir))
    if not args.append:
        os.mkdir(output_dir)
    elif not os.path.exists(output_dir):
        raise NotADirectoryError("output dir {} not exists".format(output_dir))

    sampler = TPESampler(seed=random_seed)
    if "sampler_class" in trials_config:
        if trials_config["sampler_class"] == "BruteForceSampler":
            sampler = BruteForceSampler(seed=random_seed)

    study = optuna.create_study(sampler=sampler, direction="minimize")

    # ================================
    # Datasets and data loaders
    # ================================
    train_loader, validation_loader = None, None
    if "n_train_samples" not in config or not isinstance(config["n_train_samples"], (list, np.ndarray)):
        train_set, validation_set, test_set = prepare_dataset(study, config, data_dir=data_dir, serialize_path=output_dir)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size_train"], shuffle=True)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=config["batch_size_train"], shuffle=False)
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
