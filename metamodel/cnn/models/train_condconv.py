import os
import sys
import shutil
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
from metamodel.cnn.models.cond_conv2d import CondConv2d
from metamodel.cnn.models.cond_net import CondNet
from metamodel.cnn.models.auxiliary_functions import plot_samples

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

    #plot_samples(train_loader, n_samples=25)



    optimizer_name = "Adam"  # trial.suggest_categorical("optimizer", ["Adam"])
    hidden_activation = F.relu

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
                    "input_size": trials_config["input_size"]
                    }

    # # print("config ", trials_config)
    # if "results_dir" in trials_config:
    #     #@TODO: refactor ASAP
    #
    #     executed_study = joblib.load(os.path.join(trials_config["results_dir"], "study.pkl"))
    #     model_path = os.path.join(trials_config["results_dir"], "best_trial")
    #
    #     executed_model_kwargs = executed_study.best_trial.user_attrs["model_kwargs"]
    #     model = executed_study.best_trial.user_attrs["model_class"](**executed_model_kwargs)
    #
    #     # Initialize optimizer
    #     optimizer = executed_study.best_trial.user_attrs["optimizer_class"](model.parameters(),
    #                                                                         **executed_study.best_trial.user_attrs["optimizer_kwargs"])
    #
    #
    #     checkpoint = torch.load(model_path)
    #     train_loss = checkpoint['train_loss']
    #     valid_loss = checkpoint['valid_loss']
    #
    #     # layer_kwargs = {'n_conv_layers': 1, 'max_channel': 72, 'pool': 'None', 'pool_size': 0, 'kernel_size': 3,
    #     #                 'stride': 1, 'pool_stride': 0, 'use_batch_norm': True, 'n_hidden_layers': 1,
    #     #                 'max_hidden_neurons': 48, 'input_size': 3}
    #     # model = CondNet(**layer_kwargs)
    #
    #     #print("model ", model)
    #
    #     # Initialize optimizer
    #     #optimizer = study.best_trial.user_attrs["optimizer_class"](model.parameters(),
    #     #                                                           **study.best_trial.user_attrs["optimizer_kwargs"])
    #
    #     #checkpoint = torch.load(trials_config["conv_layer_obj"])
    #     #train_loss = checkpoint['train_loss']
    #     #valid_loss = checkpoint['valid_loss']
    #
    #     model.load_state_dict(checkpoint['best_model_state_dict'])
    #     #model.out_channels = 3
    #     model.eval()
    #
    #     model._convs.stride = model_kwargs["stride"]
    #
    #     #print("model.parameters() ", model.parameters)
    #     # for param in model.parameters():
    #     #     print(param.data)
    #
    #     #print("model._convs.weights ", model._convs.weight)
    #     #print("model._fcls.weights ", model._fcls.weight)
    #
    #     # for hidden_layer in model._fcls._hidden_layers:
    #     #     print("hidden layer weight ", hidden_layer.weight)
    #     model_kwargs["convs"] = model._convs
    #     model_kwargs["fcls"] = model._fcls

    n_pretrained_layers = 0
    if "trained_layers_dir" in trials_config:
        trained_layers_dir = trials_config["trained_layers_dir"]

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

            for bnorm in model._batch_norms:
                bnorm.requires_grad_(False)
                for param in bnorm.parameters():
                    param.requires_grad = False

            for fcl in model._fcls:
                fcl.requires_grad_(False)
                for param in fcl.parameters():
                    param.requires_grad = False

            model_kwargs["convs"] = model._convs
            model_kwargs["batch_norms"] = model._batch_norms
            #model_kwargs["batch_norms"] = model._batch_norms
            model_kwargs["fcls"] = model._fcls
            #print("model_kwargs ", model_kwargs)

            n_pretrained_layers = len(model._convs)

            #layer_models.append(model)
        #model_kwargs["layer_models"] = layer_models
        #exit()
        #model_kwargs["conv_layer_obj"] = [model, model] #trials_config["conv_layer_obj"]

    print("model_kwargs ", model_kwargs)

    # print("trial trial")
    # return np.random.uniform(0, 1)
    #model = CondConv2d(**model_kwargs).to(device)
    #model = CondConv2d(use_cuda=use_cuda).to(device)

    model = CondNet(**model_kwargs).to(device)

    # for name, para in model.named_parameters():
    #     print("-" * 20)
    #     print(f"name: {name}")
    #     print("values: ")
    #     print(para)

    for index, (cvs, lin) in enumerate(zip(model._convs, model._fcls)):
        if index > n_pretrained_layers - 1:
            break
        for param in cvs.parameters():
            param.requires_grad = False
        for param in lin.parameters():
            param.requires_grad = False

    # Initialize optimizer
    optimizer_kwargs = {"lr": lr}
    non_frozen_parameters = [p for p in model.parameters() if p.requires_grad]
    # print("non frozen parameters ", non_frozen_parameters)
    #
    # for name, para in model.named_parameters():
    #     if para.requires_grad:
    #         print("-" * 20)
    #         print(f"name: {name}")
    #         print("values: ")
    #         print(para)
    #
    # exit()
    optimizer = getattr(optim, optimizer_name)(params=non_frozen_parameters, **optimizer_kwargs)

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

    scheduler = None
    avg_loss = best_vloss
    train = trials_config["train"] if "train" in trials_config else True
    if "scheduler" in config:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    for epoch in range(config["num_epochs"]):
        #try:
        if train:
            model.train(True)
            avg_loss = train_one_epoch(model, optimizer, train_loader, config, loss_fn=loss_fn_name(), use_cuda=use_cuda)  # Train the model
            if scheduler is not None:
                scheduler.step()
        else:
            if trial.number > 0:
                return avg_vloss
        model.train(False)
        avg_vloss = validate(model, validation_loader, loss_fn=loss_fn_name(), use_cuda=use_cuda)   # Evaluate the model

        avg_loss_list.append(avg_loss)
        avg_vloss_list.append(avg_vloss)

        #print("epoch: {}, loss train: {}, val: {}".format(epoch, avg_loss, avg_vloss))

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

    for key, value in trial.params.items():
        model_path += "_{}_{}".format(key, value)
    model_path = os.path.join(output_dir, model_path)

    #print("model_path save ", model_path)

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
              "log_input": True,
              "normalize_input": True,
              "log_output": False,
              "normalize_output": True}

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
        shutil.rmtree(output_dir)
        #raise IsADirectoryError("Results output dir {} already exists".format(output_dir))
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
