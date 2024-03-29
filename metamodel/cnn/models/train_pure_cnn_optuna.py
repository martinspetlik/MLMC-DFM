import os
import sys
import argparse
import logging
import joblib
import copy
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
#from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from metamodel.cnn.models.auxiliary_functions import get_mean_std, log_data, exp_data,\
    quantile_transform_fit, QuantileTRF, NormalizeData, log_all_data, init_norm, log10_data, log10_all_data
#from metamodel.cnn.visualization.visualize_data import plot_samples


def get_trained_layers(trials_config, model_kwargs):
    trained_layers_dir = trials_config["trained_layers_dir"]

    all_convs = nn.ModuleList()
    all_fcls = nn.ModuleList()
    all_batch_norms = nn.ModuleList()
    for layer_dir in trained_layers_dir:
        if not os.path.exists(os.path.join(layer_dir, "study.pkl")):
            from metamodel.cnn.models.cond_net import CondNet
            #model_path = "/home/martin/Documents/MLMC-DFM/optuna_runs/karolina/cond_conv/exp_3/seed_12345/trial_1_losses_model_cnn_net"
            model_kwargs = {'n_conv_layers': 1, 'max_channel': 156, 'pool': 'None', 'pool_size': 0, 'kernel_size': 3,
                            'stride': 1, 'pool_stride': 0, 'use_batch_norm': True, 'n_hidden_layers': 1,
                            'max_hidden_neurons': 194, 'input_size': 3, 'output_bias': True, 'pool_indices': [],
                            'use_cnn_dropout': False, 'use_fc_dropout': False, 'cnn_dropout_indices': [], 'fc_dropout_indices': [],
                            'cnn_dropout_ratios': [], 'fc_dropout_ratios': []}
            model = CondNet(**model_kwargs)
        else:
            executed_study = joblib.load(os.path.join(layer_dir, "study.pkl"))
            model_path = os.path.join(layer_dir, "best_trial")

            executed_model_kwargs = executed_study.best_trial.user_attrs["model_kwargs"]
            model = executed_study.best_trial.user_attrs["model_class"](**executed_model_kwargs)

        #print("executed model kwargs ", executed_model_kwargs)
        checkpoint = torch.load(model_path)
        #print("checkpoint['best_model_state_dict'] ", checkpoint['best_model_state_dict'])

        # checkpoint['best_model_state_dict']["_fcls.0._hidden_layers.0.weight"]  = checkpoint['best_model_state_dict']["_hidden_layers.0.weight"]
        # checkpoint['best_model_state_dict']["_fcls.0._hidden_layers.0.bias"] = checkpoint['best_model_state_dict']["_hidden_layers.0.bias"]
        # checkpoint['best_model_state_dict']["_fcls.0._output_layer.weight"] = checkpoint['best_model_state_dict']["_output_layer.weight"]
        # checkpoint['best_model_state_dict']["_fcls.0._output_layer.bias"] = checkpoint['best_model_state_dict']["_output_layer.bias"]
        # del checkpoint['best_model_state_dict']["_hidden_layers.0.weight"]
        # del checkpoint['best_model_state_dict']["_hidden_layers.0.bias"]
        # del checkpoint['best_model_state_dict']["_output_layer.weight"]
        # del checkpoint['best_model_state_dict']["_output_layer.bias"]


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

def save_output_dataset(model, data_loader, study, output_data_dir, sample_id=0, use_cuda=False):

    inverse_transform = None
    if "normalize_output" in study.user_attrs and study.user_attrs["normalize_output"]:
        std = 1/study.user_attrs["output_std"]
        zeros_mean = np.zeros(len(study.user_attrs["output_mean"]))

        ones_std = np.ones(len(zeros_mean))
        mean = -study.user_attrs["output_mean"]

        transforms_list = [transforms.Normalize(mean=zeros_mean, std=std),
                            transforms.Normalize(mean=mean, std=ones_std)]

        if "output_log" in study.user_attrs and study.user_attrs["output_log"]:
            transforms_list.append(transforms.Lambda(exp_data))

        inverse_transform = transforms.Compose(transforms_list)

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, targets = data

            if torch.cuda.is_available() and use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            inputs = inputs.float()
            #vtargets = vtargets.float()
            outputs = torch.squeeze(model(inputs, output_data_dir))

            if os.path.exists(output_data_dir):
                for sample in outputs:
                    #print("sample ", sample)
                    for i in range(3):
                        for j in range(3):
                            d = sample[:][i][j]
                            #print("inverse_transform(torch.reshape(d, (*d.shape, 1, 1))) ", inverse_transform(torch.reshape(d, (*d.shape, 1, 1))))
                            sample[:][i][j] = np.reshape(inverse_transform(torch.reshape(d, (*d.shape, 1, 1))), d.shape)

                    sample_dir = os.path.join(output_data_dir, "sample_{}".format(sample_id))
                    if not os.path.exists(sample_dir):
                        os.mkdir(sample_dir)
                    np.save(os.path.join(sample_dir, "input_tensor"), sample)

                    sample_id += 1
            else:
                os.mkdir(output_data_dir)
    return sample_id


def validate(model, validation_loader, loss_fn=nn.MSELoss(), acc_fn=nn.MSELoss(), use_cuda=False):
    """
    Validate model
    :param model:
    :param loss_fn:
    :return:
    """
    running_vloss = 0.0
    running_vacc = 0
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
            # print("vtargets.shape ", vtargets.shape)
            vloss = loss_fn(voutputs, vtargets)
            running_vloss += vloss.item()

            #print("validate running loss ", running_vloss)

            vacc = acc_fn(voutputs, vtargets)
            running_vacc += vacc.item()

        avg_vloss = running_vloss / (i + 1)
        avg_vacc = running_vacc / (i + 1)

    return avg_vloss, avg_vacc


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


def features_transform(config, data_dir, output_file_name, input_transform_list, output_transform_list, dataset_for_transform=None):
    #################################
    ## Data for Quantile Transform ##
    #################################
    quantile_trf_obj = QuantileTRF()
    if dataset_for_transform is None:
        dataset_for_transform = DFMDataset(data_dir=data_dir, output_file_name=output_file_name, two_dim=True,
                                           fractures_sep=config["fractures_sep"] if "fractures_sep" in config else False,
                                           cross_section=config["cross_section"] if "cross_section" in config else False
                                           )
    input_data = np.array([])
    output_data = np.array([])

    n_data_input = 1000000
    n_data_output = 300000
    if "input_transform" in config or "output_transform" in config:
        for index, data in enumerate(dataset_for_transform):
            input, output = data

            input = np.reshape(input, (input.shape[0], input.shape[-1] * input.shape[-2]))
            if input.shape[-1] * input.shape[-2] * index < n_data_input:
                if len(input_data) == 0:
                    input_data = input
                else:
                    input_data = np.concatenate([input_data, input], axis=1)

            if output_data.shape[-1] < n_data_output:
                output = np.reshape(output, (output.shape[0], 1))
                if len(output_data) == 0:
                    output_data = output
                else:
                    output_data = np.concatenate([output_data, output], axis=1)

    if "input_transform" in config and len(config["input_transform"]) > 0:
        quantile_trfs = quantile_transform_fit(input_data,
                                                indices=config["input_transform"]["indices"],
                                                transform_type=config["input_transform"]["type"])
        joblib.dump(quantile_trfs, os.path.join(config["output_dir"], "input_transform.pkl"))
        quantile_trf_obj.quantile_trfs_in = quantile_trfs
        input_transform_list.append(quantile_trf_obj.quantile_transform_in)

    if "output_transform" in config and len(config["output_transform"]) > 0:
        quantile_trfs_out = quantile_transform_fit(output_data,
                                                   indices=config["output_transform"]["indices"],
                                                   transform_type=config["output_transform"]["type"])
        joblib.dump(quantile_trfs_out, os.path.join(config["output_dir"], "output_transform.pkl"))
        quantile_trf_obj.quantile_trfs_out = quantile_trfs_out
        output_transform_list.append(quantile_trf_obj.quantile_transform_out)
    return input_transform_list, output_transform_list


def _append_dataset(dataset_1, dataset_2):
    dataset_1._bulk_file_paths.extend(dataset_2._bulk_file_paths)
    dataset_1._fracture_file_paths.extend(dataset_2._fracture_file_paths)
    dataset_1._cross_section_file_paths.extend(dataset_2._cross_section_file_paths)
    dataset_1._output_file_paths.extend(dataset_2._output_file_paths)


def prepare_sub_datasets(study, config, data_dir, serialize_path=None):
    complete_train_set, complete_val_set, complete_test_set = None, None, None
    for key, dset_config in config["sub_datasets"].items():
        prepare_dset_config = copy.deepcopy(config)
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
        prepare_dset_config["n_train_samples"] = dset_config["n_train_samples"]
        prepare_dset_config["n_test_samples"] = dset_config["n_test_samples"]
        prepare_dset_config["val_samples_ratio"] = dset_config["val_samples_ratio"]
        print("prepare_dset_config ", prepare_dset_config)

        sub_train_set, sub_val_set, sub_test_set = prepare_dataset(study, prepare_dset_config, dset_config['dataset_path'])

        if complete_train_set is None:
            complete_train_set = sub_train_set
            complete_val_set = sub_val_set
            complete_test_set = sub_test_set
        else:
            _append_dataset(complete_train_set, sub_train_set)
            if len(sub_val_set) > 0:
                _append_dataset(complete_val_set, sub_val_set)
            _append_dataset(complete_test_set, sub_test_set)

    # print("complete train set len ", len(complete_train_set))
    #
    # dataset_loader = torch.utils.data.DataLoader(complete_train_set, shuffle=False)
    # k_xy = []
    #
    # for input, output in dataset_loader:
    #     output = np.squeeze(output.numpy())
    #     k_xy.append(output[1])
    #
    # np.savez_compressed(os.path.join("/home/martin/Documents/MLMC-DFM", "fr_div_0_0_1_10_k_xy"), data=np.array(k_xy))
    #
    # exit()

    data_init_transform, data_input_transform, data_output_transform = prepare_dataset(study, config, data_dir, train_dataset=complete_train_set)
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

    dataset = copy.deepcopy(complete_train_set)
    if len(complete_val_set) > 0:
        _append_dataset(dataset, complete_val_set)
    _append_dataset(dataset, complete_test_set)

    if serialize_path is not None:
        joblib.dump(dataset, os.path.join(serialize_path, "dataset.pkl"))

    if study is not None:
        study.set_user_attr("n_train_samples", len(complete_train_set))
        study.set_user_attr("n_val_samples", len(complete_val_set))
        study.set_user_attr("n_test_samples", len(complete_test_set))

    # dataset_loader = torch.utils.data.DataLoader(complete_train_set, shuffle=False)
    #
    # for input, output in dataset_loader:
    #     output = np.squeeze(output.numpy())
    #     print("output shape ", output.shape)
    #     print("type output ", output)
    #     print("output[1] ", output[1])
    #     # print("output squeeze ", np.squeeze(output))
    #     k_xy.append(output[1])
    #
    #     exit()

    return complete_train_set, complete_val_set, complete_test_set


def _split_dataset(dataset, config, n_train_samples):
    if n_train_samples is None:
        n_train_samples = int(len(dataset) * config["train_samples_ratio"])

    n_train_samples = np.min([n_train_samples, int(len(dataset) * config["train_samples_ratio"])])

    train_val_set = dataset[:n_train_samples]
    if config["val_samples_ratio"] == 0.0:
        train_set = train_val_set
        validation_set = []
    else:
        train_set = train_val_set[:-int(n_train_samples * config["val_samples_ratio"])]
        validation_set = train_val_set[-int(n_train_samples * config["val_samples_ratio"]):]

    if "n_test_samples" in config and config["n_test_samples"] is not None:
        n_test_samples = config["n_test_samples"]
        test_set = dataset[-n_test_samples:]
    else:
        test_set = dataset[n_train_samples:]

    return train_set, validation_set, test_set


def prepare_dataset(study, config, data_dir, serialize_path=None, train_dataset=None):
    output_file_name = "output_tensor.npy"

    # ===================================
    # Get mean and std for each channel
    # ===================================
    input_mean, output_mean, input_std, output_std = 0, 0, 1, 1
    output_quantiles = []
    data_normalizer = NormalizeData()

    n_train_samples = None
    if "n_train_samples" in config and config["n_train_samples"] is not None:
        n_train_samples = config["n_train_samples"]

    init_transform = []
    input_transform_list = []
    output_transform_list = []

    ###########################
    ## Initial normalization ##
    ###########################

    if config["init_norm"]:
        init_transform.append(transforms.Lambda(init_norm))

    ####################
    ## Log transforms ##
    ####################
    if config["log_input"]:
        if "log_all_input_channels" in config and config["log_all_input_channels"]:
            input_transform_list.append(transforms.Lambda(log_all_data))
        else:
            input_transform_list.append(transforms.Lambda(log_data))

    if config["log_output"]:
        output_transform_list.append(transforms.Lambda(log_data))
    elif "log10_output" in config and  config["log10_output"]:
        output_transform_list.append(transforms.Lambda(log10_data))
    elif "log_all_output" in config and config["log_all_output"]:
        output_transform_list.append(transforms.Lambda(log_all_data))
    elif "log10_all_output" in config and config["log10_all_output"]:
        output_transform_list.append(transforms.Lambda(log10_all_data))


    # ########################
    # ## Quantile Transform ##
    # ########################
    input_transform_list, output_transform_list = features_transform(config,
                                                                     data_dir,
                                                                     output_file_name,
                                                                     input_transform_list,
                                                                     output_transform_list)


    input_transform = transforms.Compose(input_transform_list)
    output_transform = transforms.Compose(output_transform_list)

    if len(init_transform) > 0:
        init_transform = transforms.Compose(init_transform)
    else:
        init_transform = None

    if config["normalize_input"] or config["normalize_output"]:
        if train_dataset is not None:
            dataset_for_mean_std = train_dataset
            dataset_for_mean_std.init_transform = init_transform
            dataset_for_mean_std.input_transform = input_transform
            dataset_for_mean_std.output_transform = output_transform
        else:
            dataset_for_mean_std = DFMDataset(data_dir=data_dir,
                                              output_file_name=output_file_name,
                                              input_transform=input_transform,
                                              output_transform=output_transform,
                                              init_transform=init_transform,
                                              init_norm_use_all_features=config["init_norm_use_all_features"] if "init_norm_use_all_features" in config else False,
                                              two_dim=True,
                                              input_channels=config["input_channels"] if "input_channels" in config else None,
                                              output_channels=config["output_channels"] if "output_channels" in config else None,
                                              fractures_sep=config["fractures_sep"] if "fractures_sep" in config else False,
                                              cross_section=config["cross_section"] if "cross_section" in config else False
                                              )
        dataset_for_mean_std.shuffle(seed=config["seed"])

        if n_train_samples is None:
            n_train_samples = int(len(dataset_for_mean_std) * config["train_samples_ratio"])

        n_train_samples = np.min([n_train_samples, int(len(dataset_for_mean_std) * config["train_samples_ratio"])])

        train_val_set = dataset_for_mean_std[:n_train_samples]
        if config["val_samples_ratio"] == 0:
            train_set = train_val_set
        else:
            train_set = train_val_set[:-int(n_train_samples * config["val_samples_ratio"])]

        train_loader_mean_std = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size_train"], shuffle=False)
        iqr = []
        if "output_iqr_scale" in config:
            iqr = config["output_iqr_scale"]
        input_mean, input_std, output_mean, output_std, output_quantiles = get_mean_std(train_loader_mean_std, output_iqr=iqr)
        print("input mean: {}, std:{}, output mean: {}, std: {}".format(input_mean, input_std, output_mean, output_std))
        print("output quantiles: {}".format(output_quantiles))

        # validation_set = train_val_set[-int(n_train_samples * config["val_samples_ratio"]):]
        #
        # if "n_test_samples" in config and config["n_test_samples"] is not None:
        #     n_test_samples = config["n_test_samples"]
        #     test_set = dataset_for_mean_std[-n_test_samples:]
        # else:
        #     test_set = dataset_for_mean_std[n_train_samples:]
        #
        # train_loader_mean_std = torch.utils.data.DataLoader(validation_set, batch_size=config["batch_size_train"], shuffle=False)
        # v_input_mean, v_input_std, v_output_mean, v_output_std, quantiles = get_mean_std(train_loader_mean_std, output_iqr=iqr)
        # print("VAL SET input mean: {}, std:{}, output mean: {}, std: {}".format(v_input_mean, v_input_std, v_output_mean, v_output_std))
        # print("VAL SET output quantiles: {}".format(quantiles))
        #
        # train_loader_mean_std = torch.utils.data.DataLoader(test_set, batch_size=config["batch_size_train"], shuffle=False)
        # t_input_mean, t_input_std, t_output_mean, t_output_std, t_quantiles = get_mean_std(train_loader_mean_std, output_iqr=iqr)
        # print("TEST SET input mean: {}, std:{}, output mean: {}, std: {}".format(t_input_mean, t_input_std, t_output_mean, t_output_std))
        # print("TEST SET output quantiles: {}".format(t_quantiles))

        #input_mean, input_std, output_mean, output_std = v_input_mean, v_input_std, v_output_mean, v_output_std

    # =======================
    # data transforms
    # =======================
    input_transformations = []
    output_transformations = []
    init_transform = []

    ###########################
    ## Initial normalization ##
    ###########################
    data_init_transform = None
    if config["init_norm"]:
        init_transform.append(transforms.Lambda(init_norm))

    if len(init_transform) > 0:
        data_init_transform = transforms.Compose(init_transform)

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
        data_normalizer.input_mean = input_mean
        data_normalizer.input_std = input_std
        input_transformations.append(data_normalizer.normalize_input)

    if len(input_transformations) > 0:
        data_input_transform = transforms.Compose(input_transformations)

    # Standardize output
    if config["log_output"]:
        output_transformations.append(transforms.Lambda(log_data))
    elif "log10_output" in config and  config["log10_output"]:
        output_transformations.append(transforms.Lambda(log10_data))
    elif "log_all_output" in config and config["log_all_output"]:
        output_transformations.append(transforms.Lambda(log_all_data))
    elif "log10_all_output" in config and config["log10_all_output"]:
        output_transformations.append(transforms.Lambda(log10_all_data))

    if config["normalize_output"]:
        if "normalize_output_indices" in config:
            data_normalizer.input_indices = config["normalize_output_indices"]
        data_normalizer.output_mean = output_mean
        data_normalizer.output_std = output_std
        data_normalizer.output_quantiles = output_quantiles
        output_transformations.append(data_normalizer.normalize_output)

    if len(output_transformations) > 0:
        data_output_transform = transforms.Compose(output_transformations)

    if train_dataset is None:
        # ============================
        # Datasets and data loaders
        # ============================
        dataset = DFMDataset(data_dir=data_dir,
                             output_file_name=output_file_name,
                             input_transform=data_input_transform,
                             output_transform=data_output_transform,
                             init_transform=data_init_transform,
                             init_norm_use_all_features=config[
                                 "init_norm_use_all_features"] if "init_norm_use_all_features" in config else False,
                             input_channels=config["input_channels"] if "input_channels" in config else None,
                             output_channels=config["output_channels"] if "output_channels" in config else None,
                             fractures_sep=config["fractures_sep"] if "fractures_sep" in config else False,
                             cross_section=config["cross_section"] if "cross_section" in config else False
                             )
        dataset.shuffle(config["seed"])

        train_set, validation_set, test_set = _split_dataset(dataset, config, n_train_samples)
    else:
        train_dataset.init_transform = data_init_transform
        train_dataset.input_transform = data_input_transform
        train_dataset.output_transform = data_output_transform

    if "input_transform" in config or "output_transform" in config:
        if train_dataset is not None:
            train_set = train_dataset
        input_transformations, output_transformations = features_transform(config, data_dir, output_file_name,
                                                                           input_transformations,
                                                                           output_transformations, train_set)
        if len(output_transformations) > 0:
            data_output_transform = transforms.Compose(output_transformations)

        if len(input_transformations) > 0:
            data_input_transform = transforms.Compose(input_transformations)

        dataset = DFMDataset(data_dir=data_dir,
                             output_file_name=output_file_name,
                             input_transform=data_input_transform,
                             output_transform=data_output_transform,
                             init_transform=data_init_transform,
                             init_norm_use_all_features=config[
                                 "init_norm_use_all_features"] if "init_norm_use_all_features" in config else False,
                             input_channels=config["input_channels"] if "input_channels" in config else None,
                             output_channels=config["output_channels"] if "output_channels" in config else None,
                             fractures_sep=config["fractures_sep"] if "fractures_sep" in config else False,
                             cross_section=config["cross_section"] if "cross_section" in config else False
                             )
        dataset.shuffle(config["seed"])

        train_set, validation_set, test_set = _split_dataset(dataset, config, n_train_samples)

        # train_loader_mean_std = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size_train"],
        #                                                     shuffle=False)
        # input_mean, input_std, output_mean, output_std, _ = get_mean_std(train_loader_mean_std)
        # print("TRAIN SET input mean: {}, std:{}, output mean: {}, std: {}".format(input_mean, input_std, output_mean, output_std))
        #
        # train_loader_mean_std = torch.utils.data.DataLoader(validation_set, batch_size=config["batch_size_train"],
        #                                                     shuffle=False)
        # input_mean, input_std, output_mean, output_std, _ = get_mean_std(train_loader_mean_std)
        # print("VAL SET input mean: {}, std:{}, output mean: {}, std: {}".format(input_mean, input_std, output_mean,
        #                                                                           output_std))
        #
        # train_loader_mean_std = torch.utils.data.DataLoader(test_set, batch_size=config["batch_size_train"],
        #                                                     shuffle=False)
        # input_mean, input_std, output_mean, output_std, _ = get_mean_std(train_loader_mean_std)
        # print("TEST SET input mean: {}, std:{}, output mean: {}, std: {}".format(input_mean, input_std, output_mean,
        #                                                                           output_std))

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

    if train_dataset is not None:
        return data_init_transform, data_input_transform, data_output_transform

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
