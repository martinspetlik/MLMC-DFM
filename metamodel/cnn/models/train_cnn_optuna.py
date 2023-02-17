import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna
from optuna.trial import TrialState
from metamodel.cnn.models.trials.net_optuna import Net
import time
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from metamodel.cnn.models.trials.net_1 import Net1
from metamodel.cnn.datasets.dfm_dataset import DFMDataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from metamodel.cnn.models.auxiliary_functions import get_mean_std, log_data


def train(network, optimizer):
    """Trains the model.

    Parameters:
        - network (__main__.Net):              The CNN
        - optimizer (torch.optim.<optimizer>): The optimizer for the CNN
    """
    network.train()  # Set the module in training mode (only affects certain modules)
    for batch_i, (data, target) in enumerate(train_loader):  # For each batch

        # # Limit training data for faster computation
        # if batch_i * batch_size_train > n_train_samples:
        #     break

        optimizer.zero_grad()                                 # Clear gradients
        output = network(data.to(device))                     # Forward propagation
        loss = nn.MSELoss(output, target.to(device))          # Compute loss (negative log likelihood: −log(y))
        loss.backward()                                       # Compute gradients
        optimizer.step()                                      # Update weights


def test(network):
    """Tests the model.

    Parameters:
        - network (__main__.Net): The CNN

    Returns:
        - accuracy_test (torch.Tensor): The test accuracy
    """
    network.eval()         # Set the module in evaluation mode (only affects certain modules)
    correct = 0
    with torch.no_grad():  # Disable gradient calculation (when you are sure that you will not call Tensor.backward())
        for batch_i, (data, target) in enumerate(test_loader):  # For each batch

            # # Limit testing data for faster computation
            # if batch_i * batch_size_test > numbtest_examples:
            #     break

            output = network(data.to(device))               # Forward propagation
            pred = output.data.max(1, keepdim=True)[1]      # Find max value in each row, return indexes of max values
            correct += pred.eq(target.to(device).data.view_as(pred)).sum()  # Compute correct predictions

    accuracy_test = correct / len(test_loader.dataset)

    return accuracy_test


def objective(trial):
    """Objective function to be optimized by Optuna.

    Hyperparameters chosen to be optimized: optimizer, learning rate,
    dropout values, number of convolutional layers, number of filters of
    convolutional layers, number of neurons of fully connected layers.

    Inputs:
        - trial (optuna.trial._trial.Trial): Optuna trial
    Returns:
        - accuracy(torch.Tensor): The test accuracy. Parameter to be maximized.
    """

    # Define range of values to be tested for the hyperparameters
    num_conv_layers = trial.suggest_int("num_conv_layers", 2, 4)  # Number of convolutional layers
    same_channels = trial.suggest_categorical("same_channels", [False, True])

    channels = []

    print("range", list(range(3, 48, 6)))
    print("np linspace ", int(np.linspace(3, 48, num=num_conv_layers)))
    exit()
    for i in range(num_conv_layers):
        channels.append(trial.suggest_int())

    #@TODO: suggest discrete uniform with suggest_int

    channels = [int(trial.suggest_discrete_uniform("channels_"+str(i), 3, 48, 6))
                   for i in range(num_conv_layers)]              # Number of filters for the convolutional layers

    print("channels ", channels)


    num_neurons = trial.suggest_int("num_neurons", 10, 400, 10)  # Number of neurons of FC1 layer

    pool = trial.suggest_categorical("pool", ["max", "avg"])
    drop_conv2 = trial.suggest_float("drop_conv2", 0., 0.5)     # Dropout for convolutional layer 2
    drop_fc1 = trial.suggest_float("drop_fc1", 0., 0.5)         # Dropout for FC1 layer

    # Generate the model
    model = Net(trial, num_conv_layers, pool, channels, num_neurons, drop_conv2,  drop_fc1, same_channels).to(device)

    # Generate the optimizers
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])  # Optimizers
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)                                 # Learning rates
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Training of the model
    for epoch in range(num_epochs):
        train(model, optimizer)  # Train the model
        accuracy = test(model)   # Evaluate the model

        # For pruning (stops trial early if not promising)
        trial.report(accuracy, epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Optimization study for a PyTorch CNN with Optuna
    # -------------------------------------------------------------------------

    # Use cuda if available for faster computations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ============
    # config
    # ===========
    data_dir = '/home/martin/Documents/MLMC-DFM/test/01_cond_field/nn_data/homogenization_samples_no_fractures'
    use_cuda = False
    num_epochs = 50
    batch_size_train = 25
    batch_size_test = 250
    train_samples_ratio = 0.8
    val_samples_ratio = 0.2
    print_batches = 10
    log_input = False
    normalize_input = True

    # Ooptuna params
    num_trials = 100

    # limit_obs = True
    #
    #
    # # *** Note: For more accurate results, do not limit the observations.
    # #           If not limited, however, it might take a very long time to run.
    # #           Another option is to limit the number of epochs. ***
    #
    # if limit_obs:  # Limit number of observations
    #     number_of_train_examples = 500 * batch_size_train  # Max train observations
    #     number_of_test_examples = 5 * batch_size_test      # Max test observations
    # else:
    #     number_of_train_examples = 60000                   # Max train observations
    #     number_of_test_examples = 10000                    # Max test observations
    # # -------------------------------------------------------------------------

    # Make runs repeatable
    random_seed = 12345
    torch.backends.cudnn.enabled = False  # Disable cuDNN use of nondeterministic algorithms
    torch.manual_seed(random_seed)

    # # Create directory 'files', if it doesn't exist, to save the dataset
    # directory_name = 'files'
    # if not os.path.exists(directory_name):
    #     os.mkdir(directory_name)
    #
    # # Download MNIST dataset to 'files' directory and normalize it
    # train_loader = torch.utils.data.DataLoader(
    #     torchvision.datasets.MNIST('/files/', train=True, download=True,
    #                                transform=torchvision.transforms.Compose([
    #                                    torchvision.transforms.ToTensor(),
    #                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
    #     batch_size=batch_size_train, shuffle=True)
    #
    # test_loader = torch.utils.data.DataLoader(
    #     torchvision.datasets.MNIST('/files/', train=False, download=True,
    #                                transform=torchvision.transforms.Compose([
    #                                    torchvision.transforms.ToTensor(),
    #                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
    #     batch_size=batch_size_test, shuffle=True)

    # ===================================
    # Get mean and std for each channel
    # ===================================
    mean, std = 0, 1
    if normalize_input:
        dataset_for_mean_std = DFMDataset(data_dir=data_dir, transform=None, two_dim=True)

        n_train_samples = int(len(dataset_for_mean_std) * train_samples_ratio)

        train_val_set = dataset_for_mean_std[:n_train_samples]
        train_set = train_val_set[int(n_train_samples * val_samples_ratio):]

        train_loader_mean_std = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=False)
        mean, std, output_mean, output_std = get_mean_std(train_loader_mean_std)

    # =======================
    # data transforms
    # =======================
    transformations = []
    data_transform = None
    if log_input:
        transformations.append(transforms.Lambda(log_data))
    if normalize_input:
        transformations.append(transforms.Normalize(mean=mean, std=std))

    if len(transformations) > 0:
        data_transform = transforms.Compose(transformations)

    # ============================
    # Datasets and data loaders
    # ============================
    dataset = DFMDataset(data_dir=data_dir, transform=data_transform)

    n_train_samples = int(len(dataset) * train_samples_ratio)

    train_val_set = dataset[:n_train_samples]
    validation_set = train_val_set[:int(n_train_samples * val_samples_ratio)]
    train_set = train_val_set[int(n_train_samples * val_samples_ratio):]
    test_set = dataset[n_train_samples:]
    print("len(trainset): {}, len(valset): {}, len(testset): {}".format(len(train_set), len(validation_set),
                                                                        len(test_set)))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_test, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, shuffle=False)

    # Create an Optuna study to maximize test accuracy
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=num_trials)

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------

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
