import os
import sys
import argparse
import joblib
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from metamodel.cnn.models.net1 import Net1
from metamodel.cnn.models.trials.net_optuna_2 import Net
from metamodel.cnn.datasets.dfm_dataset import DFMDataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from metamodel.cnn.visualization.visualize_tensor import plot_tensors


def get_saved_model_path(results_dir, best_trial):
    model_path = 'model_{}_{}'.format(best_trial.user_attrs["model_name"], best_trial.user_attrs["epoch"])
    for key, value in best_trial.params.items():
        model_path += "_{}_{}".format(key, value)
    return os.path.join(results_dir, model_path)



def load_models(results_dir, study):
    model_path = get_saved_model_path(results_dir, study.best_trial)
    dataset = joblib.load(os.path.join(results_dir, "dataset.pkl"))
    test_set = dataset[:-study.user_attrs["n_test_samples"]]
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False)

    # Disable grad
    with torch.no_grad():
        # Initialize model
        model = study.best_trial.user_attrs["model_class"](**study.best_trial.user_attrs["model_kwargs"])
        # Initialize optimizer
        optimizer = study.best_trial.user_attrs["optimizer_class"](model.parameters(),
                                                                   **study.best_trial.user_attrs["optimizer_kwargs"])
        print("optimizer ", optimizer)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("model ", model)
        exit()
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("checkpoint['optimizer_state_dict'] ", checkpoint['optimizer_state_dict'])
        print("optimizer ", optimizer)
        exit()
        epoch = checkpoint['epoch']
        train_loss = checkpoint['train_loss']
        valid_loss = checkpoint['valid_loss']

        running_loss = 0
        for i, test_sample in enumerate(test_loader):
            inputs, targets = test_sample
            inputs = inputs.float()
            targets = targets.float()

            outputs = cnn_model(inputs)

            outputs_numpy = outputs.numpy()
            targets_numpy = targets.numpy()

            if i % 10 == 9:
                plot_tensors(outputs_numpy, targets_numpy, label="test_sample_{}".format(i), separate_images=separate_images)

            loss = loss_fn(outputs, targets)
            running_loss += loss

        test_loss = running_loss / (i + 1)

        print("epochs: {}, train loss: {}, valid loss: {}, test loss".format(epoch, train_loss, valid_loss, test_loss))


def load_study(results_dir):
    study = joblib.load(os.path.join(results_dir, "study.pkl"))
    print("Best trial until now:")
    print(" Value: ", study.best_trial.value)
    print(" Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', help='results directory')
    parser.add_argument("-c", "--cuda", default=False, action='store_true', help="use cuda")
    args = parser.parse_args(sys.argv[1:])

    study = load_study(args.results_dir)
    load_models(args.results_dir, study)
