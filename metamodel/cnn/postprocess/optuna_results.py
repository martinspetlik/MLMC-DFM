import os
import sys
import argparse
import joblib
import torch
import torchvision
import numpy as np
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
from metamodel.cnn.visualization.visualize_data import plot_target_prediction
from metamodel.cnn.models.auxiliary_functions import exp_data


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

    inverse_transform = None
    if "normalize_output" in study.user_attrs and study.user_attrs["normalize_output"]:
        std = 1/study.user_attrs["output_std"]
        zeros_mean = np.zeros(len(study.user_attrs["output_mean"]))

        print("output_mean ", study.user_attrs["output_mean"])
        print("output_std ",  study.user_attrs["output_std"])

        ones_std = np.ones(len(zeros_mean))
        mean = -study.user_attrs["output_mean"]

        transforms_list = [transforms.Normalize(mean=zeros_mean, std=std),
                            transforms.Normalize(mean=mean, std=ones_std)]

        if "output_log" in study.user_attrs:
            transforms_list.append(transforms.Lambda(exp_data))

        inverse_transform = transforms.Compose(transforms_list)


    plot_separate_images = False
    # Disable grad
    with torch.no_grad():
        # Initialize model
        model = study.best_trial.user_attrs["model_class"](**study.best_trial.user_attrs["model_kwargs"])
        # Initialize optimizer
        optimizer = study.best_trial.user_attrs["optimizer_class"](model.parameters(),
                                                                   **study.best_trial.user_attrs["optimizer_kwargs"])

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        train_loss = checkpoint['train_loss']
        valid_loss = checkpoint['valid_loss']

        running_loss, inv_running_loss = 0, 0
        targets_list, predictions_list = [], []
        inv_targets_list, inv_predictions_list = [], []
        for i, test_sample in enumerate(test_loader):
            inputs, targets = test_sample
            inputs = inputs.float()
            predictions = model(inputs)

            targets = torch.squeeze(targets.float())
            predictions = torch.squeeze(predictions)

            targets_list.append(targets.numpy())
            predictions_list.append(predictions.numpy())

            loss_fn = study.best_trial.user_attrs["loss_fn_name"]()
            loss = loss_fn(predictions, targets)
            running_loss += loss

            inv_targets = targets
            inv_predictions = predictions
            if inverse_transform is not None:
                inv_targets = inverse_transform(torch.reshape(targets, (*targets.shape, 1, 1)))
                inv_predictions = inverse_transform(torch.reshape(predictions, (*predictions.shape, 1, 1)))

            inv_running_loss += loss_fn(inv_predictions, inv_targets)


            inv_targets_list.append(inv_targets.numpy())
            inv_predictions_list.append(inv_predictions.numpy())

            # if i % 10 == 9:
            #     plot_tensors(predictions.numpy(), targets.numpy(), label="test_sample_{}".format(i),
            #                  plot_separate_images=plot_separate_images)

        test_loss = running_loss / (i + 1)
        inv_test_loss = inv_running_loss / (i + 1)

        #plot_target_prediction(targets_list, predictions_list)
        plot_target_prediction(np.array(inv_targets_list), np.array(inv_predictions_list))

        print("epochs: {}, train loss: {}, valid loss: {}, test loss: {}, inv test loss: {}".format(epoch,
                                                                                                    train_loss,
                                                                                                    valid_loss,
                                                                                                    test_loss,
                                                                                                    inv_test_loss))

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
