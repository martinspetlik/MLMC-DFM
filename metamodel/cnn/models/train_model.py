import os
import sys
import time
import torch
#import torchvision
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from metamodel.cnn.models.trials.net_1 import Net1
from metamodel.cnn.models.trials.net_optuna_2 import Net
from metamodel.cnn.datasets.dfm_dataset import DFMDataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from metamodel.cnn.models.auxiliary_functions import get_mean_std, log_data

def train_model(data_dir, use_cuda=False):

    #============
    # config
    #===========
    num_epochs = 5
    batch_size = 10
    train_samples_ratio = 0.8
    val_samples_ratio = 0.2
    print_batches = 50
    log_input = False
    normalize_input = True


    #===================================
    # Get mean and std for each channel
    #===================================
    mean, std = 0, 1
    if normalize_input:
        dataset_for_mean_std = DFMDataset(data_dir=data_dir,transform=None, two_dim=True)

        n_train_samples = int(len(dataset_for_mean_std) * train_samples_ratio)

        train_val_set = dataset_for_mean_std[:n_train_samples]
        train_set = train_val_set[int(n_train_samples * val_samples_ratio):]

        train_loader_mean_std = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
        mean, std, mean_out, std_out = get_mean_std(train_loader_mean_std)


    #=======================
    # data transforms
    #=======================
    transformations = []
    data_transform = None
    if log_input:
        transformations.append(transforms.Lambda(log_data))
    if normalize_input:
        transformations.append(transforms.Normalize(mean=mean, std=std))

    if len(transformations) > 0:
        data_transform = transforms.Compose(transformations)


    #============================
    # Datasets and data loaders
    #============================
    dataset = DFMDataset(data_dir=data_dir, transform=data_transform)

    n_train_samples = int(len(dataset) * train_samples_ratio)

    train_val_set = dataset[:n_train_samples]
    validation_set = train_val_set[:int(n_train_samples*val_samples_ratio)]
    train_set = train_val_set[int(n_train_samples*val_samples_ratio):]
    test_set = dataset[n_train_samples:]
    print("len(trainset): {}, len(valset): {}, len(testset): {}".format(len(train_set), len(validation_set), len(test_set)))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)


    #===========================
    # Model, loss, optimizer
    #===========================
    def set_model(code):

        if code == "Net":
            trial = None
            max_channel = 3
            kernel_size = 3
            stride = 2
            pool = None

            return Net(trial, pool=pool, max_channel=max_channel, kernel_size=kernel_size, stride=stride)

        else:
            return Net1()

    cnn_model = set_model("Net")
    print("torch.cuda.is_available ", torch.cuda.is_available())

    if torch.cuda.is_available() and use_cuda:
        cnn_model = cnn_model.cuda()

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(cnn_model.parameters(), lr=0.001, momentum=0.9)

    best_vloss = 1_000_000.

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/dfm_trainer_{}'.format(timestamp))

    #=========================
    # Loop through epochs
    #=========================
    start_time = time.time()
    for epoch in range(num_epochs):
        # Make sure gradient tracking is on, and do a pass over the data
        cnn_model.train(True)
        avg_loss = train_one_epoch(train_loader, cnn_model, optimizer, loss_fn, epoch, writer, use_cuda=use_cuda,
                                   print_batches=print_batches)

        # We don't need gradients on to do reporting
        cnn_model.train(False)
        #torch.cuda.empty_cache()

        # validation loss
        running_vloss = 0.0
        for i, vdata in enumerate(validation_loader):
            vinputs, vtargets = vdata

            if torch.cuda.is_available() and use_cuda:
                vinputs = vinputs.cuda()
                vtargets = vtargets.cuda()

            vinputs = vinputs.float()
            vtargets = vtargets.float()

            voutputs = torch.squeeze(cnn_model(vinputs))
            vloss = loss_fn(voutputs, vtargets)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        {'Training' : avg_loss, 'Validation' : avg_vloss },
                        int(epoch)+1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            print("os.getcwd() ", os.getcwd())
            model_path = os.path.join(os.getcwd(),
                                      'model_{}_{}_{}'.format(cnn_model._name, timestamp, epoch))
            torch.save(cnn_model.state_dict(), model_path)

            torch.save({
                'epoch': int(epoch)+1,
                'model_state_dict': cnn_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_loss,
                'valid_loss': avg_vloss,
                'training_time': time.time()-start_time,
                'data_mean': mean,
                'data_std': std
            }, model_path)


#================================
# one epoch training procedure
#================================
def train_one_epoch(train_loader, cnn_model, optimizer, loss_fn, epoch_index, tb_writer, use_cuda=False, print_batches=10):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(train_loader):
        inputs, targets = data

        if torch.cuda.is_available() and use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        inputs = inputs.float()
        targets = targets.float()

        optimizer.zero_grad()

        outputs = torch.squeeze(cnn_model(inputs))
        loss = loss_fn(outputs, targets)
        loss.backward()

        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % print_batches == print_batches-1:
            last_loss = running_loss / print_batches # loss per batch
            print(' batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


if __name__ == "__main__":
    print("torch.cuda.is_available() ", torch.cuda.is_available())
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='Data directory')
    parser.add_argument("-c", "--cuda", default=False, action='store_true', help="use cuda")
    args = parser.parse_args(sys.argv[1:])
    train_model(args.data_dir, use_cuda=args.cuda)
