import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from metamodel.cnn.models.net1 import Net1
from metamodel.cnn.datasets.dfm_dataset import DFMDataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


num_epochs = 5
batch_size = 25
n_train_samples = 400
val_samples_ratio = 0.2

dataset = DFMDataset(data_dir='/home/martin/Documents/MLMC-DFM/test/01_cond_field/homogenization_samples')

train_val_set = dataset[:n_train_samples]
validation_set = train_val_set[:int(n_train_samples*val_samples_ratio)]
train_set = train_val_set[int(n_train_samples*val_samples_ratio):]
test_set = dataset[n_train_samples:]
print("len(trainset): {}, len(valset): {}, len(testset): {}".format(train_set, validation_set, test_set))

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

cnn_model = Net1()
if torch.cuda.is_available():
    cnn_model = cnn_model.cuda()

loss_fn = nn.MSELoss()
optimizer = optim.SGD(cnn_model.parameters(), lr=0.001, momentum=0.9)

best_vloss = 1_000_000.


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        inputs, targets = data

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            target = targets.cuda()

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = cnn_model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, targets)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

for epoch in enumerate(range(num_epochs)):
    print('EPOCH {}:'.format(epoch ))

    # Make sure gradient tracking is on, and do a pass over the data
    cnn_model.train(True)
    avg_loss = train_one_epoch(epoch, writer)

    # We don't need gradients on to do reporting
    cnn_model.train(False)

    running_vloss = 0.0
    for i, vdata in enumerate(validation_loader):
        vinputs, vlabels = vdata
        voutputs = cnn_model(vinputs)
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch)
        torch.save(cnn_model.state_dict(), model_path)



# for epoch in range(20):  # loop over the dataset multiple times
#
#     running_loss = 0.0
#     for i, data in enumerate(train_loader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, targets = data
#
#         # print("torch.cuda.is_available() ", torch.cuda.is_available())
#         # if torch.cuda.is_available():
#         #     print("cuda is available")
#         #inputs, targets = inputs.cuda(), targets.cuda()
#
#         inputs = inputs.float()
#         targets = targets.float()
#
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # forward + backward + optimize
#         outputs = cnn_model(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()
#
#         # print statistics
#         running_loss += loss.item()
#         if i % 10 == 9:    # print every 10 mini-batches
#             print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
#             running_loss = 0.0




#print('Finished Training')