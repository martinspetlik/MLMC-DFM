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
from metamodel.cnn.visualization.visualize_tensor import plot_tensors

saved_model_path = "model_net1_20230214_155947_49"
loss_fn = nn.MSELoss()

# cnn_model = Net1()
# cnn_model.load_state_dict(torch.load(saved_model_path))

dataset = DFMDataset(data_dir='/home/martin/Documents/MLMC-DFM/test/01_cond_field/homogenization_samples')
#@TODO: test data generation should not depend on train samples
n_train_samples = 400
test_data = test_set = dataset[n_train_samples:]
test_loader = torch.utils.data.DataLoader(test_set, shuffle=False)

separate_images=True


# Disable grad
with torch.no_grad():
    cnn_model = Net1()
    optimizer = optim.SGD(cnn_model.parameters(), lr=0.001, momentum=0.9)
    checkpoint = torch.load(saved_model_path)
    cnn_model.load_state_dict(checkpoint['model_state_dict'])
    cnn_model.eval()
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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
