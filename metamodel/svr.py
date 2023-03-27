import numpy as np
import torch
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
from metamodel.cnn.datasets.dfm_dataset import DFMDataset
from metamodel.cnn.models.auxiliary_functions import get_mean_std, log_data


dir_name = "/home/martin/Documents/MLMC-DFM_data/nn_data/homogenization_samples_3_3_charon"

log_input, log_output = True, False
norm_input, norm_output = True, True

input_trf, output_trf = None, None
if log_input:
    input_trf = transforms.Lambda(log_data)
if log_output:
    output_trf = transforms.Lambda(log_data)

dataset_for_mean_std = DFMDataset(data_dir=dir_name,
                                  input_transform=input_trf,
                                  output_transform=output_trf,
                                  two_dim=True)

train_loader_mean_std = torch.utils.data.DataLoader(dataset_for_mean_std, batch_size=20, shuffle=False)
mean, std, output_mean, output_std = get_mean_std(train_loader_mean_std)

print("mean: {}, std: {}".format(mean, std))
print("output mean: {}, output std: {}".format(output_mean, output_std))

dfm_dataset_no_trf = DFMDataset(data_dir=dir_name, two_dim=True)

input_trf_list = []
output_trf_list = []

if log_input:
    input_trf_list.append(transforms.Lambda(log_data))
if log_output:
    output_trf_list.append(transforms.Lambda(log_data))
if norm_input:
    input_trf_list.append(transforms.Normalize(mean=mean, std=std))
if norm_output:
    output_trf_list.append(transforms.Normalize(mean=output_mean, std=output_std))

trf_input = transforms.Compose(input_trf_list)
trf_output = transforms.Compose(output_trf_list)

dfm_dataset = DFMDataset(data_dir=dir_name, input_transform=trf_input, output_transform=trf_output, two_dim=True)

train_samples_ratio = 0.8
val_samples_ratio = 0.2
batch_size = 4096

train_val_set = dfm_dataset[:int(len(dfm_dataset) * train_samples_ratio)]
train_set = train_val_set[:-int(len(train_val_set) * val_samples_ratio)]
val_set = train_val_set[-int(len(train_val_set) * val_samples_ratio):]
test_set = dfm_dataset[int(len(dfm_dataset) * train_samples_ratio):]

print("len(train_set): {}, len(val_set): {}, len(test_set): {}".format(len(train_set), len(val_set), len(test_set)))

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(val_set, shuffle=False)
test_loader = DataLoader(test_set, shuffle=False)


# Create SVR model
svr = SVR(kernel='rbf', C=1, epsilon=0.1, gamma='auto', verbose=True)
multi_svr = MultiOutputRegressor(svr)

loss_fn = nn.MSELoss()

for data in train_loader:
    inputs, outputs = data
    print("inputs.shape ", inputs.shape)
    multi_svr.fit(inputs.reshape(np.min([inputs.shape[0], batch_size]), -1), outputs)


running_loss = 0
total_score = 0
for i, (inputs, outputs) in enumerate(validation_loader):
    #inputs, outputs = data
    y_pred = multi_svr.predict(np.reshape(inputs, (inputs.shape[0], -1)))

    # Reshape output data to original shape
    y_pred = np.reshape(y_pred, (y_pred.shape[0], 3))

    loss = mean_squared_error(outputs, y_pred)
    score = multi_svr.score(np.reshape(inputs, (inputs.shape[0], -1)), outputs)
    # Gather data and report
    running_loss += loss.item()
    total_score += score


validation_loss = running_loss / (i + 1)
score = total_score / (i+1)

print("validation loss: {}".format(validation_loss))
#print("R2 score: {}".format(score))

# validation loss: 0.8179482723302068
