import os
import re
import copy
import torch
import pandas as pd
#from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from metamodel.cnn.datasets.dfm_dataset import DFMDataset
from metamodel.cnn.models.auxiliary_functions import get_mean_std, log_data

dir_name = '/home/martin/Documents/MLMC-DFM/test/01_cond_field/homogenization_samples_dfm'


def test_dataset():
    pass
    # @TODO: first create dataset and then check data shapes etc. should also works on github

def test_transform():
    dataset_for_mean_std = DFMDataset(data_dir=dir_name, transform=None, two_dim=True)
    train_loader_mean_std = torch.utils.data.DataLoader(dataset_for_mean_std, batch_size=20, shuffle=False)
    mean, std = get_mean_std(train_loader_mean_std)


    dfm_dataset_no_trf = DFMDataset(data_dir=dir_name, two_dim=True)
    trf = transforms.Compose([transforms.Lambda(log_data),transforms.Normalize(mean=mean, std=std)])
    dfm_dataset_trf = DFMDataset(data_dir=dir_name, transform=trf, two_dim=True)

    # features = final_features  # np.transpose(final_features, (2, 0, 1))
    for data, data_trf in zip(dfm_dataset_no_trf, dfm_dataset_trf):
        input, output = data
        input_trf, output_trf = data_trf

    #     plt.gray()
    #     plt.imshow(feature)
    #     plt.show()
    #
    #     plt.gray()
    #     plt.imshow(feature_transform)
    #     plt.show()
    #
    # exit()



if __name__ == "__main__":
    test_transform()


for i in range(len(dfm_dataset)):
    sample = dfm_dataset[i]

    # Plot features
    features = np.transpose(sample[0], (2, 0, 1))
    for feature in features:
        plt.gray()
        plt.imshow(feature)
        plt.show()


#print("final features shape ", final_features.shape)

        # Display final features
        #features = np.transpose(final_features, (2, 0, 1))
        #print("features.shape ", features.shape)
        #exit()
        # for feature in final_features:
        #     plt.gray()
        #     plt.imshow(feature)
        #     plt.show()

        #print("final_features.shape ", final_features.shape)


      #print("final features torch type ", type(final_features))

        from PIL import Image

        # counts, bins = np.histogram(final_features[0].ravel())
        # plt.stairs(counts, bins)

        # plt.hist(np.log(final_features[0].ravel()), bins=60, density=True)
        # plt.xlabel("Values of Pixel")
        # plt.ylabel("Frequency for relative")
        # plt.title("pixel distribution no transform")
        # plt.show()

    # plt.hist(final_features[0].ravel(), bins=60, density=True)
    # plt.xlabel("Values of Pixel")
    # plt.ylabel("Frequency for relative")
    # plt.title("pixel distribution transform")
    # plt.show()
    #
    #
    # exit()

    #
    # features = final_features  # np.transpose(final_features, (2, 0, 1))
    # for feature, feature_transform in zip(final_features, final_features_transforms):
    #     print("feature ", feature)
    #     print("feature_transform ", feature_transform)
    #     plt.gray()
    #     plt.imshow(feature)
    #     plt.show()
    #
    #     plt.gray()
    #     plt.imshow(feature_transform)
    #     plt.show()
    #
    # exit()

    # final_features = np.transpose(final_features, (1, 2, 0))

