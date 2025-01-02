import zarr
import numpy as np


def merge_zarr_datasets(output_path, zarr_paths, dataset_keys=['inputs', 'outputs', 'bulk_avg']):
    # Open the Zarr datasets and store them in a list
    zarr_files = [zarr.open(zarr_path, mode='r') for zarr_path in zarr_paths]

    # Get shape and dtype from the first dataset (assuming all have the same structure)
    shape1 = zarr_files[0][dataset_keys[0]].shape
    shape2 = zarr_files[0][dataset_keys[1]].shape
    dtype1 = zarr_files[0][dataset_keys[0]].dtype
    dtype2 = zarr_files[0][dataset_keys[1]].dtype
    chunks1 = zarr_files[0][dataset_keys[0]].chunks
    chunks2 = zarr_files[0][dataset_keys[1]].chunks

    # Calculate the total number of samples across all datasets
    total_samples = sum(zarr_file[dataset_keys[0]].shape[0] for zarr_file in zarr_files)

    # Create the merged Zarr store with the combined shape for both datasets
    merged_store = zarr.open(output_path, mode='w')
    merged_store.create_dataset(dataset_keys[0], shape=(total_samples,) + shape1[1:], dtype=dtype1, chunks=chunks1)
    merged_store.create_dataset(dataset_keys[1], shape=(total_samples,) + shape2[1:], dtype=dtype2, chunks=chunks2)
    merged_store.create_dataset(dataset_keys[2], shape=(total_samples,) + shape2[1:], dtype=dtype2, chunks=chunks2)

    # Merge datasets chunk by chunk into the merged_store
    sample_offset = 0
    for zarr_file in zarr_files:
        num_samples = zarr_file[dataset_keys[0]].shape[0]

        # Copy each chunk into the merged_store at the appropriate offset for both datasets
        for i in range(num_samples):
            merged_store[dataset_keys[0]][sample_offset + i] = zarr_file[dataset_keys[0]][i]
            merged_store[dataset_keys[1]][sample_offset + i] = zarr_file[dataset_keys[1]][i]
            merged_store[dataset_keys[2]][sample_offset + i] = zarr_file[dataset_keys[2]][i]

        sample_offset += num_samples

    return merged_store


def merge_zarr_testdatasets(test_dataset_path, zarr_paths, dataset_keys=['inputs', 'outputs', 'bulk_avg']):
    # Open the Zarr datasets and store them in a list

    zarr_files = [(zarr.open(zarr_path, mode='r'), n_train_val) for zarr_path, n_train_val in zarr_paths.items()]
    total_samples_test = sum([n_train_val["n_test"] for n_train_val in zarr_paths.values()])
    #total_samples_val = sum([n_train_val["n_val"] for n_train_val in zarr_paths.values()])

    print("total total_samples_test ", total_samples_test)
    #print("total samples val ", total_samples_val)


    # Get shape and dtype from the first dataset (assuming all have the same structure)
    shape1 = zarr_files[0][0][dataset_keys[0]].shape
    shape2 = zarr_files[0][0][dataset_keys[1]].shape
    dtype1 = zarr_files[0][0][dataset_keys[0]].dtype
    dtype2 = zarr_files[0][0][dataset_keys[1]].dtype
    chunks1 = zarr_files[0][0][dataset_keys[0]].chunks
    chunks2 = zarr_files[0][0][dataset_keys[1]].chunks

    # # Calculate the total number of samples across all datasets
    # total_samples = sum(zarr_file[dataset_keys[0]].shape[0] for zarr_file in zarr_files)

    #####
    ### Test datasets
    #####
    # Create the merged Zarr store with the combined shape for both datasets
    testset_store = zarr.open(test_dataset_path, mode='w')
    testset_store.create_dataset(dataset_keys[0], shape=(total_samples_test,) + shape1[1:], dtype=dtype1, chunks=chunks1)
    testset_store.create_dataset(dataset_keys[1], shape=(total_samples_test,) + shape2[1:], dtype=dtype2, chunks=chunks2)
    testset_store.create_dataset(dataset_keys[2], shape=(total_samples_test,) + shape2[1:], dtype=dtype2, chunks=chunks2)

    # Merge datasets chunk by chunk into the merged_store
    sample_offset = 0
    for zarr_file, n_train_val in zarr_files:
        num_samples = zarr_file[dataset_keys[0]].shape[0]

        # Copy each chunk into the merged_store at the appropriate offset for both datasets
        for i in range(n_train_val["n_test"]):
            testset_store[dataset_keys[0]][sample_offset + i] = zarr_file[dataset_keys[0]][i]
            testset_store[dataset_keys[1]][sample_offset + i] = zarr_file[dataset_keys[1]][i]
            testset_store[dataset_keys[2]][sample_offset + i] = zarr_file[dataset_keys[2]][i]

        sample_offset += n_train_val["n_test"]

    return testset_store


def shuffle_zarr_datasets(merged_store, output_path, dataset_keys=['inputs', 'outputs', 'bulk_avg']):
    # Get the total number of samples in the merged datasets
    total_samples = merged_store[dataset_keys[0]].shape[0]

    # Create a new Zarr store for the shuffled datasets
    shuffled_store = zarr.open(output_path, mode='w')
    shuffled_store.create_dataset(dataset_keys[0], shape=merged_store[dataset_keys[0]].shape,
                                  dtype=merged_store[dataset_keys[0]].dtype,
                                  chunks=merged_store[dataset_keys[0]].chunks)
    shuffled_store.create_dataset(dataset_keys[1], shape=merged_store[dataset_keys[1]].shape,
                                  dtype=merged_store[dataset_keys[1]].dtype,
                                  chunks=merged_store[dataset_keys[1]].chunks)
    shuffled_store.create_dataset(dataset_keys[2], shape=merged_store[dataset_keys[2]].shape,
                                  dtype=merged_store[dataset_keys[2]].dtype,
                                  chunks=merged_store[dataset_keys[2]].chunks)

    # Generate a shuffled list of indices
    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    # Write the shuffled data to the new Zarr store for both datasets
    for i, idx in enumerate(indices):
        shuffled_store[dataset_keys[0]][i] = merged_store[dataset_keys[0]][idx]
        shuffled_store[dataset_keys[1]][i] = merged_store[dataset_keys[1]][idx]
        shuffled_store[dataset_keys[2]][i] = merged_store[dataset_keys[2]][idx]

    return shuffled_store


# Example usage: Provide paths to your datasets
zarr_paths = {"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_4/cond_nearest_interp/test_data/MLMC-DFM_3D_n_voxels_64_save_bulk_avg/samples_data.zarr": {"n_test": 1500},
              "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_5/cond_nearest_interp/test_data/MLMC-DFM_3D_n_voxels_64_save_bulk_avg/samples_data.zarr": {"n_test": 1500},
              "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_6/cond_nearest_interp/test_data/MLMC-DFM_3D_n_voxels_64_save_bulk_avg/samples_data.zarr": {"n_test": 1500},
              }

import os
import shutil
for i, (zarr_file, n_samples) in enumerate(zarr_paths.items()):
    print("i ", i)
    print("zarr file ", zarr_file)
    print("n samples ", n_samples)
    # scratch_zarr_path = os.path.join(scratch_dir, "large_dataset.zarr")
    # shutil.move(original_zarr_path, scratch_zarr_path)

testset_store = merge_zarr_testdatasets('test_dataset.zarr', zarr_paths, dataset_keys=['inputs', 'outputs', 'bulk_avg'])

shuffled_trainset_store = shuffle_zarr_datasets(testset_store, 'shuffled_testset.zarr', dataset_keys=['inputs', 'outputs', 'bulk_avg'])
#shuffled_valset_store = shuffle_zarr_datasets(valset_store, 'shuffled_valset.zarr', dataset_keys=['inputs', 'outputs', 'bulk_avg'])

# zarr_paths_list = ['shuffled_trainset.zarr',  'shuffled_valset.zarr']
#
# merged_store = merge_zarr_datasets('merged_final_dset.zarr', zarr_paths_list, dataset_keys=['inputs', 'outputs', 'bulk_avg'])
#
#
#
# print("merged_store['inputs'].shape ", merged_store['inputs'].shape)


