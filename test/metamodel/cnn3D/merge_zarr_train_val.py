import zarr
import numpy as np
import gc


def merge_zarr_datasets(output_path, zarr_paths, dataset_keys=['inputs', 'outputs', 'bulk_avg']):
    # Open the Zarr datasets and store them in a list
    zarr_files = [zarr.open(zarr_path, mode='r', cache_attrs=False) for zarr_path in zarr_paths]

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
    merged_store = zarr.open(output_path, mode='a')
    merged_store.create_dataset(dataset_keys[0], shape=(total_samples,) + shape1[1:], dtype=dtype1, chunks=chunks1)
    merged_store.create_dataset(dataset_keys[1], shape=(total_samples,) + shape2[1:], dtype=dtype2, chunks=chunks2)
    merged_store.create_dataset(dataset_keys[2], shape=(total_samples,) + shape2[1:], dtype=dtype2, chunks=chunks2)

    merged_store.store.cache_size = 0

    # Merge datasets chunk by chunk into the merged_store
    sample_offset = 0
    for zarr_file in zarr_files:
        zarr_file.store.cache_size = 0
        num_samples = zarr_file[dataset_keys[0]].shape[0]

        # Copy each chunk into the merged_store at the appropriate offset for both datasets
        for i in range(num_samples):
            merged_store[dataset_keys[0]][sample_offset + i] = zarr_file[dataset_keys[0]][i]
            merged_store[dataset_keys[1]][sample_offset + i] = zarr_file[dataset_keys[1]][i]
            merged_store[dataset_keys[2]][sample_offset + i] = zarr_file[dataset_keys[2]][i]

            gc.collect()

        sample_offset += num_samples

    return merged_store


def merge_zarr_datasets_train_val(train_dataset_path, val_dataset_path, zarr_paths, dataset_keys=['inputs', 'outputs', 'bulk_avg']):
    # Open the Zarr datasets and store them in a list
    zarr_files = [(zarr.open(zarr_path, mode='r', cache_attrs=False), n_train_val) for zarr_path, n_train_val in zarr_paths.items()]
    total_samples_train = sum([n_train_val["n_train"] for n_train_val in zarr_paths.values()])
    total_samples_val = sum([n_train_val["n_val"] for n_train_val in zarr_paths.values()])

    print("total samples train ", total_samples_train)
    print("total samples val ", total_samples_val)

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
    ### Train datasets
    #####
    # Create the merged Zarr store with the combined shape for both datasets
    trainset_store = zarr.open(train_dataset_path, mode='a')
    trainset_store.create_dataset(dataset_keys[0], shape=(total_samples_train,) + shape1[1:], dtype=dtype1, chunks=chunks1)
    trainset_store.create_dataset(dataset_keys[1], shape=(total_samples_train,) + shape2[1:], dtype=dtype2, chunks=chunks2)
    trainset_store.create_dataset(dataset_keys[2], shape=(total_samples_train,) + shape2[1:], dtype=dtype2, chunks=chunks2)

    trainset_store.store.cache_size = 0

    # Merge datasets chunk by chunk into the merged_store
    sample_offset = 0
    for zarr_file, n_train_val in zarr_files:
        zarr_file.store.cache_size = 0
        num_samples = zarr_file[dataset_keys[0]].shape[0]

        # Copy each chunk into the merged_store at the appropriate offset for both datasets
        for i in range(n_train_val["n_train"]):
            trainset_store[dataset_keys[0]][sample_offset + i] = zarr_file[dataset_keys[0]][i]
            trainset_store[dataset_keys[1]][sample_offset + i] = zarr_file[dataset_keys[1]][i]
            trainset_store[dataset_keys[2]][sample_offset + i] = zarr_file[dataset_keys[2]][i]

            gc.collect()

        sample_offset += n_train_val["n_train"]

    ###############
    ##  Validation data
    ##############
    # Create the merged Zarr store with the combined shape for both datasets
    valset_store = zarr.open(val_dataset_path, mode='a')
    valset_store.create_dataset(dataset_keys[0], shape=(total_samples_val,) + shape1[1:], dtype=dtype1, chunks=chunks1)
    valset_store.create_dataset(dataset_keys[1], shape=(total_samples_val,) + shape2[1:], dtype=dtype2, chunks=chunks2)
    valset_store.create_dataset(dataset_keys[2], shape=(total_samples_val,) + shape2[1:], dtype=dtype2, chunks=chunks2)

    valset_store.store.cache_size = 0

    # Merge datasets chunk by chunk into the merged_store
    sample_offset = 0
    for zarr_file, n_train_val in zarr_files:
        num_samples = zarr_file[dataset_keys[0]].shape[0]
        zarr_file.store.cache_size = 0

        # Copy each chunk into the merged_store at the appropriate offset for both datasets
        for i in range(n_train_val["n_val"]):
            valset_store[dataset_keys[0]][sample_offset + i] = zarr_file[dataset_keys[0]][num_samples - i - 1]
            valset_store[dataset_keys[1]][sample_offset + i] = zarr_file[dataset_keys[1]][num_samples - i - 1]
            valset_store[dataset_keys[2]][sample_offset + i] = zarr_file[dataset_keys[2]][num_samples - i - 1]

            gc.collect()

        sample_offset += n_train_val["n_val"]

    return trainset_store, valset_store


def shuffle_zarr_datasets(merged_store, output_path, dataset_keys=['inputs', 'outputs', 'bulk_avg']):
    # Get the total number of samples in the merged datasets
    total_samples = merged_store[dataset_keys[0]].shape[0]

    # Create a new Zarr store for the shuffled datasets
    shuffled_store = zarr.open(output_path, mode='a')
    shuffled_store.create_dataset(dataset_keys[0], shape=merged_store[dataset_keys[0]].shape,
                                  dtype=merged_store[dataset_keys[0]].dtype,
                                  chunks=merged_store[dataset_keys[0]].chunks)
    shuffled_store.create_dataset(dataset_keys[1], shape=merged_store[dataset_keys[1]].shape,
                                  dtype=merged_store[dataset_keys[1]].dtype,
                                  chunks=merged_store[dataset_keys[1]].chunks)
    shuffled_store.create_dataset(dataset_keys[2], shape=merged_store[dataset_keys[2]].shape,
                                  dtype=merged_store[dataset_keys[2]].dtype,
                                  chunks=merged_store[dataset_keys[2]].chunks)

    shuffled_store.store.cache_size = 0

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
zarr_paths = {"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_3/cond_nearest_interp/test_data/MLMC-DFM_3D_n_voxels_64_save_bulk_avg/samples_data.zarr": {"n_train": 1000, "n_val": 170},
              "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_3/cond_nearest_interp/test_data/MLMC-DFM_3D_n_voxels_64_save_bulk_avg_cl_25/samples_data.zarr": {"n_train": 900, "n_val": 230}
              }


trainset_store, valset_store = merge_zarr_datasets_train_val('train_dataset.zarr', 'val_dataset.zarr', zarr_paths, dataset_keys=['inputs', 'outputs', 'bulk_avg'])

shuffled_trainset_store = shuffle_zarr_datasets(trainset_store, 'shuffled_trainset.zarr', dataset_keys=['inputs', 'outputs', 'bulk_avg'])
shuffled_valset_store = shuffle_zarr_datasets(valset_store, 'shuffled_valset.zarr', dataset_keys=['inputs', 'outputs', 'bulk_avg'])


zarr_paths_list = ['shuffled_trainset.zarr',  'shuffled_valset.zarr']

merged_store = merge_zarr_datasets('merged_final_dset.zarr', zarr_paths_list, dataset_keys=['inputs', 'outputs', 'bulk_avg'])


original_zarr_file = zarr.open('merged_final_dset.zarr', mode='r+')

n_samples = 20
#
# # Cut datasets by slicing to the first 10,500 samples
# new_inputs = original_zarr_file['inputs'][:n_samples]
# new_outputs = original_zarr_file['outputs'][:n_samples]
# new_bulk_avg = original_zarr_file['bulk_avg'][:n_samples]
#
# # Create new datasets with the sliced data (overwrite the existing datasets)
# original_zarr_file['inputs'] = new_inputs  # Update the dataset
# original_zarr_file['outputs'] = new_outputs  # Update the dataset
# original_zarr_file['bulk_avg'] = new_bulk_avg  # Update the dataset


# # Open the original Zarr file in read mode
original_zarr_file = zarr.open('merged_final_dset.zarr', mode='a')

dataset_keys=['inputs', 'outputs', 'bulk_avg']


# Get shape and dtype from the first dataset (assuming all have the same structure)
shape1 = original_zarr_file[dataset_keys[0]].shape
shape2 = original_zarr_file[dataset_keys[1]].shape
dtype1 = original_zarr_file[dataset_keys[0]].dtype
dtype2 = original_zarr_file[dataset_keys[1]].dtype
chunks1 = original_zarr_file[dataset_keys[0]].chunks
chunks2 = original_zarr_file[dataset_keys[1]].chunks


#####
### Train datasets
#####
# Create the merged Zarr store with the combined shape for both datasets
trainset_store = zarr.open('cut_merged_final_dset.zarr', mode='a')
trainset_store.create_dataset(dataset_keys[0], shape=(n_samples,) + shape1[1:], dtype=dtype1, chunks=chunks1)
trainset_store.create_dataset(dataset_keys[1], shape=(n_samples,) + shape2[1:], dtype=dtype2, chunks=chunks2)
trainset_store.create_dataset(dataset_keys[2], shape=(n_samples,) + shape2[1:], dtype=dtype2, chunks=chunks2)

trainset_store.store.cache_size = 0

# Merge datasets chunk by chunk into the merged_store
sample_offset = 0

original_zarr_file.store.cache_size = 0
#num_samples = zarr_file[dataset_keys[0]].shape[0]

# Copy each chunk into the merged_store at the appropriate offset for both datasets
for i in range(n_samples):
    trainset_store[dataset_keys[0]][i] = original_zarr_file[dataset_keys[0]][i]
    trainset_store[dataset_keys[1]][i] = original_zarr_file[dataset_keys[1]][i]
    trainset_store[dataset_keys[2]][i] = original_zarr_file[dataset_keys[2]][i]


cut_zarr_file = zarr.open('cut_merged_final_dset.zarr', mode='r', cache_attrs=False)

print("cut_zarr_file[inputs].shape ", cut_zarr_file["inputs"].shape)
print("cut_zarr_file[outputs].shape ", cut_zarr_file["outputs"].shape)
print("cut_zarr_file[bulk_avg].shape ", cut_zarr_file["bulk_avg"].shape)


print("merged_store['inputs'].shape ", merged_store['inputs'].shape)


