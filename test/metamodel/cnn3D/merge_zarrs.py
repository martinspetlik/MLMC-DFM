import zarr
import numpy as np


def merge_zarr_datasets(output_path, zarr_paths, dataset_keys=['inputs', 'outputs']):
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

    # Merge datasets chunk by chunk into the merged_store
    sample_offset = 0
    for zarr_file in zarr_files:
        num_samples = zarr_file[dataset_keys[0]].shape[0]

        # Copy each chunk into the merged_store at the appropriate offset for both datasets
        for i in range(num_samples):
            merged_store[dataset_keys[0]][sample_offset + i] = zarr_file[dataset_keys[0]][i]
            merged_store[dataset_keys[1]][sample_offset + i] = zarr_file[dataset_keys[1]][i]

        sample_offset += num_samples

    return merged_store


def shuffle_zarr_datasets(merged_store, output_path, dataset_keys=['inputs', 'outputs']):
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

    # Generate a shuffled list of indices
    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    # Write the shuffled data to the new Zarr store for both datasets
    for i, idx in enumerate(indices):
        shuffled_store[dataset_keys[0]][i] = merged_store[dataset_keys[0]][idx]
        shuffled_store[dataset_keys[1]][i] = merged_store[dataset_keys[1]][idx]

    return shuffled_store


# Example usage: Provide paths to your datasets
zarr_paths = ["/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/MLMC-DFM_3D_n_voxels_64/samples_data.zarr",
              "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/MLMC-DFM_3D_n_voxels_64/samples_data.zarr"]# 'file3.zarr', 'file4.zarr']

merged_store = merge_zarr_datasets('merged.zarr', zarr_paths, dataset_keys=['inputs', 'outputs'])

shuffled_store = shuffle_zarr_datasets(merged_store, 'shuffled.zarr', dataset_keys=['inputs', 'outputs'])

