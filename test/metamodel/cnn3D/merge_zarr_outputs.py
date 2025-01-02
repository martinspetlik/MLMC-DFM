import zarr
import numpy as np
import matplotlib.pyplot as plt

def merge_zarr_outputs(zarr_paths, dataset_keys=['inputs', 'outputs', 'bulk_avg']):
    # Open the Zarr datasets and store them in a list

    outputs = []
    zarr_files = [(zarr.open(zarr_path, mode='r'), n_train_val) for zarr_path, n_train_val in zarr_paths.items()]

    # Merge datasets chunk by chunk into the merged_store
    sample_offset = 0
    for zarr_file, n_samples in zarr_files:

        outputs.extend(list(zarr_file["outputs"][:n_samples["n_samples"]]))


    outputs = np.array(outputs)

    print("outputs shape ", outputs.shape)

    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(outputs[:, 0], bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='black')
    plt.title("Histogram of Data")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.hist(np.log10(outputs[:, 0]), bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='black')
    plt.title("Histogram of log Data")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.show()


# Example usage: Provide paths to your datasets
zarr_paths = {"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_5/cond_nearest_interp/test_data/MLMC-DFM_3D_n_voxels_64_save_bulk_avg_n_frac_2500/samples_data.zarr": {"n_samples": 2500},
"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_5/cond_nearest_interp/test_data/MLMC-DFM_3D_n_voxels_64_save_bulk_avg/samples_data.zarr": {"n_samples" : 2500}
              #"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_5/cond_nearest_interp/test_data/MLMC-DFM_3D_n_voxels_64_500_frac/samples_data.zarr": {"n_samples" : 625}
              }

merge_zarr_outputs(zarr_paths, dataset_keys=['inputs', 'outputs', 'bulk_avg'])



