import os
import shutil
import zarr
import numpy as np

zarr_dir = "/storage/liberec3-tul/home/martin_spetlik/MLMC-DFM_3D_experiments/homogenization_samples/datasets/cond_frac_1_3/dataset_1/"

zarr_name = "samples_data.zarr"
zarr_name_reduced = "reduced_samples_data.zarr"


# Open your Zarr array (assuming it's stored in a file)
zarr_file = zarr.open(os.path.join(zarr_dir, zarr_name), mode='r')

inputs = zarr_file['inputs']
outputs = zarr_file['outputs']


N = inputs.shape[0]  # Number of samples (size of the first dimension)
mask = np.zeros(N, dtype=bool)  # Initialize the mask with False

if N > 100:
    # Iterate over each sample
    for i in range(N):
        # Check if any element in the flattened sub-array is 0
        if np.any(inputs[i].flatten() == 0):
            mask[i] = True

    output_mask = np.any(outputs[:] == 0, axis=1)

else:
    mask = np.any(inputs[:] == 0, axis=(1, 2, 3, 4))
    output_mask = np.any(outputs[:] == 0, axis=1)


mask = ~mask
output_mask = ~output_mask

#print("outputs[65] ", outputs[65])
#print("outputs ", outputs)

rows_with_zeros = np.where(mask)[0]
rows_with_zeros_output = np.where(output_mask)[0]

# print("mask ", mask)
# print("output mask ", output_mask)
# print("with zeros ", rows_with_zeros)
# print("with zeros output ", rows_with_zeros_output)


# Determine the number of rows to keep
n_samples = np.min([np.sum(mask), np.sum(output_mask)])


reduced_zarr_file_path = os.path.join(zarr_dir, zarr_name_reduced)
#if not os.path.exists(reduced_zarr_file_path):
zarr_file = zarr.open(reduced_zarr_file_path, mode='w')

# # Create the 'inputs' dataset with the specified shape
new_inputs = zarr_file.create_dataset('inputs',
                                  shape=(n_samples, *inputs.shape[1:]),
                                  dtype='float32',
                                  chunks=(1,  *inputs.shape[1:]))
new_inputs[:, :, :, :, :] = np.zeros((n_samples,  *inputs.shape[1:]))  # Populate the first 6 channels
# inputs[:, :, :, :, n_cond_tn_channels] = np.random.rand(n_samples,
#                                                         *input_shape_n_voxels)  # Populate the last channel

# Create the 'outputs' dataset with the specified shape
new_outputs = zarr_file.create_dataset('outputs', shape=(n_samples, outputs.shape[1]), dtype='float32',
                                   chunks=(1, outputs.shape[1]))
new_outputs[:, :] = np.zeros((n_samples,  outputs.shape[1]))  # Populate the first 6 channels
# outputs[:, n_cond_tn_channels] = np.random.rand(n_samples)  # Populate the last channel

# Assign metadata to indicate channel names
new_inputs.attrs['channel_names'] = ['cond_tn_0', 'cond_tn_1', 'cond_tn_2', 'cond_tn_3', 'cond_tn_4', 'cond_tn_5']
new_outputs.attrs['channel_names'] = ['cond_tn_0', 'cond_tn_1', 'cond_tn_2', 'cond_tn_3', 'cond_tn_4', 'cond_tn_5']


# Apply the mask manually
filtered_index = 0
for i, (orig_input, orig_output) in enumerate(zip(inputs, outputs)):
    if mask[i] and output_mask[i]:  # If the mask is True for this row
        new_inputs[filtered_index] = inputs[i]
        new_outputs[filtered_index] = outputs[i]
        filtered_index += 1

if os.path.exists(os.path.join(zarr_dir, "orig_samples_data.zarr")):
    shutil.rmtree(os.path.join(zarr_dir, "orig_samples_data.zarr"))

shutil.move(os.path.join(zarr_dir, zarr_name), os.path.join(zarr_dir, "orig_samples_data.zarr"))
shutil.move(reduced_zarr_file_path, os.path.join(zarr_dir, zarr_name))
