import os
import shutil
import zarr
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Process an optional work directory argument.")

# Adding the optional argument `work_dir` with default None
parser.add_argument(
    "--work_dir",
    type=str,
    default=None,  # Default to None
    help="Path to the working directory (default: None)."
)

# Parse the arguments
args = parser.parse_args()

# Print the processed work_dir
if args.work_dir is not None:
    zarr_dir = args.work_dir

else:
    zarr_dir = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_3/cond_nearest_interp/test_data/" #"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/MLMC-DFM_3D_n_voxels_64"
    zarr_dir = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_5/cond_nearest_interp/test_data/"
    zarr_dir = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_3/cond_nearest_interp/test_data/MLMC-DFM_3D_n_voxels_64_n/"
    zarr_dir = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_5/cond_nearest_interp/test_data/MLMC-DFM_3D_n_voxels_64_500_frac"
    zarr_dir = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_3/cond_nearest_interp/dataset_2_other_method"

    zarr_dir = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_5/cond_nearest_interp/test_data/MLMC-DFM_3D_n_voxels_64_save_bulk_avg/"

    zarr_dir = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_7/cond_nearest_interp/MLMC-DFM_3D_n_voxels_64"
    zarr_dir = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_3/cond_nearest_interp/test_data/MLMC-DFM_3D_n_voxels_64_save_bulk_avg_cl_25"

    zarr_dir = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_5/cond_nearest_interp/test_data/MLMC-DFM_3D_n_voxels_64_save_bulk_avg_n_frac_3000"


#zarr_dir = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_5/cond_nearest_interp/test_data/MLMC-DFM_3D_n_voxels_64_save_bulk_avg_disp_0"


#zarr_dir = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_5/cond_nearest_interp/test_data/MLMC-DFM_3D_n_voxels_64_save_bulk_avg_n_frac_2500"

#zarr_dir = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_6/cond_nearest_interp/test_data/MLMC-DFM_3D_n_voxels_64_save_bulk_avg"

#zarr_dir = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_5/cond_nearest_interp/test_data/MLMC-DFM_3D_n_voxels_64_hom_samples/"

print("zarr dir ", zarr_dir)

zarr_name = "samples_data.zarr"
#zarr_name = "merged.zarr"
#zarr_name = "sub_dataset_2_2500.zarr"
zarr_name_reduced = "reduced_samples_data.zarr"

zarr_name_removed_zeros = "removed_zeros_samples_data.zarr"
print("os.path.join(zarr_dir, zarr_name) ", os.path.join(zarr_dir, zarr_name))
# Open your Zarr array (assuming it's stored in a file)
zarr_file = zarr.open(os.path.join(zarr_dir, zarr_name), mode='r')


inputs = zarr_file['inputs']
outputs = zarr_file['outputs']
bulk_avg = zarr_file['bulk_avg']


N = inputs.shape[0]  # Number of samples (size of the first dimension)
mask = np.zeros(N, dtype=bool)  # Initialize the mask with False
output_mask = np.zeros(N, dtype=bool)

print("N ", N)


####################
##  Remove zeros  ##
####################
all_outputs = []
indices = []
if N > 100:
    # Iterate over each sample
    for i in range(N):
        # print("inputs[i] ", inputs[i].shape)
        # print("type(inputs[i])", type(inputs[i]))
        # print("type(inputs[i][0][0][0][0])", type(inputs[i][0][0][0][0]))
        if np.any(inputs[i][[0, 1, 2]] < 0):
            mask[i] = True
        if np.any(outputs[i][[0, 1, 2]] < 0):
            output_mask[i] = True

        # Check if any element in the flattened sub-array is 0
        if np.any(inputs[i].flatten() == 0):
            mask[i] = True
        if np.any(outputs[i].flatten() == 0):
            output_mask[i] = True
        # else:
        #     print("outputs[i] ", outputs[i])
        #     all_outputs.append(outputs[i])
        #     indices.append(i)
    #output_mask = np.any(outputs[:] == 0, axis=1)
    #print("output_mask ", output_mask)
else:
    mask = np.any(inputs[:] == 0, axis=(1, 2, 3, 4))
    output_mask = np.any(outputs[:] == 0, axis=1)

mask = ~mask
output_mask = ~output_mask

n_samples = np.min([np.sum(mask), np.sum(output_mask)])
print("zeros mask n samples ", n_samples)
reduced_zarr_file_path = os.path.join(zarr_dir, zarr_name_removed_zeros)


zarr_file = zarr.open(reduced_zarr_file_path, mode='w')

# # Create the 'inputs' dataset with the specified shape
new_inputs = zarr_file.create_dataset('inputs',
                                  shape=(n_samples, *inputs.shape[1:]),
                                  dtype='float32',
                                  chunks=(1,  *inputs.shape[1:]))
#new_inputs[:, :, :, :, :] = np.zeros((n_samples,  *inputs.shape[1:]))  # Populate the first 6 channels
# inputs[:, :, :, :, n_cond_tn_channels] = np.random.rand(n_samples,
#                                                         *input_shape_n_voxels)  # Populate the last channel

# Create the 'outputs' dataset with the specified shape
new_outputs = zarr_file.create_dataset('outputs', shape=(n_samples, outputs.shape[1]), dtype='float32',
                                   chunks=(1, outputs.shape[1]))

new_bulk_avg = zarr_file.create_dataset('bulk_avg', shape=(n_samples, outputs.shape[1]), dtype='float32',
                                               chunks=(1, outputs.shape[1]), fill_value=0)

#new_outputs[:, :] = np.zeros((n_samples,  outputs.shape[1]))  # Populate the first 6 channels
# outputs[:, n_cond_tn_channels] = np.random.rand(n_samples)  # Populate the last channel

# Assign metadata to indicate channel names
new_inputs.attrs['channel_names'] = ['cond_tn_0', 'cond_tn_1', 'cond_tn_2', 'cond_tn_3', 'cond_tn_4', 'cond_tn_5']
new_outputs.attrs['channel_names'] = ['cond_tn_0', 'cond_tn_1', 'cond_tn_2', 'cond_tn_3', 'cond_tn_4', 'cond_tn_5']
new_bulk_avg.attrs['channel_names'] = ['cond_tn_0', 'cond_tn_1', 'cond_tn_2', 'cond_tn_3', 'cond_tn_4', 'cond_tn_5']

# Apply the mask manually
filtered_index = 0
for i, (orig_input, orig_output) in enumerate(zip(inputs, outputs)):
    if mask[i] and output_mask[i]:  # If the mask is True for this row
        new_inputs[filtered_index] = inputs[i]
        new_outputs[filtered_index] = outputs[i]
        new_bulk_avg[filtered_index] = bulk_avg[i]
        filtered_index += 1


#######################
### Remove outliers ###
#######################
zarr_file = zarr.open(os.path.join(zarr_dir, zarr_name_removed_zeros), mode='r')

inputs = zarr_file['inputs']
outputs = zarr_file['outputs']
bulk_avg = zarr_file['bulk_avg']

iqr_multiplier = 1.5

#inputs = inputs[indices]
all_outputs = outputs
all_outputs = np.array(all_outputs)


all_indices = []
for i_ch in range(all_outputs[0].shape[0]):
    t_data = all_outputs[:, i_ch]
    q1 = np.percentile(t_data, 10)
    q3 = np.percentile(t_data, 90)
    iqr = q3 - q1
    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr
    if i_ch < 3:
        indices = (t_data >= lower_bound) & (t_data <= upper_bound) | (t_data < 0)
    else:
        indices = (t_data >= lower_bound) & (t_data <= upper_bound)
    all_indices.append(indices)

outliers_mask = np.logical_and.reduce(all_indices)


#######################
### init norm mask  ###
#######################
bulk_avg_per_sample = np.mean(bulk_avg, axis=1)

all_outputs_init_norm = []


mask_init_norm = np.zeros(all_outputs.shape[0], dtype=bool)
output_mask_init_norm = np.zeros(all_outputs.shape[0], dtype=bool)
for i in range(all_outputs.shape[0]):
    inputs_init_norm = inputs[i] / bulk_avg_per_sample[i]

    all_outputs_init_norm.append(outputs[i] / bulk_avg_per_sample[i])
    outputs_init_norm = outputs[i] / bulk_avg_per_sample[i]

    if np.any(inputs_init_norm[[0,1,2]] < 0):
        mask_init_norm[i] = True
    if np.any(outputs_init_norm[[0,1,2]] < 0):
        output_mask_init_norm[i] = True

    # Check if any element in the flattened sub-array is 0
    if np.any(inputs_init_norm.flatten() == 0) or np.any(np.isnan(inputs_init_norm.flatten())):
        mask_init_norm[i] = True
    if np.any(outputs_init_norm.flatten() == 0) or np.any(np.isnan(outputs_init_norm.flatten())):
        output_mask_init_norm[i] = True


mask_init_norm = ~mask_init_norm
output_mask_init_norm = ~output_mask_init_norm
all_outputs_init_norm = np.array(all_outputs_init_norm)


all_indices = []
for i_ch in range(all_outputs_init_norm[0].shape[0]):
    t_data = all_outputs_init_norm[:, i_ch]
    q1 = np.percentile(t_data, 10)
    q3 = np.percentile(t_data, 90)
    iqr = q3 - q1
    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr
    if i_ch < 3:
        indices = (t_data >= lower_bound) & (t_data <= upper_bound) | (t_data < 0)
    else:
        indices = (t_data >= lower_bound) & (t_data <= upper_bound)
    all_indices.append(indices)

outliers_mask_init_norm = np.logical_and.reduce(all_indices)
n_samples = np.min([np.sum(mask), np.sum(output_mask), np.sum(outliers_mask), np.sum(outliers_mask_init_norm)])

print("init norm mask n_samples ", n_samples)


#exit()

#print("outputs[65] ", outputs[65])
#print("outputs ", outputs)

# rows_with_zeros = np.where(mask)[0]
# rows_with_zeros_output = np.where(output_mask)[0]

# print("mask ", mask)
# print("output mask ", output_mask)
# print("with zeros ", rows_with_zeros)
# print("with zeros output ", rows_with_zeros_output)


# Determine the number of rows to keep
#n_samples = np.min([np.sum(outliers_mask)])


print("n samples ", n_samples)

reduced_zarr_file_path = os.path.join(zarr_dir, zarr_name_reduced)
#if not os.path.exists(reduced_zarr_file_path):
zarr_file = zarr.open(reduced_zarr_file_path, mode='w')

# # Create the 'inputs' dataset with the specified shape
new_inputs = zarr_file.create_dataset('inputs',
                                  shape=(n_samples, *inputs.shape[1:]),
                                  dtype='float32',
                                  chunks=(1,  *inputs.shape[1:]))
#new_inputs[:, :, :, :, :] = np.zeros((n_samples,  *inputs.shape[1:]))  # Populate the first 6 channels
# inputs[:, :, :, :, n_cond_tn_channels] = np.random.rand(n_samples,
#                                                         *input_shape_n_voxels)  # Populate the last channel

# Create the 'outputs' dataset with the specified shape
new_outputs = zarr_file.create_dataset('outputs', shape=(n_samples, outputs.shape[1]), dtype='float32',
                                   chunks=(1, outputs.shape[1]))
new_bulk_avg = zarr_file.create_dataset('bulk_avg', shape=(n_samples, outputs.shape[1]), dtype='float32',
                                               chunks=(1, outputs.shape[1]), fill_value=0)
#new_outputs[:, :] = np.zeros((n_samples,  outputs.shape[1]))  # Populate the first 6 channels
# outputs[:, n_cond_tn_channels] = np.random.rand(n_samples)  # Populate the last channel

# Assign metadata to indicate channel names
new_inputs.attrs['channel_names'] = ['cond_tn_0', 'cond_tn_1', 'cond_tn_2', 'cond_tn_3', 'cond_tn_4', 'cond_tn_5']
new_outputs.attrs['channel_names'] = ['cond_tn_0', 'cond_tn_1', 'cond_tn_2', 'cond_tn_3', 'cond_tn_4', 'cond_tn_5']
new_bulk_avg.attrs['channel_names'] = ['cond_tn_0', 'cond_tn_1', 'cond_tn_2', 'cond_tn_3', 'cond_tn_4', 'cond_tn_5']


# Apply the mask manually
filtered_index = 0
for i, (orig_input, orig_output) in enumerate(zip(inputs, outputs)):
    if outliers_mask[i] and outliers_mask_init_norm[i] and mask_init_norm[i] and output_mask_init_norm[i]:  # If the mask is True for this row
        new_inputs[filtered_index] = inputs[i]
        new_outputs[filtered_index] = outputs[i]
        new_bulk_avg[filtered_index] = bulk_avg[i]
        filtered_index += 1


#filtered_index -= 4

print("filtered index ", filtered_index)

new_inputs.resize((filtered_index,) + new_inputs.shape[1:])
new_outputs.resize((filtered_index,) + new_outputs.shape[1:])
new_bulk_avg.resize((filtered_index,) + new_bulk_avg.shape[1:])

print("new_inputs.shape ", new_inputs.shape)
print("new_outputs.shape ", new_outputs.shape)
print("new_bulk_avg.shape ", new_bulk_avg.shape)

if os.path.exists(os.path.join(zarr_dir, "orig_samples_data.zarr")):
    shutil.rmtree(os.path.join(zarr_dir, "orig_samples_data.zarr"))

shutil.move(os.path.join(zarr_dir, zarr_name), os.path.join(zarr_dir, "orig_samples_data.zarr"))
shutil.move(reduced_zarr_file_path, os.path.join(zarr_dir, zarr_name))


if os.path.exists(os.path.join(zarr_dir, "orig_samples_data.zarr")):
    shutil.rmtree(os.path.join(zarr_dir, "orig_samples_data.zarr"))

if os.path.exists(os.path.join(zarr_dir, zarr_name_removed_zeros)):
    shutil.rmtree(os.path.join(zarr_dir, zarr_name_removed_zeros))