import os
import zarr
import numpy as np


# Example usage: Provide paths to your datasets
zarr_path = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_3/cond_nearest_interp/test_data/MLMC-DFM_3D_n_voxels_64_save_bulk_avg_cl_10/samples_data.zarr"

start_index = 0
end_index = 15000

outputs = []
zarr_file = zarr.open(zarr_path, mode='r')

outputs = zarr_file["outputs"][start_index:end_index]
bulk_avg = zarr_file["bulk_avg"][start_index:end_index]

np.savez_compressed("output_data", data=outputs)
np.savez_compressed("bulk_avg", data=bulk_avg)


current_dir = os.getcwd()

loaded_outputs = np.load(os.path.join(current_dir ,"output_data.npz"))["data"]
loaded_bulk_avg = np.load(os.path.join(current_dir , "bulk_avg.npz"))["data"]

print("loaded outpus shape ", loaded_outputs.shape)
print("loaded bulk avg shape ", loaded_bulk_avg.shape)




