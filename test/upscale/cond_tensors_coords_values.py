import numpy as np
import glob
import os
from bgem.upscale import tn_to_voigt

# Set your directory here
data_dir = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_10_30_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population/l_step_10_common_files"
data_dir = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/4LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_10_20_40_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population/l_step_40_common_files"

data_dir = "/home/martin/Documents/MLMC-DFM/test/01_cond_field/l_step_10.0_common_files"

fine_samples_dir = "/home/martin/Documents/MLMC-DFM/test/01_cond_field/l_step_10.0_common_files/fine_samples"

# Load all coords files (.npz)
coords_files = sorted(glob.glob(os.path.join(data_dir, "cond_tensors_coords_*.npz")))
for f in coords_files:
    coords_data = np.load(f)["data"]
    print("coords data ", coords_data)

# Each coords .npz may have multiple arrays inside
#coords_arrays = [d[d.files[0]] for d in coords_data]

# Load all values files (.npy or .npz)
values_files = sorted(glob.glob(os.path.join(data_dir, "cond_tensors_values_*")))

pos = 0

all_cond_tn_values = []
for f in values_files:
    cond_tn_values = tn_to_voigt(np.load(f)["data"])
    print("cond tn values shape ", cond_tn_values.shape)
    all_cond_tn_values.append(cond_tn_values[pos])




all_cond_tn_values = np.array(all_cond_tn_values)
print("all cond tn values ", all_cond_tn_values.shape)

#for i in range()



values_data = [np.load(f) for f in values_files]

# print("Coords files found:", coords_files)
# print("Values files found:", values_files)
# print("Coords array shapes:", [a.shape for a in coords_arrays])
# print("Values array shapes:", [v.shape for v in values_data])



