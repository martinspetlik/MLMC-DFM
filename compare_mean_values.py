import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from mlmc.sample_storage_hdf import SampleStorageHDF
from mlmc.quantity.quantity import make_root_quantity
import h5py
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib import ticker


def load_hdf(filepath, level):
    with h5py.File(filepath, 'r') as f:
        ids = f["Levels"][level]['collected_ids'][:]
        values = f["Levels"][level]['collected_values'][:]

    ids = [id[0].decode() for id in ids]
    return ids, values


def plot_hist(samples_1, samples_2, x_label="value", title="", label=""):
    # from matplotlib import ticker
    # formatter = ticker.ScalarFormatter(useMathText=True)
    # formatter.set_scientific(True)
    figsize = (7, 4.5)
    fontsize = 17
    fontsize_ticks = 15

    # Style
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.2)
    palette = sns.color_palette("Set2", n_colors=6)

    fig, ax = plt.subplots(figsize=figsize)

    # Formatter
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(formatter)

    # Theme and colors
    #palette = sns.color_palette("Set2", n_colors=6)
    sns.set_theme(context="paper", style="white", palette="muted", font="serif")

    # Histograms on *same* axis
    sns.histplot(samples_1, bins=60, kde=False, stat="density",
                 color=palette[0], edgecolor="black", alpha=0.6,
                 label="approach (A)", ax=ax)

    sns.histplot(samples_2, bins=60, kde=False, stat="density",
                 color=palette[1], edgecolor="black", alpha=0.6,
                 label="approach (B)", ax=ax)

    # Dummy line for p-value in legend
    dummy = Line2D([0], [0], color="none", label=label)

    # Collect legend handles
    handles, labels = ax.get_legend_handles_labels()
    handles.append(dummy)
    labels.append(label)

    # Draw legend
    ax.legend(handles, labels, fontsize=fontsize)

    # Labels and ticks
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel("density", fontsize=fontsize)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    #ax.tick_params(axis='both', fontsize=fontsize)

    # Make x-ticks nice and readable
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune='both'))
    ax.tick_params(labelsize=fontsize_ticks)

    ax.yaxis.get_offset_text().set_fontsize(fontsize_ticks)
    ax.xaxis.get_offset_text().set_fontsize(fontsize_ticks)
    # ax.tick_params(axis='x', labelsize=fontsize_ticks)
    # ax.tick_params(axis='y', labelsize=fontsize_ticks)
    #plt.setp(ax.get_xticklabels(), rotation=rotation)

    plt.tight_layout()
    plt.savefig(title + ".pdf", dpi=300, bbox_inches="tight")
    plt.show()

    # # Create single figure and axis
    # fig, ax = plt.subplots(figsize=(8, 6))
    #
    # from matplotlib import ticker
    # formatter = ticker.ScalarFormatter(useMathText=True)
    # formatter.set_scientific(True)
    # formatter.set_powerlimits((-1, 1))
    # ax.yaxis.set_major_formatter(formatter)
    #
    # palette = sns.color_palette("Set2", n_colors=6)
    #
    # # palette=["blue", "red", "green"]
    # sns.set_theme(context="paper", style="white", palette="muted", font="serif")
    # #plt.figure(figsize=(8, 6))
    #
    # sns.histplot(samples_1, bins=100, kde=False, stat="density", color=palette[0],
    #              edgecolor="black", alpha=0.6, label="approach (A)")
    # sns.histplot(samples_2, bins=100, kde=False, stat="density", color=palette[1], edgecolor="black",
    #              alpha=0.6, label="approach (B)")
    #
    # plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter())
    # plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # plt.gca().ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    #
    # # Dummy artist for p-value legend entry
    # dummy = Line2D([0], [0], color="none", label=label)
    #
    # # Legend
    # handles, labels = ax.get_legend_handles_labels()
    # handles.append(dummy)
    # labels.append(label)
    # plt.legend(handles, labels, fontsize=12)
    #
    # # Axes labels
    # ax.set_xlabel(x_label, fontsize=14)
    # ax.set_ylabel("density", fontsize=14)
    # #ax.tick_params(axis='both', labelsize=14)
    #
    # ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    # ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    #
    # plt.tight_layout()
    # #plt.savefig(title + ".pdf")
    # plt.show()

    # # Add labels and title
    # plt.xlabel(x_label, fontsize=14)
    # plt.ylabel("density", fontsize=14)
    #
    # # Customize ticks
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.legend(fontsize=12)
    #
    # # Save the figure
    # plt.tight_layout()
    # plt.savefig(title+ ".pdf")
    #
    # # Show the plot
    # plt.show()


def get_statistics(samples_1, samples_2):
    # samples1 = np.random.randn(12500)  # placeholder
    # samples2 = np.random.randn(12500)  # placeholder

    # # ---------------------------------------------------------
    # # 1. Welch's two-sample t-test (default safe option)
    # # ---------------------------------------------------------
    # t_stat, p_value = stats.ttest_ind(samples_1, samples_2, equal_var=False)
    #
    # print("Welch's t-test:")
    # print(f"t-statistic = {t_stat:.6f}")
    # print(f"p-value     = {p_value:.6e}")
    #
    # # Interpretation:
    # # If p-value < 0.05 → significant difference in means.
    # # ---------------------------------------------------------
    #
    # # ---------------------------------------------------------
    # # 2. OPTIONAL: TOST equivalence test
    # # You must define an equivalence margin delta (practical tolerance)
    # # Example: means considered equivalent if |μ1 - μ2| < delta
    # # ---------------------------------------------------------
    # delta = 0.01  # change based on your domain
    #
    # # Lower and upper bounds for equivalence
    # low, high = -delta, delta
    #
    # # One-sided t-tests:
    # t1, p1 = stats.ttest_ind(samples_1, samples_2, equal_var=False, alternative='greater')
    # t2, p2 = stats.ttest_ind(samples_1, samples_2, equal_var=False, alternative='less')

    # Convert to TOST structure
    # We test: μ1 − μ2 > -delta  AND  μ1 − μ2 < delta
    # Interpretation: both p1 and p2 must be < alpha (e.g., 0.05)
    # print("\nTOST equivalence test:")
    # print(f"Lower test p-value = {p1:.6e}")
    # print(f"Upper test p-value = {p2:.6e}")
    # print("Conclusion: Means are equivalent if BOTH p-values < alpha (e.g., 0.05).")

    print("MEANS sample_1 : {}, sample_2: {}, diff: {}".format(np.mean(samples_1_orig), np.mean(samples_2_orig),
                                                               np.mean(samples_1_orig) - np.mean(samples_2_orig)))

    statistic, p_value = stats.ttest_ind(np.squeeze(samples_1), np.squeeze(samples_2), equal_var=False)
    # statistic, p_value = self.two_sample_ztest(np.squeeze(samples_a), np.squeeze(samples_b))
    # statistic, p_value = stats.chisquare(np.squeeze(samples_a), np.squeeze(samples_b))
    alpha = 0.05
    print("p value ", p_value)
    # Check if the p-value is less than alpha
    if p_value < alpha:
        print("Means are significantly different.")
    else:
        print("There is no significant difference between means")

    return p_value, alpha


def get_samples(hdf_file):
    sample_storage = SampleStorageHDF(file_path=hdf_file)
    sample_storage.chunk_size = 1e8
    root_quantity = make_root_quantity(storage=sample_storage,
                                       q_specs=sample_storage.load_result_format())

    q_idx = 0
    cond_tn_quantity = root_quantity["cond_tn"]
    time_mean = cond_tn_quantity[1]  # times: [1]
    location_mean = time_mean['0']  # locations: ['0']
    q_value = location_mean[q_idx]
    print("q_value ", q_value)

    chunk_spec = next(sample_storage.chunks(level_id=0, n_samples=sample_storage.get_n_collected()[0]))
    return np.squeeze(q_value.samples(chunk_spec=chunk_spec))


############
# H 7.5 h 5#
############
hdf_file_population = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_75_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5"
hdf_file_hom = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_75_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed_2/mlmc_2.hdf5"

############
# H 10 h 5 #
############
# hdf_file_population = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_10_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5"
# hdf_file_hom = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_10_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed_2/mlmc_2.hdf5"
# #
############
# H 15 h 5 #
############
#hdf_file_population = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_15_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5"
#hdf_file_hom = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_15_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed_2/mlmc_2.hdf5"

############
# H 20 h 5 #
############
hdf_file_population = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_20_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5"
hdf_file_hom = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_20_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed_2/mlmc_2.hdf5"

############
# H 40 h 5 #
############
#hdf_file_population = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_40_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5"
#hdf_file_hom = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_40_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed_2/mlmc_2.hdf5"


# hdf_file_population = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_20_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5"
# hdf_file_hom = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_20_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed_no_nn/mlmc_2.hdf5"

# hdf_file_population = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_20_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed_2/mlmc_2.hdf5"
# hdf_file_hom = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_20_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed_no_nn/mlmc_2.hdf5"


# hdf_file_population = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_40_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5"
# #hdf_file_hom = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_40_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed_2/mlmc_2.hdf5"
# hdf_file_hom = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_40_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed_2/mlmc_2.hdf5"


n_samples = 15000

samples_1 = np.log10(get_samples(hdf_file_population))[:15000]
samples_2 = np.log10(get_samples(hdf_file_hom))[:15000]


level = 0
collected_ids, collected_values = load_hdf(hdf_file_population, level="{}".format(level))
collected_ids_hom, collected_values_hom = load_hdf(hdf_file_hom, level="0")
print("collected ids ", collected_ids)
print("collected values ", collected_values.shape)
n_samples = 15000
idx = 0

print("collected_values[:, {}, idx].shape ", collected_values[:, level, idx].shape)

# --- Align by collected_id ---
df1 = pd.DataFrame({'id': collected_ids, 'samples_population': collected_values[:, level, idx]})
df2 = pd.DataFrame({'id': collected_ids_hom, 'samples_hom': collected_values_hom[:, 0, idx]})

if level == 0:
    # --- Merge them on 'id' to align by collected_ids ---
    merged = df1.merge(df2, on='id', how='inner')

    samples_1_orig = merged['samples_population'][:n_samples]
    samples_2_orig = merged['samples_hom'][:n_samples]

    samples_1 = np.log10(merged['samples_population'][:n_samples])
    samples_2 = np.log10(merged['samples_hom'][:n_samples])

else:
    samples_1_orig = df1['samples_population'][:n_samples]
    samples_2_orig = df2['samples_hom'][:n_samples]

    samples_1 = np.log10(df1['samples_population'][:n_samples])
    samples_2 = np.log10(df2['samples_hom'][:n_samples])


var_samples_1 = (samples_1_orig - np.mean(samples_1_orig))**2
var_samples_2 = (samples_2_orig - np.mean(samples_2_orig))**2


p_value, alpha = get_statistics(samples_1_orig, samples_2_orig)

if p_value < alpha:
    sig_text = "significant"
else:
    sig_text = "n.s."

plot_hist(samples_1, samples_2, x_label=r'$log_{10}(\phi_1)$', title="hist_samples_l0_mean", label="Welch t-test:\np = {:.3g}".format(p_value))

print("samples_1.shape ", samples_1.shape)
print("samples_2.shape ", samples_2.shape)

p_value, alpha = get_statistics(var_samples_1, var_samples_2)
plot_hist(np.log10(var_samples_1), np.log10(var_samples_2), x_label=r'$log_{10}(\phi_2)$',  title="hist_samples_l0_var", label="Welch t-test:\np = {:.3g}".format(p_value))
#get_statistics(np.log10(var_samples_1), np.log10(var_samples_2))
