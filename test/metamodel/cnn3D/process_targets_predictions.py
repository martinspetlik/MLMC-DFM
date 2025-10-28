import os
import numpy as np
import seaborn as sns
from metamodel.cnn.visualization.visualize_data import plot_target_prediction, plot_train_valid_loss
from metamodel.cnn.models.auxiliary_functions import exp_data, get_eigendecomp, get_mse_nrmse_r2, get_mean_std, log_data, exp_data,\
    quantile_transform_fit, QuantileTRF, NormalizeData, log_all_data, init_norm, get_mse_nrmse_r2_eigh_3D, log10_data, log10_all_data, power_10_all_data, power_10_data, CorrelatedOutputLoss
import matplotlib.pyplot as plt


def plot_target_prediction_more_channels(targets_arr, predictions_arr, title_prefix="log_orig_", r2=None, nrmse=None,
                       x_labels=None, titles=None):
    titles = ['k_xx', 'k_yy', 'k_zz', 'k_yz', 'k_xz', 'k_xy']

    x_labels = [r'$k_{xx}$', r'$k_{yy}$', r'$k_{zz}$', r'$k_{yz}$', r'$k_{xz}$', r'$k_{xy}$']
    x_labels_log = [r'$log(k_{xx})$', r'$log(k_{yy})$', r'$log(k_{zz})$', r'$k_{yz}$', r'$k_{xz}$', r'$k_{xy}$']

    # for i in range(data.shape[1]):
    #     plot_hist(data[:, i], xlabel=x_labels[i],
    #               title="Target_{}".format(titles[i]))

    # plot_hist(data[:, 1], xlabel="Values of pixel " + r'$k_{yy}$', ylabel="Frequency",
    #           title="Target_k_yy")
    # plot_hist(data[:, 2], xlabel="Values of pixel " + r'$k_{zz}$', ylabel="Frequency",
    #           title="Target_k_zz")
    # plot_hist(data[:, 3], xlabel="Values of pixel " + r'$k_{yz}$', ylabel="Frequency",
    #           title="Target_k_yz")
    # plot_hist(data[:, 4], xlabel="Values of pixel " + r'$k_{xz}$', ylabel="Frequency",
    #           title="Target_k_xz")
    # plot_hist(data[:, 5], xlabel="Values of pixel " + r'$k_{xy}$', ylabel="Frequency",
    #           title="Target_k_xy")

    import matplotlib


    from matplotlib import ticker
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    matplotlib.rcParams.update({'font.size': 26})

    fig, ax = plt.subplots(1, 1)

    # ax.set_yticks([0, 50, 100, 150])

    from matplotlib import ticker
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(formatter)

    # ax.hist(data, bins=60)
    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    #
    # plt.legend()
    # plt.savefig(title + ".pdf")
    # plt.show()

    # palette = sns.color_palette("dark:#5A9_r")
    # palette = sns.color_palette("tab10", n_colors=3)
    #palette = sns.color_palette("bright", n_colors=3)

    palette = sns.color_palette("Set2", n_colors=6)

    # palette=["blue", "red", "green"]

    k_target, k_predict = targets_arr[:, i], predictions_arr[:, i]

    print("predictions arr shape ", predictions_arr.shape)
    print("r2.shape ", r2)

    sns.set_theme(context="paper", style="white", palette="muted", font="serif")
    #plt.figure(figsize=(8, 6))
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=600)
    sns.scatterplot(x=targets_arr[:, 0], y=predictions_arr[:, 0], s=15, alpha=0.99, label=x_labels_log[0] + ", " + r'$R^2 = {:.5f}$'.format(float(r2[0])), color=palette[0], ax=ax)  # , edgecolors=(0, 0, 0))
    sns.scatterplot(x=targets_arr[:, 1], y=predictions_arr[:, 1], s=15, alpha=0.99, label=x_labels_log[1] + ", " + r'$R^2 = {:.5f}$'.format(float(r2[1])), color=palette[1], ax=ax)  # , edgecolors=(0, 0, 0))
    sns.scatterplot(x=targets_arr[:, 2], y=predictions_arr[:, 2], s=15, alpha=0.99, label=x_labels_log[2] + ", " + r'$R^2 = {:.5f}$'.format(float(r2[2])), color=palette[2], ax=ax)  # , edgecolors=(0, 0, 0))

    min_value = min(targets_arr[:, 0].min(), predictions_arr[:, 0].min(), targets_arr[:, 1].min(), predictions_arr[:, 1].min(), targets_arr[:, 2].min(),
                    predictions_arr[:, 2].min())
    max_value = max(targets_arr[:, 0].max(), predictions_arr[:, 0].max(), targets_arr[:, 1].max(), predictions_arr[:, 1].max(), targets_arr[:, 2].max(),
                    predictions_arr[:, 2].max())
    plt.plot([min_value, max_value], [min_value, max_value], color="black", linestyle="--")

    # Enforce scientific notation for x-ticks
    # Format the x-axis to use scientific notation
    plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter())
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.gca().ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    # Add labels and title
    # plt.xlabel(r'$k_{xx}$', fontsize=14)
    # plt.ylabel("density", fontsize=14)
    # # plt.title(fontsize=16, weight="bold")

    # # Customize ticks
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    #
    # plt.legend(fontsize=12)

    ax.xaxis.get_offset_text().set_fontsize(24)
    ax.yaxis.get_offset_text().set_fontsize(24)

    # Customize ticks
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    plt.legend(markerscale=3, fontsize=24)

    # Add grid for readability
    # plt.grid(True, which="major", linestyle="--", linewidth=0.5)
    plt.xlabel('Targets', fontsize=24)
    plt.ylabel('Predictions', fontsize=24)
    plt.gca().ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    # ax.ticklabel_format(style='sci')
    #plt.title(label)
    #plt.legend()
    plt.tight_layout()
    plt.savefig("log_orig_xx_yy_zz" + ".pdf")
    plt.show()


    #################################################
    #################################################
    #### OFF diagonal components              #######
    #################################################
    #################################################

    from matplotlib import ticker
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)

    fig, ax = plt.subplots(1, 1)

    # ax.set_yticks([0, 50, 100, 150])

    from matplotlib import ticker
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(formatter)

    # ax.hist(data, bins=60)
    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    #
    # plt.legend()
    # plt.savefig(title + ".pdf")
    # plt.show()

    # palette = sns.color_palette("dark:#5A9_r")
    # palette = sns.color_palette("tab10", n_colors=3)
    # palette = sns.color_palette("bright", n_colors=3)

    palette = sns.color_palette("Set2", n_colors=6)

    # palette=["blue", "red", "green"]

    k_target, k_predict = targets_arr[:, i], predictions_arr[:, i]

    print("predictions arr shape ", predictions_arr.shape)
    print("r2.shape ", r2)

    sns.set_theme(context="paper", style="white", palette="muted", font="serif")
    #plt.figure(figsize=(8, 6))
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=600)
    sns.scatterplot(x=targets_arr[:, 3], y=predictions_arr[:, 3], s=15, alpha=0.99,
                    label=x_labels_log[3] + ", " + r'$R^2 = {:.5f}$'.format(float(r2[3])),
                    color=palette[3], ax=ax)  # , edgecolors=(0, 0, 0))
    sns.scatterplot(x=targets_arr[:, 4], y=predictions_arr[:, 4], s=15, alpha=0.99,
                    label=x_labels_log[4] + ", " + r'$R^2 = {:.5f}$'.format(float(r2[4])),
                    color=palette[4], ax=ax)  # , edgecolors=(0, 0, 0))
    sns.scatterplot(x=targets_arr[:, 5], y=predictions_arr[:, 5], s=15, alpha=0.99,
                    label=x_labels_log[5] + ", " + r'$R^2 = {:.5f}$'.format(float(r2[5])),
                    color=palette[5], ax=ax)  # , edgecolors=(0, 0, 0))

    min_value = min(targets_arr[:, 3].min(), predictions_arr[:, 3].min(), targets_arr[:, 4].min(),
                    predictions_arr[:, 4].min(), targets_arr[:, 5].min(),
                    predictions_arr[:, 5].min())
    max_value = max(targets_arr[:, 3].max(), predictions_arr[:, 3].max(), targets_arr[:, 4].max(),
                    predictions_arr[:, 4].max(), targets_arr[:, 5].max(),
                    predictions_arr[:, 5].max())
    plt.plot([min_value, max_value], [min_value, max_value], color="black", linestyle="--")

    # Enforce scientific notation for x-ticks
    # Format the x-axis to use scientific notation
    plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter())
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.gca().ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    ax.xaxis.get_offset_text().set_fontsize(24)
    ax.yaxis.get_offset_text().set_fontsize(24)

    # Add labels and title
    # plt.xlabel(r'$k_{xx}$', fontsize=14)
    # plt.ylabel("density", fontsize=14)
    # # plt.title(fontsize=16, weight="bold")

    # Customize ticks
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    plt.legend(markerscale=3, fontsize=24)

    # Add grid for readability
    # plt.grid(True, which="major", linestyle="--", linewidth=0.5)
    plt.xlabel('Targets', fontsize=24)
    plt.ylabel('Predictions', fontsize=24)
    plt.gca().ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    # ax.ticklabel_format(style='sci')
    # plt.title(label)
    #plt.legend()
    plt.tight_layout()
    plt.savefig("log_orig_yz_xz_xy" + ".pdf")
    plt.show()


model_dir = "/home/martin/Documents/MLMC-DFM/optuna_runs/3D_cnn/lumi/cond_frac_1_3/cond_nearest_interp/dataset_2_save_bulk_avg_n_frac_2500_cl_0_10_25_seed_merge_new/init_norm/correlated_output_loss/exp_12_14/seed_12345"

base_dir_predictions = os.path.join(model_dir, "predictions")

data_dirs = ["MLMC-DFM_3D_n_voxels_64_save_bulk_avg",
             "MLMC-DFM_3D_n_voxels_64_save_bulk_avg_cl_10",
             "MLMC-DFM_3D_n_voxels_64_save_bulk_avg_cl_25",
             "MLMC-DFM_3D_n_voxels_64_save_bulk_avg_n_frac_2500",
             "MLMC-DFM_3D_n_voxels_64_save_bulk_avg_n_frac_2500_cl_10",
             "MLMC-DFM_3D_n_voxels_64_save_bulk_avg_n_frac_2500_cl_25"
]

all_predictions_list = []
all_targets_list = []

all_inv_predictions_arr = []
all_inv_targets_arr = []

training_data_cov_matrix = None

if os.path.exists(os.path.join(model_dir, "cov_mat.npy")):
    training_data_cov_matrix = np.load(os.path.join(model_dir, "cov_mat.npy"))

for data_dir in data_dirs:
    path_to_data_dir = os.path.join(base_dir_predictions, data_dir)

    predictions_list = np.load(os.path.join(path_to_data_dir, "predictions_list.npz"))["data"]
    targets_list = np.load(os.path.join(path_to_data_dir, "targets_list.npz"))["data"]

    all_predictions_list.extend(predictions_list)
    all_targets_list.extend(targets_list)

    inv_predictions_arr = np.load(os.path.join(path_to_data_dir, "inv_predictions_arr.npz"))["data"]
    inv_targets_arr = np.load(os.path.join(path_to_data_dir, "inv_targets_arr.npz"))["data"]

    all_inv_predictions_arr.extend(inv_predictions_arr)
    all_inv_targets_arr.extend(inv_targets_arr)

    covariance_matrix = np.load(os.path.join(path_to_data_dir, "covariance_matrix .npy"))

#targets_list = all_targets_list
#predictions_list = all_predictions_list




print("targets list ", targets_list[0])
print("predictions list ", predictions_list[0])

print("targets list shape ", np.array(targets_list).shape)


cov_from_targets = np.cov(np.array(targets_list).T)

print("cov from targets ", cov_from_targets)

if training_data_cov_matrix is not None:
    cov_diff_norm = np.linalg.norm(training_data_cov_matrix - cov_from_targets)

    print("cov_diff_norm ", cov_diff_norm)

    # Plot heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(np.abs(training_data_cov_matrix - cov_from_targets), annot=True, fmt=".2f", cmap="viridis", cbar=True)
    plt.title("Absolute Difference Heatmap")
    plt.xlabel("Features")
    plt.ylabel("Features")
    plt.savefig("cov_absolute_difference_heatmap.pdf")
    plt.show()

    plt.figure(figsize=(6, 5))
    sns.heatmap(np.abs(training_data_cov_matrix - cov_from_targets)/np.abs(training_data_cov_matrix), annot=True, fmt=".2f", cmap="viridis", cbar=True)
    plt.title("Covariance relative difference heatmap")
    plt.xlabel("Features")
    plt.ylabel("Features")
    plt.savefig("cov_relative_difference_heatmap.pdf")
    plt.show()

#print("inv_predictions_arr.shape ", inv_predictions_arr.shape)
#print("all_inv_predictions_arr.shape ", np.array(all_inv_predictions_arr).shape)


inv_predictions_arr = np.array(all_inv_predictions_arr)
inv_targets_arr = np.array(all_inv_targets_arr)

mse, rmse, nrmse, r2 = get_mse_nrmse_r2(targets_list, predictions_list)
inv_mse, inv_rmse, inv_nrmse, inv_r2 = get_mse_nrmse_r2(inv_targets_arr, inv_predictions_arr)

titles = ['k_xx', 'k_yy', 'k_zz', 'k_yz', 'k_xz', 'k_xy']
x_labels = [r'$log(k_{xx})$', r'$log(k_{yy})$', r'$log(k_{zz})$', r'$k_{yz}$', r'$k_{xz}$', r'$k_{xy}$']

mse_str, inv_mse_str = "MSE", "Original data MSE"
r2_str, inv_r2_str = "R2", "Original data R2"
rmse_str, inv_rmse_str = "RMSE", "Original data RMSE"
nrmse_str, inv_nrmse_str = "NRMSE", "Original data NRMSE"
for i in range(len(mse)):
    mse_str += " {}: {}".format(titles[i], mse[i])
    r2_str += " {}: {}".format(titles[i], r2[i])
    rmse_str += " {}: {}".format(titles[i], rmse[i])
    nrmse_str += " {}: {}".format(titles[i], nrmse[i])

    inv_mse_str += " {}: {}".format(titles[i], inv_mse[i])
    inv_r2_str += " {}: {}".format(titles[i], inv_r2[i])
    inv_rmse_str += " {}: {}".format(titles[i], inv_rmse[i])
    inv_nrmse_str += " {}: {}".format(titles[i], inv_nrmse[i])

    # print("MSE k_xx: {}, k_xy: {}, k_yy: {}".format(mse[0], mse[1], mse[2]))
    # print("R2 k_xx: {}, k_xy: {}, k_yy: {}".format(r2[0], r2[1], r2[2]))
    # print("RMSE  k_xx: {}, k_xy: {}, k_yy: {}".format(rmse[0], rmse[1], rmse[2]))
    # print("NRMSE  k_xx: {}, k_xy: {}, k_yy: {}".format(nrmse[0], nrmse[1], nrmse[2]))

    # print("Original data MSE k_xx: {}, k_xy: {}, k_yy: {}".format(inv_mse[0], inv_mse[1], inv_mse[2]))
    # print("Original data R2 k_xx: {}, k_xy: {}, k_yy: {}".format(inv_r2[0], inv_r2[1], inv_r2[2]))
    # print("Original data RMSE k_xx: {}, k_xy: {}, k_yy: {}".format(inv_rmse[0], inv_rmse[1], inv_rmse[2]))
    # print("Original data NRMSE  k_xx: {}, k_xy: {}, k_yy: {}".format(inv_nrmse[0], inv_nrmse[1], inv_nrmse[2]))


print(mse_str)
print(r2_str)
print(rmse_str)
print(nrmse_str)


print("mean R2: {}, NRMSE: {}".format(np.mean(r2), np.mean(nrmse)))

print(inv_mse_str)
print(inv_r2_str)
print(inv_rmse_str)
print(inv_nrmse_str)

print("Original data mean R2: {}, NRMSE: {}".format(np.mean(inv_r2), np.mean(inv_nrmse)))

import copy
log_inv_targets_arr = copy.deepcopy(inv_targets_arr)
log_inv_predictions_arr = copy.deepcopy(inv_predictions_arr)

print("log_inv_targets_arr.shape ", log_inv_targets_arr.shape)

log_inv_targets_arr[:, 0] = np.log10(log_inv_targets_arr[:, 0])
log_inv_targets_arr[:, 1] = np.log10(log_inv_targets_arr[:, 1])
log_inv_targets_arr[:, 2] = np.log10(log_inv_targets_arr[:, 2])

log_inv_predictions_arr[:, 0] = np.log10(log_inv_predictions_arr[:, 0])
log_inv_predictions_arr[:, 1] = np.log10(log_inv_predictions_arr[:, 1])
log_inv_predictions_arr[:, 2] = np.log10(log_inv_predictions_arr[:, 2])

log_inv_mse, log_inv_rmse, log_inv_nrmse, log_inv_r2 = get_mse_nrmse_r2(log_inv_targets_arr, log_inv_predictions_arr)

mse_str, inv_mse_str = "MSE", "LOG Original data MSE"
r2_str, inv_r2_str = "R2", "LOG Original data R2"
rmse_str, inv_rmse_str = "RMSE", "LOG Original data RMSE"
nrmse_str, inv_nrmse_str = "NRMSE", "LOG Original data NRMSE"
for i in range(len(mse)):
    inv_mse_str += " {}: {}".format(titles[i], log_inv_mse[i])
    inv_r2_str += " {}: {}".format(titles[i], log_inv_r2[i])
    inv_rmse_str += " {}: {}".format(titles[i], log_inv_rmse[i])
    inv_nrmse_str += " {}: {}".format(titles[i], log_inv_nrmse[i])

print(inv_mse_str)
print(inv_r2_str)
print(inv_rmse_str)
print(inv_nrmse_str)

get_mse_nrmse_r2_eigh_3D(targets_list, predictions_list)
print("ORIGINAL DATA")
get_mse_nrmse_r2_eigh_3D(inv_targets_arr, inv_predictions_arr)


print("log_inv_r2 ", log_inv_r2)
print("log_inv_nrmse ", log_inv_nrmse)

print("mean log_inv_r2 ", np.mean(log_inv_r2))
print("mean log_inv_nrmse ", np.mean(log_inv_nrmse))


#plot_target_prediction(np.array(targets_list), np.array(predictions_list), "preprocessed_", x_labels=x_labels, titles=titles)
#plot_target_prediction(inv_targets_arr, inv_predictions_arr, x_labels=x_labels, titles=titles)

np.save("inv_tragets_arr_fr_div_10", inv_targets_arr)
np.save("inv_predictions_arr_fr_div_10", inv_predictions_arr)

# inv_targets_arr[:, 0] = np.log10(inv_targets_arr[:, 0])
# inv_targets_arr[:, 2] = np.log10(inv_targets_arr[:, 2])
#
# inv_predictions_arr[:, 0] = np.log10(inv_predictions_arr[:, 0])
# inv_predictions_arr[:, 2] = np.log10(inv_predictions_arr[:, 2])

# wrong_targets = np.array(wrong_targets)
# wrong_predictions = np.array(wrong_predictions)
# wrong_targets[:, 0] = np.log10(wrong_targets[:, 0])
# wrong_targets[:, 2] = np.log10(wrong_targets[:, 2])
#
# wrong_predictions[:, 0] = np.log10(wrong_predictions[:, 0])
# wrong_predictions[:, 2] = np.log10(wrong_predictions[:, 2])


# plot_target_prediction(log_inv_targets_arr, log_inv_predictions_arr, title_prefix="log_orig_", r2=log_inv_r2, nrmse=log_inv_nrmse,
#                        x_labels=x_labels, titles=titles)

plot_target_prediction_more_channels(log_inv_targets_arr, log_inv_predictions_arr, title_prefix="log_orig_", r2=log_inv_r2, nrmse=log_inv_nrmse,
                       x_labels=x_labels, titles=titles)

# plot_target_prediction(wrong_targets, wrong_predictions, title_prefix="wrong_log_orig_", r2=log_inv_r2,
#                        nrmse=log_inv_nrmse,
#                        x_labels=[r'$log(k_{xx})$', r'$k_{xy}$', r'$log(k_{yy})$'])

######
## main peak fr div 0.1
######
#print("inv_targets_arr[inv_targets_arr[:, 0] > -3.8]", inv_targets_arr[inv_targets_arr[:, 0] > -3.8])
#print(" inv_predictions_arr[inv_targets_arr[:, 0] > -3.8]",  inv_predictions_arr[inv_targets_arr[:, 0] > -3.8])


# log_inv_mse, log_inv_rmse, log_inv_nrmse, log_inv_r2 = get_mse_nrmse_r2(inv_targets_arr[inv_targets_arr[:, 0] > -3.8],
#                                                                         inv_predictions_arr[inv_targets_arr[:, 0] > -3.8])

log_inv_mse, log_inv_rmse, log_inv_nrmse, log_inv_r2 = get_mse_nrmse_r2(log_inv_targets_arr, log_inv_predictions_arr)

print("log_inv_r2 main peak", log_inv_r2)
print("log_inv_nrmse main peak", log_inv_nrmse)




