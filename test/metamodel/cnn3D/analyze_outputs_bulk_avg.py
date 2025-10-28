import os
import zarr
import numpy as np
import matplotlib.pyplot as plt
from test_dataset import plot_hist
import seaborn as sns
from matplotlib.ticker import ScalarFormatter, MaxNLocator


def plot_hist_output(data):
    print("data.shape ", data.shape)

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
    palette = sns.color_palette("Set2", n_colors=6)

    # palette=["blue", "red", "green"]
    sns.set_theme(context="paper", style="white", palette="muted", font="serif")
    plt.figure(figsize=(8, 6))
    # sns.histplot(data[:, 3], bins=60, kde=False, stat="density", color=palette[0],
    #              edgecolor="black", alpha=0.4, label=r'$k_{yz}$')
    # sns.histplot(data[:, 4], bins=60, kde=False, stat="density", color=palette[1], edgecolor="black",
    #              alpha=0.4, label=r'$k_{xz}$')
    # sns.histplot(data[:, 5],
    #              bins=60, kde=False, stat="density", color=palette[2], edgecolor="black", alpha=0.4, label=r'$k_{xy}$')

    sns.histplot(np.log10(data[:, 0]), bins=60, kde=False, stat="density", color=palette[0],
                 edgecolor="black", alpha=0.4, label=r'$log(k_{xx})$')
    sns.histplot(np.log10(data[:, 1]), bins=60, kde=False, stat="density", color=palette[1], edgecolor="black",
                 alpha=0.4, label=r'$log(k_{yy})$')
    sns.histplot(np.log10(data[:, 2]),
                 bins=60, kde=False, stat="density", color=palette[2], edgecolor="black", alpha=0.4,
                 label=r'$log(k_{zz})$')

    # sns.histplot(np.log10(data[:, 0]), bins=60, kde=False, stat="density", color=palette[0],
    #              edgecolor="black", alpha=0.4, label=r'$log(k_{xx})$')
    # sns.histplot(np.log10(data[:, 1]), bins=60, kde=False, stat="density", color=palette[1], edgecolor="black",
    #              alpha=0.4, label=r'$log(k_{yy})$')
    # sns.histplot(np.log10(data[:, 2]),
    #              bins=60, kde=False, stat="density", color=palette[2], edgecolor="black", alpha=0.4,
    #              label=r'$log(k_{zz})$')

    # sns.histplot(np.log10(data[:, 0]), bins=60, kde=False, stat="density", color="dodgerblue", edgecolor="black", alpha=0.3)
    # sns.histplot(np.log10(data[:, 1]), bins=60, kde=False, stat="density", color="salmon", edgecolor="black", alpha=0.3)
    # sns.histplot(np.log10(data[:, 2]), bins = 60, kde = False, stat = "density", color = "limegreen", edgecolor = "black", alpha = 0.3)
    # plt.hist(data, bins=30, density=True, color="dodgerblue", edgecolor="black", alpha=0.7)

    # Enforce scientific notation for x-ticks
    # Format the x-axis to use scientific notation
    plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter())
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.gca().ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    # Add labels and title
    plt.xlabel("value", fontsize=14)
    plt.ylabel("density", fontsize=14)
    # plt.title(fontsize=16, weight="bold")

    # Customize ticks
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.legend(fontsize=12)

    # Add grid for readability
    # plt.grid(True, which="major", linestyle="--", linewidth=0.5)

    # Save the figure
    plt.tight_layout()
    # plt.savefig("data_yz_xz_xy" + ".pdf")
    plt.savefig("data_xx_yy_zz" + ".pdf")

    # Show the plot
    plt.show()

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
    #palette = sns.color_palette("bright", n_colors=3)

    # palette=["blue", "red", "green"]

    #palette = sns.color_palette("husl", n_colors=6)




    sns.set_theme(context="paper", style="white", palette="muted", font="serif")
    plt.figure(figsize=(8, 6))
    sns.histplot(data[:, 3], bins=60, kde=False, stat="density", color=palette[3],
                 edgecolor="black", alpha=0.4, label=r'$k_{yz}$')
    sns.histplot(data[:, 4], bins=60, kde=False, stat="density", color=palette[4], edgecolor="black",
                 alpha=0.4, label=r'$k_{xz}$')
    sns.histplot(data[:, 5],
                 bins=60, kde=False, stat="density", color=palette[5], edgecolor="black", alpha=0.4, label=r'$k_{xy}$')

    # sns.histplot(np.log10(data[:, 0]), bins=60, kde=False, stat="density", color="dodgerblue", edgecolor="black", alpha=0.3)
    # sns.histplot(np.log10(data[:, 1]), bins=60, kde=False, stat="density", color="salmon", edgecolor="black", alpha=0.3)
    # sns.histplot(np.log10(data[:, 2]), bins = 60, kde = False, stat = "density", color = "limegreen", edgecolor = "black", alpha = 0.3)
    # plt.hist(data, bins=30, density=True, color="dodgerblue", edgecolor="black", alpha=0.7)

    # Enforce scientific notation for x-ticks
    # Format the x-axis to use scientific notation
    plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter())
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.gca().ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    # Add labels and title
    plt.xlabel("value", fontsize=14)
    plt.ylabel("density", fontsize=14)
    # plt.title(fontsize=16, weight="bold")

    # Customize ticks
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.legend(fontsize=12)

    # Add grid for readability
    # plt.grid(True, which="major", linestyle="--", linewidth=0.5)

    # Save the figure
    plt.tight_layout()
    plt.savefig("data_yz_xz_xy" + ".pdf")
    #plt.savefig("data_xx_yy_zz" + ".pdf")

    # Show the plot
    plt.show()

    # for i in range(data.shape[1]):
    #     # if i in [0,1,2]:
    #     #     plot_hist(np.log10(data[:, i]), xlabel=x_labels_log[i], title="trf_target_{}".format(titles[i]))
    #     # else:
    #     plot_hist(data[:, i], xlabel=x_labels_log[i], title="trf_target_{}".format(titles[i]))

    # # plot_hist(np.log10(data[:, 1]), xlabel="Values of pixel " + r'$k_{yy}$', ylabel="Frequency",
    # #           title="trf_target_k_yy")
    # # plot_hist(np.log10(data[:, 2]), xlabel="Values of pixel " + r'$k_{zz}$', ylabel="Frequency",
    # #           title="trf_target_k_zz")
    # # plot_hist(data[:, 3], xlabel="Values of pixel " + r'$k_{yz}$', ylabel="Frequency",
    # #           title="trf_target_k_yz")
    # # plot_hist(data[:, 4], xlabel="Values of pixel " + r'$k_{xz}$', ylabel="Frequency",
    # #           title="trf_target_k_xz")
    # # plot_hist(data[:, 5], xlabel="Values of pixel " + r'$k_{xy}$', ylabel="Frequency",
    # #           title="trf_target_k_xy")
    #

def plot_data(loaded_outputs, loaded_bulk_avg, color):
    # Plot histogram
    #print("loaded_outputs[48000:, 0] ", loaded_outputs[59000:, 0])

    plt.figure(figsize=(8, 6))
    plt.hist(loaded_outputs[:, 0], bins=50, density=True, alpha=0.6, color=color, edgecolor='black')
    plt.title("Histogram of Data")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.show()

    print(np.any(loaded_outputs[:, 0] == 0))

    plt.figure(figsize=(8, 6))
    plt.hist(np.log10(loaded_outputs[:, 0]), bins=50, density=True, alpha=0.6, color=color, edgecolor='black')
    plt.title("Histogram of log Data")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.show()

    print("p.mean(bulk_avg_file,axis=1) ", np.mean(loaded_bulk_avg,axis=1))
    #loaded_outputs[:, i]/np.mean(loaded_bulk_avg,axis=1)

    # for i in range(6):
    #     plt.figure(figsize=(8, 6))
    #     plt.hist(loaded_outputs[:, i]/np.mean(loaded_bulk_avg,axis=1), bins=50, density=True, alpha=0.6, color=color, edgecolor='black')
    #     plt.title("Histogram of Data, i: {}".format(i))
    #     plt.xlabel("Value")
    #     plt.ylabel("Density")
    #     plt.show()
    #
    #     if i in range(3):
    #
    #         plt.figure(figsize=(8, 6))
    #         plt.hist(np.log10(loaded_outputs[:, i]/np.mean(loaded_bulk_avg, axis=1)), bins=50, density=True, alpha=0.6, color=color, edgecolor='black')
    #         plt.title("Histogram of log Data, i: {}".format(i))
    #         plt.xlabel("Value")
    #         plt.ylabel("Density")
    #         plt.show()

def plot_data_different_lambda(loaded_outputs_corr_length):


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
    palette = sns.color_palette("bright", n_colors=3)

    # palette=["blue", "red", "green"]

    sns.set_theme(context="paper", style="white", palette="muted", font="serif")
    plt.figure(figsize=(8, 6))
    sns.histplot(np.log10(loaded_outputs_corr_length[0][:20000, 0]), bins=10, kde=False, stat="density", color=palette[0],
                 edgecolor="black", alpha=0.4, label=r'$\lambda=0$')
    sns.histplot(np.log10(loaded_outputs_corr_length[1][:20000, 0]), bins=30, kde=False, stat="density", color=palette[1], edgecolor="black",
                 alpha=0.4, label=r'$\lambda=10$')
    sns.histplot(np.log10(loaded_outputs_corr_length[2][:20000, 0]),
                 bins=30, kde=False, stat="density", color=palette[2], edgecolor="black", alpha=0.4, label=r'$\lambda=25$')

    # sns.histplot(np.log10(data[:, 0]), bins=60, kde=False, stat="density", color=palette[0],
    #              edgecolor="black", alpha=0.4, label=r'$log(k_{xx})$')
    # sns.histplot(np.log10(data[:, 1]), bins=60, kde=False, stat="density", color=palette[1], edgecolor="black",
    #              alpha=0.4, label=r'$log(k_{yy})$')
    # sns.histplot(np.log10(data[:, 2]),
    #              bins=60, kde=False, stat="density", color=palette[2], edgecolor="black", alpha=0.4,
    #              label=r'$log(k_{zz})$')

    # sns.histplot(np.log10(data[:, 0]), bins=60, kde=False, stat="density", color="dodgerblue", edgecolor="black", alpha=0.3)
    # sns.histplot(np.log10(data[:, 1]), bins=60, kde=False, stat="density", color="salmon", edgecolor="black", alpha=0.3)
    # sns.histplot(np.log10(data[:, 2]), bins = 60, kde = False, stat = "density", color = "limegreen", edgecolor = "black", alpha = 0.3)
    # plt.hist(data, bins=30, density=True, color="dodgerblue", edgecolor="black", alpha=0.7)

    # Enforce scientific notation for x-ticks
    # Format the x-axis to use scientific notation
    plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter())
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.gca().ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    # Add labels and title
    plt.xlabel(r'$k_{xx}$', fontsize=14)
    plt.ylabel("density", fontsize=14)
    # plt.title(fontsize=16, weight="bold")

    # Customize ticks
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.legend(fontsize=12)

    # Add grid for readability
    # plt.grid(True, which="major", linestyle="--", linewidth=0.5)

    # Save the figure
    plt.tight_layout()
    # plt.savefig("data_yz_xz_xy" + ".pdf")
    plt.savefig("data_xx_cl_0_10_25" + ".pdf")

    plt.show()

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
    palette = sns.color_palette("bright", n_colors=3)

    # palette=["blue", "red", "green"]

    sns.set_theme(context="paper", style="white", palette="muted", font="serif")
    plt.figure(figsize=(8, 6))
    sns.histplot(loaded_outputs_corr_length[0][:20000, 5], bins=60, kde=False, stat="density",
                 color=palette[0],
                 edgecolor="black", alpha=0.4, label=r'$\lambda=0$')
    # sns.histplot(loaded_outputs_corr_length[1][:20000, 5], bins=60, kde=False, stat="density",
    #              color=palette[1], edgecolor="black",
    #              alpha=0.4, label=r'$\lambda=10$')
    # sns.histplot(loaded_outputs_corr_length[2][:20000, 5],
    #              bins=60, kde=False, stat="density", color=palette[2], edgecolor="black", alpha=0.4,
    #              label=r'$\lambda=25$')

    # sns.histplot(np.log10(data[:, 0]), bins=60, kde=False, stat="density", color=palette[0],
    #              edgecolor="black", alpha=0.4, label=r'$log(k_{xx})$')
    # sns.histplot(np.log10(data[:, 1]), bins=60, kde=False, stat="density", color=palette[1], edgecolor="black",
    #              alpha=0.4, label=r'$log(k_{yy})$')
    # sns.histplot(np.log10(data[:, 2]),
    #              bins=60, kde=False, stat="density", color=palette[2], edgecolor="black", alpha=0.4,
    #              label=r'$log(k_{zz})$')

    # sns.histplot(np.log10(data[:, 0]), bins=60, kde=False, stat="density", color="dodgerblue", edgecolor="black", alpha=0.3)
    # sns.histplot(np.log10(data[:, 1]), bins=60, kde=False, stat="density", color="salmon", edgecolor="black", alpha=0.3)
    # sns.histplot(np.log10(data[:, 2]), bins = 60, kde = False, stat = "density", color = "limegreen", edgecolor = "black", alpha = 0.3)
    # plt.hist(data, bins=30, density=True, color="dodgerblue", edgecolor="black", alpha=0.7)

    # Enforce scientific notation for x-ticks
    # Format the x-axis to use scientific notation
    plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter())
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.gca().ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    # Add labels and title
    plt.xlabel(r'$k_{xy}$', fontsize=14)
    plt.ylabel("density", fontsize=14)
    # plt.title(fontsize=16, weight="bold")

    # Customize ticks
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.legend(fontsize=12)

    # Add grid for readability
    # plt.grid(True, which="major", linestyle="--", linewidth=0.5)

    # Save the figure
    plt.tight_layout()
    plt.savefig("data_xy_cl_0" + ".pdf")
    #plt.savefig("data_xx_yy_zz" + ".pdf")

    plt.show()
    ##########################################################3
    ###########################################################3

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
    palette = sns.color_palette("bright", n_colors=3)

    # palette=["blue", "red", "green"]

    sns.set_theme(context="paper", style="white", palette="muted", font="serif")
    plt.figure(figsize=(8, 6))
    # sns.histplot(loaded_outputs_corr_length[0][:20000, 5], bins=60, kde=False, stat="density",
    #              color=palette[0],
    #              edgecolor="black", alpha=0.4, label=r'$\lambda=0$')
    sns.histplot(loaded_outputs_corr_length[1][:20000, 5], bins=60, kde=False, stat="density",
                 color=palette[1], edgecolor="black",
                 alpha=0.4, label=r'$\lambda=10$')
    sns.histplot(loaded_outputs_corr_length[2][:20000, 5],
                 bins=60, kde=False, stat="density", color=palette[2], edgecolor="black", alpha=0.4,
                 label=r'$\lambda=25$')

    # sns.histplot(np.log10(data[:, 0]), bins=60, kde=False, stat="density", color=palette[0],
    #              edgecolor="black", alpha=0.4, label=r'$log(k_{xx})$')
    # sns.histplot(np.log10(data[:, 1]), bins=60, kde=False, stat="density", color=palette[1], edgecolor="black",
    #              alpha=0.4, label=r'$log(k_{yy})$')
    # sns.histplot(np.log10(data[:, 2]),
    #              bins=60, kde=False, stat="density", color=palette[2], edgecolor="black", alpha=0.4,
    #              label=r'$log(k_{zz})$')

    # sns.histplot(np.log10(data[:, 0]), bins=60, kde=False, stat="density", color="dodgerblue", edgecolor="black", alpha=0.3)
    # sns.histplot(np.log10(data[:, 1]), bins=60, kde=False, stat="density", color="salmon", edgecolor="black", alpha=0.3)
    # sns.histplot(np.log10(data[:, 2]), bins = 60, kde = False, stat = "density", color = "limegreen", edgecolor = "black", alpha = 0.3)
    # plt.hist(data, bins=30, density=True, color="dodgerblue", edgecolor="black", alpha=0.7)

    # Enforce scientific notation for x-ticks
    # Format the x-axis to use scientific notation
    plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter())
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.gca().ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    # Add labels and title
    plt.xlabel(r'$k_{xy}$', fontsize=14)
    plt.ylabel("density", fontsize=14)
    # plt.title(fontsize=16, weight="bold")

    # Customize ticks
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.legend(fontsize=12)

    # Add grid for readability
    # plt.grid(True, which="major", linestyle="--", linewidth=0.5)

    # Save the figure
    plt.tight_layout()
    plt.savefig("data_xy_cl_10_25" + ".pdf")
    # plt.savefig("data_xx_yy_zz" + ".pdf")

    plt.show()

    # Show the plot


# Example usage: Provide paths to your datasets
data_paths = ["/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_5/cond_nearest_interp/train_data/dataset_2_save_bulk_avg",
              "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_5/cond_nearest_interp/train_data/dataset_2_save_bulk_avg_n_frac_2500"]

data_paths = [#"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_3/cond_nearest_interp/train_data/dataset_2_save_bulk_avg_n_frac_2500_cl_0_10_25_seed_merge",
            #"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_3/cond_nearest_interp/train_data/dataset_2_save_bulk_avg_n_frac_2500_seed",
"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_3/cond_nearest_interp/train_data/dataset_2_save_bulk_avg_cl_10",
#"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_3/cond_nearest_interp/test_data/MLMC-DFM_3D_n_voxels_64_save_bulk_avg_cl_10",
"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_3/cond_nearest_interp/train_data/dataset_2_save_bulk_avg_cl_25",
#"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_3/cond_nearest_interp/test_data/MLMC-DFM_3D_n_voxels_64_save_bulk_avg_cl_2",
              #"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_3/cond_nearest_interp/test_data/MLMC-DFM_3D_n_voxels_64_save_bulk_avg_n_frac_2500",
              #"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_3/cond_nearest_interp/test_data/MLMC-DFM_3D_n_voxels_64_save_bulk_avg"
]

#data_paths = ["/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_3/cond_nearest_interp/train_data/dataset_2_save_bulk_avg_n_frac_2500_cl_0_10_25_seed_merge"]

data_paths = ["/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_3/cond_nearest_interp/train_data/dataset_2_save_bulk_avg_no_frac_cl_0_10_25_seed_merge"]
#data_paths = ["/home/martin/Documents/MLMC-DFM/optuna_runs/3D_cnn/lumi/cond_frac_1_7/cond_nearest_interp/dataset_2_save_bulk_avg_n_frac_2500_cl_0_10_25_seed_merge"]


data_paths = ["/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_3/cond_nearest_interp/train_data/dataset_2_save_bulk_avg_no_frac",
             "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_3/cond_nearest_interp/train_data/dataset_2_save_bulk_avg_cl_10_no_frac",
             "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_3/cond_nearest_interp/train_data/dataset_2_save_bulk_avg_cl_25_no_frac",
            ]

data_paths = ["/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_3/cond_nearest_interp/train_data/dataset_2_save_bulk_avg",
             "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_3/cond_nearest_interp/train_data/dataset_2_save_bulk_avg_cl_10",
             "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_3/cond_nearest_interp/train_data/dataset_2_save_bulk_avg_cl_25",
            ]

data_paths = ["/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_5/cond_nearest_interp/train_data/dataset_2_save_bulk_avg",
             "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_5/cond_nearest_interp/train_data/dataset_2_save_bulk_avg_cl_10",
             "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_5/cond_nearest_interp/train_data/dataset_2_save_bulk_avg_cl_25",
            ]


data_paths = ["/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_7/cond_nearest_interp/train_data/dataset_2_save_bulk_avg",
             "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_7/cond_nearest_interp/train_data/dataset_2_save_bulk_avg_cl_10",
             "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_7/cond_nearest_interp/train_data/dataset_2_save_bulk_avg_cl_25",
            ]


all_loaded_outputs = []
all_loaded_bulk_avg = []

loaded_outputs_corr_length = []

colors = ["red", "orange", "blue", "green"]
for i, data_path in enumerate(data_paths):
    outputs_file = "output_data.npz"
    bulk_avg_file = "bulk_avg.npz"

    loaded_outputs = np.load(os.path.join(data_path, outputs_file))["data"]
    loaded_bulk_avg = np.load(os.path.join(data_path, bulk_avg_file))["data"]


    print("loaded outputs shape ", loaded_outputs.shape)

    # start_index = 0 #14500
    # end_index = 20000
    # for output in loaded_outputs[start_index:end_index]:
    #     #print("output ", output)
    #
    #     if np.any(output.flatten() == 0):
    #         print("idx: {}".format(start_index))
    #         print("output ", output)
    #
    #
    #     start_index += 1
    #
    # start_index = 0  # 14500
    # end_index = 20000
    # for output in loaded_bulk_avg[start_index:end_index]:
    #     # print("output ", output)
    #
    #     if np.any(output.flatten() == 0):
    #         print("idx: {}".format(start_index))
    #         print("output ", output)
    #
    #     start_index += 1
    #exit()

    all_loaded_outputs.extend(list(loaded_outputs))
    all_loaded_bulk_avg.extend(list(loaded_bulk_avg))

    loaded_outputs_corr_length.append(loaded_outputs)

    print("loaded outpus shape ", loaded_outputs.shape)
    print("loaded bulk avg shape ", loaded_bulk_avg.shape)

    #plot_hist_output(loaded_outputs)


    #plot_data(loaded_outputs, loaded_bulk_avg, color=colors[i])

#plot_data(np.array(all_loaded_outputs), np.array(all_loaded_bulk_avg), color="blue")

plot_data_different_lambda(loaded_outputs_corr_length)
