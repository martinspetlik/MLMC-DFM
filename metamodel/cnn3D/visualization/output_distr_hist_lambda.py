import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker

def plot_dist_hist(
    output_data,
    cols=(0, 1, 2),
    n_colors=6,
    labels=(r'$ \log_{10}(k_{xx}) $', r'$ \log_{10}(k_{yy}) $', r'$ \log_{10}(k_{zz}) $'),
    bins=50,
    figsize=(7, 4.5),
    filename="data_xx_yy_zz.pdf",
    dpi=300,
    alpha=0.6,
    use_log=True,
    fontsize=12,
    fontsize_ticks=11,
    rotation=0  # set to 30 if still crowded
):
    """
    Create a clean, publication-quality histogram plot for log10-transformed permeability components.
    """

    # Style
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.2)
    palette = sns.color_palette("Set2", n_colors=n_colors)

    fig, ax = plt.subplots(figsize=figsize)

    # Format y-axis in scientific notation
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(formatter)

    ####
    # Plot K_xx
    ####
    col = 0
    i = 0
    # Plot histograms
    for corr_length, values in output_data.items():
        data = np.log10(values[:, col])

        used_bins = bins
        if i == 0:
            used_bins = 20

        ax.hist(
            data,
            bins=used_bins,
            density=True,
            color=palette[i],
            label=r'$\lambda=$' + "{}".format(corr_length),
            alpha=alpha,
            edgecolor="black",
            linewidth=0.4,
        )

        i+= 1

    # ax.xaxis.get_offset_text().set_fontsize(24)
    # ax.yaxis.get_offset_text().set_fontsize(24)
    #
    # # Customize ticks
    # plt.xticks(fontsize=24)
    # plt.yticks(fontsize=24)

    #plt.legend(markerscale=3, fontsize=24)
    # plt.rcParams['text.usetex'] = True
    # plt.rcParams['font.family'] = 'serif'
    ax.set_xlabel(r'$\log(k_{xx})$', fontsize=fontsize)

    # Labels, legend
    #ax.set_xlabel(r'$log(\boldsymbol{K}^{eq}_{xx})$', fontsize=fontsize)
    ax.set_ylabel("Density", fontsize=fontsize)
    ax.legend(fontsize=fontsize)

    # Make x-ticks nice and readable
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune='both'))
    ax.tick_params(labelsize=fontsize_ticks)
    ax.yaxis.get_offset_text().set_fontsize(fontsize_ticks)
    ax.xaxis.get_offset_text().set_fontsize(fontsize_ticks)
    # ax.tick_params(axis='x', labelsize=fontsize_ticks)
    # ax.tick_params(axis='y', labelsize=fontsize_ticks)
    plt.setp(ax.get_xticklabels(), rotation=rotation)

    # Add a small margin to prevent tick overlap with edges
    ax.margins(x=0.05)

    # Remove grid for a cleaner dissertation look
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.show()


    ############
    ############
    fig, ax = plt.subplots(figsize=figsize)

    # Format y-axis in scientific notation
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(formatter)

    print("output_data['0'][:, 5].shape ", output_data["0"][:, 5].shape)

    ####
    # Plot K_xx
    ####
    ax.hist(output_data["0"][:, 5],
            bins=bins,
            density=True,
            color=palette[0],
            label=r'$\lambda=0$',
            alpha=alpha,
            edgecolor="black",
            linewidth=0.4,
     )

    # ax.xaxis.get_offset_text().set_fontsize(24)
    # ax.yaxis.get_offset_text().set_fontsize(24)
    #
    # # Customize ticks
    # plt.xticks(fontsize=24)
    # plt.yticks(fontsize=24)

    # plt.legend(markerscale=3, fontsize=24)
    # plt.rcParams['text.usetex'] = True
    # plt.rcParams['font.family'] = 'serif'
    ax.set_xlabel(r'$k_{xy}$', fontsize=fontsize)

    # Labels, legend
    # ax.set_xlabel(r'$log(\boldsymbol{K}^{eq}_{xx})$', fontsize=fontsize)
    ax.set_ylabel("Density", fontsize=fontsize)
    ax.legend(fontsize=fontsize)

    # Make x-ticks nice and readable
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune='both'))
    ax.tick_params(labelsize=fontsize_ticks)
    ax.yaxis.get_offset_text().set_fontsize(fontsize_ticks)
    ax.xaxis.get_offset_text().set_fontsize(fontsize_ticks)
    # ax.tick_params(axis='x', labelsize=fontsize_ticks)
    # ax.tick_params(axis='y', labelsize=fontsize_ticks)
    plt.setp(ax.get_xticklabels(), rotation=rotation)

    # Add a small margin to prevent tick overlap with edges
    ax.margins(x=0.05)

    # Remove grid for a cleaner dissertation look
    ax.grid(False)

    plt.tight_layout()
    plt.savefig("data_xy_cl_0_new.pdf", dpi=dpi, bbox_inches="tight")
    plt.show()

    ##############
    ##############
    ############
    ############
    fig, ax = plt.subplots(figsize=figsize)

    # Format y-axis in scientific notation
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(formatter)

    ####
    # Plot K_xx
    ####
    ax.hist(output_data["10"][:, 5],
            bins=bins,
            density=True,
            color=palette[1],
            label=r'$\lambda=10$',
            alpha=alpha,
            edgecolor="black",
            linewidth=0.4,
            )

    ax.hist(output_data["25"][:, 5],
            bins=bins,
            density=True,
            color=palette[2],
            label=r'$\lambda=25$',
            alpha=alpha,
            edgecolor="black",
            linewidth=0.4,
            )

    # ax.xaxis.get_offset_text().set_fontsize(24)
    # ax.yaxis.get_offset_text().set_fontsize(24)
    #
    # # Customize ticks
    # plt.xticks(fontsize=24)
    # plt.yticks(fontsize=24)

    # plt.legend(markerscale=3, fontsize=24)
    # plt.rcParams['text.usetex'] = True
    # plt.rcParams['font.family'] = 'serif'
    ax.set_xlabel(r'$k_{xy}$', fontsize=fontsize)

    # Labels, legend
    # ax.set_xlabel(r'$log(\boldsymbol{K}^{eq}_{xx})$', fontsize=fontsize)
    ax.set_ylabel("Density", fontsize=fontsize)
    ax.legend(fontsize=fontsize)

    # Make x-ticks nice and readable
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune='both'))
    ax.tick_params(labelsize=fontsize_ticks)
    ax.yaxis.get_offset_text().set_fontsize(fontsize_ticks)
    ax.xaxis.get_offset_text().set_fontsize(fontsize_ticks)
    # ax.tick_params(axis='x', labelsize=fontsize_ticks)
    # ax.tick_params(axis='y', labelsize=fontsize_ticks)
    plt.setp(ax.get_xticklabels(), rotation=rotation)

    # Add a small margin to prevent tick overlap with edges
    ax.margins(x=0.05)

    # Remove grid for a cleaner dissertation look
    ax.grid(False)

    plt.tight_layout()
    plt.savefig("data_xy_cl_10_25_new.pdf", dpi=dpi, bbox_inches="tight")
    plt.show()




corr_length_data_dirs = {"0": ["/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_3/cond_nearest_interp/train_data/fractures_diag_cond/dataset_2_save_bulk_avg", "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_3/cond_nearest_interp/train_data/fractures_diag_cond/dataset_2_save_bulk_avg_n_frac_2500_seed"],
             "10": ["/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_3/cond_nearest_interp/train_data/fractures_diag_cond/dataset_2_save_bulk_avg_cl_10", "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_3/cond_nearest_interp/train_data/fractures_diag_cond/dataset_2_save_bulk_avg_n_frac_2500_seed_cl_10"],
             "25": ["/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_3/cond_nearest_interp/train_data/fractures_diag_cond/dataset_2_save_bulk_avg_cl_25", "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_3/cond_nearest_interp/train_data/fractures_diag_cond/dataset_2_save_bulk_avg_n_frac_2500_seed_cl_25"]}

all_corr_length_output_data = {}
for corr_length, data_dirs in corr_length_data_dirs.items():
    all_output_data = []
    for data_dir in data_dirs:
        path_to_data_dir = data_dir

        output_data = np.load(os.path.join(path_to_data_dir, "output_data.npz"))["data"][:10000]
        all_output_data.extend(output_data)

    all_corr_length_output_data[corr_length] = np.array(all_output_data)


plot_dist_hist(all_corr_length_output_data, cols=(0, 1, 2), labels=(r'$$', r'$ \log_{10}(k_{yy}) $', r'$ \log_{10}(k_{zz}) $'), filename="data_xx_cl_0_10_25_new.pdf", use_log=True, fontsize=17, fontsize_ticks=15)