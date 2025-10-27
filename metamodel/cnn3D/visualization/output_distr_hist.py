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

    # Plot histograms
    for i, col in enumerate(cols):
        if use_log:
            data = np.log10(output_data[:, col])
        else:
            data = output_data[:, col]

        ax.hist(
            data,
            bins=bins,
            density=True,
            color=palette[col],
            label=labels[i],
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

    #plt.legend(markerscale=3, fontsize=24)

    # Labels, legend
    ax.set_xlabel("Value", fontsize=fontsize)
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




data_dirs = ["/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_3/cond_nearest_interp/train_data/fractures_diag_cond/dataset_2_save_bulk_avg_n_frac_2500_cl_0_10_25_seed_merge"]
#data_dirs = ["/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_5/cond_nearest_interp/train_data/fractures_diag_cond/dataset_2_save_bulk_avg_n_frac_2500_cl_0_10_25_seed_merge"]
#data_dirs = ["/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/homogenization_samples/cond_frac_1_7/cond_nearest_interp/train_data/fractures_diag_cond/dataset_2_save_bulk_avg_n_frac_2500_cl_0_10_25_seed_merge"]
all_output_data = []
for data_dir in data_dirs:
    path_to_data_dir = data_dir

    output_data = np.load(os.path.join(path_to_data_dir, "output_data.npz"))["data"]

    all_output_data.extend(output_data)


all_data_np = np.array(all_output_data)

print("all_data_np[0] ", all_data_np[0])


plot_dist_hist(all_data_np, cols=(0, 1, 2), labels=(r'$ \log_{10}(k_{xx}) $', r'$ \log_{10}(k_{yy}) $', r'$ \log_{10}(k_{zz}) $'), filename="data_xx_yy_zz_new.pdf", use_log=True, fontsize=17, fontsize_ticks=15)
plot_dist_hist(all_data_np, cols=(3, 4, 5), labels=(r'$k_{yz}$', r'$k_{xz}$', r'$k_{xy}$'), filename="data_yz_xz_xy_new.pdf", use_log=False, fontsize=17, fontsize_ticks=15)

print("all_data_np.shape ", all_data_np.shape)