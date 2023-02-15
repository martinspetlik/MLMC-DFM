import numpy as np
import matplotlib.pyplot as plt


def reshape_to_tensors(tn_array):
    tn3d = np.eye(3)
    tn3d[np.triu_indices(3)] = tn_array

    diagonal_values = np.diag(tn3d)
    symmetric_tn = tn3d + tn3d.T
    np.fill_diagonal(symmetric_tn, diagonal_values)
    return symmetric_tn


def plot_tensors(cond_tn_prediction, cond_tn_target, label="tensors", separate_images=False):
    """
    Plot principal components of tensors
    """
    if cond_tn_target.shape[0] < 3:
        cond_tn_prediction = reshape_to_tensors(cond_tn_prediction)
        cond_tn_target = reshape_to_tensors(cond_tn_target)

    cond_tn_prediction_2d = cond_tn_prediction[0:2, 0:2]
    cond_tn_target_2d = cond_tn_target[0:2, 0:2]

    if separate_images:
        plot_cond_tn(cond_tn_target_2d, label="target_tn_"+label, color="red")
        plot_cond_tn(cond_tn_prediction_2d, label="prediction_tn_"+label, color="blue")
    else:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        ax = axes
        ax.set_aspect('equal')

        plot_evecs(ax, cond_tn_target_2d, label="target", color="red")
        plot_evecs(ax, cond_tn_prediction_2d, label="prediction", color="blue")
        fig.suptitle(label)

        # ax_polar.grid(True)
        fig.savefig("{}.pdf".format(label))
        # plt.close(fig)
        plt.legend()
        plt.show()



def plot_cond_tn(cond_tn, label, color):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    ax = axes
    ax.set_aspect('equal')

    plot_evecs(ax, cond_tn, label=label, color=color)
    fig.suptitle(label)

    fig.savefig("{}.pdf".format(label))
    plt.legend()
    plt.show()




def plot_evecs(ax, tn, color, label):
    e_val, e_vec = np.linalg.eigh(tn)
    labeled_arrow(ax, [0, 0], 0.9 * e_val[0] * e_vec[:, 0], "{:5.2g}".format(e_val[0]), color=color)
    labeled_arrow(ax, [0, 0], 0.9 * e_val[1] * e_vec[:, 1], "{:5.2g}".format(e_val[1]), color=color)



def labeled_arrow(ax, start, end, label, color="red"):
    """
    Labeled and properly scaled arrow.
    :param start: origin point, [x,y]
    :param end: tip point [x,y]
    :param label: string label, placed near the tip
    :return:
    """
    scale = np.linalg.norm(end - start)
    ax.arrow(*start, *end, width=0.003 * scale, head_length=0.1 * scale, head_width =0.05 * scale, facecolor=color)
    if (end - start)[1] > 0:
        vert_align = 'bottom'
    else:
        vert_align = 'top'
    ax.annotate(label, end + 0.1*(end - start), va=vert_align)