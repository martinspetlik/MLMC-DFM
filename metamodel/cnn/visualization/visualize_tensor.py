import numpy as np
import matplotlib.pyplot as plt
from metamodel.cnn.models.auxiliary_functions import reshape_to_tensors


def plot_tensors(cond_tn_prediction, cond_tn_target, label="tensors", plot_separate_images=False, dim=2):
    """
    Plot principal components of tensors
    """
    cond_tn_prediction = np.squeeze(cond_tn_prediction)
    cond_tn_target = np.squeeze(cond_tn_target)

    cond_tn_prediction = reshape_to_tensors(cond_tn_prediction, dim=dim)[0:dim, 0:dim]
    cond_tn_target = reshape_to_tensors(cond_tn_target, dim=dim)[0:dim, 0:dim]

    if plot_separate_images:
        plot_cond_tn(cond_tn_target, label="target_tn_"+label, color="red")
        plot_cond_tn(cond_tn_prediction, label="prediction_tn_"+label, color="blue")
    else:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        ax = axes
        ax.set_aspect('equal')

        plot_evecs(ax, cond_tn_target, label="target", color="red")
        plot_evecs(ax, cond_tn_prediction, label="prediction", color="blue")
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