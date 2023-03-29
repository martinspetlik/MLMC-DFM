import numpy as np
import matplotlib.pyplot as plt


def plot_target_prediction(all_targets, all_predictions):
    all_targets = np.squeeze(all_targets)
    all_predictions = np.squeeze(all_predictions)

    print("len(all_targets.shape) ", len(all_targets.shape))

    if len(all_targets.shape) == 1:
        all_targets = np.reshape(all_targets, (*all_targets.shape, 1))
        all_predictions = np.reshape(all_predictions, (*all_predictions.shape, 1))

    n_channels = 1 if len(all_targets.shape) == 1 else all_targets.shape[1]

    x_labels = [r'$k_{xx}$', r'$k_{xy}$', r'$k_{yy}$']
    titles = ["k_xx", "k_xy", "k_yy"]

    print("n_channels ", n_channels)

    for i in range(n_channels):
        k_target, k_predict = all_targets[:, i], all_predictions[:, i]
        plot_hist(k_target, k_predict, xlabel=x_labels[i], title=titles[i])
        plot_t_p(k_target, k_predict, label=x_labels[i], title=titles[i])


    # k_xx_target, k_xx_pred = all_targets[:, 0], all_predictions[:, 0]
    # k_xy_target, k_xy_pred = all_targets[:, 1], all_predictions[:, 1]
    # k_yy_target, k_yy_pred = all_targets[:, 2], all_predictions[:, 2]
    #
    # plot_hist(k_xx_target, k_xx_pred, xlabel=r'$k_{xx}$', title="k_xx")
    # plot_hist(k_xy_target, k_xy_pred, xlabel=r'$k_{xy}$', title="k_xy")
    # plot_hist(k_yy_target, k_yy_pred, xlabel=r'$k_{yy}$', title="k_yy")
    #
    # plot_t_p(k_xx_target, k_xx_pred, label=r'$k_{xx}$', title="k_xx")
    # plot_t_p(k_xy_target, k_xy_pred, label=r'$k_{xy}$', title="k_xy")
    # plot_t_p(k_yy_target, k_yy_pred, label=r'$k_{yy}$', title="k_yy")


def plot_hist(target, prediction, xlabel="k", title="hist"):
    plt.hist(target, bins=60,  color="red", label="target", density=True)
    plt.hist(prediction, bins=60, color="blue", label="prediction", density=True)
    plt.xlabel(xlabel)
    #plt.ylabel("Frequency for relative")
    plt.legend()
    plt.savefig("hist_" + title + ".pdf")
    plt.show()


def plot_t_p(targets, predictions, label="k", title="k"):
    fig, ax = plt.subplots()
    ax.scatter(targets, predictions, edgecolors=(0, 0, 0))
    ax.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'k--', lw=4)
    ax.set_xlabel('Targets')
    ax.set_ylabel('Predictions')
    plt.title(label)
    plt.legend()
    plt.savefig(title + ".pdf")
    plt.show()


def plot_train_valid_loss(train_loss, valid_loss):
    plt.plot(train_loss, label="train loss")
    plt.plot(valid_loss, label="valid loss")
    plt.xlabel("epochs")
    plt.legend()
    plt.savefig("train_val_loss.pdf")
    plt.show()


def plot_samples(data_loader, n_samples=10):
    import matplotlib.pyplot as plt
    for idx, data in enumerate(data_loader):
        if idx > n_samples:
            break
        input, output = data
        #img = img / 2 + 0.5  # unnormalize
        #npimg = img.numpy()
        plt_input = input[0]
        plt_output = output[0]

        print("plt_input ", plt_input)

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
        axes[0].matshow(plt_input[0])
        axes[1].matshow(plt_input[1])
        axes[2].matshow(plt_input[2])
        #fig.colorbar(caxes)
        plt.savefig("input_{}.pdf".format(idx))
        plt.show()

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        axes.matshow(np.reshape(plt_output, (3, 1)))
        # fig.colorbar(caxes)
        plt.savefig("output_{}.pdf".format(idx))
        plt.show()


        from metamodel.cnn.visualization.visualize_tensor import reshape_to_tensors, plot_cond_tn
        cond_tn_target = reshape_to_tensors(plt_output, dim=2)[0:2, 0:2]
        plot_cond_tn(cond_tn_target, label="target_tn_", color="red")


        # plt.matshow(plt_input[0])
        # plt.matshow(plt_input[1])
        # plt.matshow(plt_input[2])
        # plt.show()

    exit()
