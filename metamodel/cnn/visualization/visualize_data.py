import numpy as np
import matplotlib.pyplot as plt


def plot_target_prediction(all_targets, all_predictions):
    all_targets = np.squeeze(all_targets)
    all_predictions = np.squeeze(all_predictions)

    k_xx_target, k_xx_pred = all_targets[:, 0], all_predictions[:, 0]
    k_xy_target, k_xy_pred = all_targets[:, 1], all_predictions[:, 1]
    k_yy_target, k_yy_pred = all_targets[:, 2], all_predictions[:, 2]

    plot_hist(k_xx_target, k_xx_pred, xlabel=r'$k_{xx}$')
    plot_hist(k_xy_target, k_xy_pred, xlabel=r'$k_{xy}$')
    plot_hist(k_yy_target, k_yy_pred, xlabel=r'$k_{yy}$')

    plot_t_p(k_xx_target, k_xx_pred, title=r'$k_{yy}$')
    plot_t_p(k_xy_target, k_xy_pred, title=r'$k_{yy}$')
    plot_t_p(k_yy_target, k_yy_pred, title=r'$k_{yy}$')

    print("k_xx_target ", k_xx_target)
    print("k_xx_pred ", k_xx_pred)

    # plt.hist(k_xx_target, bins=60, density=True)
    # plt.xlabel(r'$k_{xx}$')
    # plt.ylabel("Frequency for relative")
    # plt.title("target")
    # plt.show()
    #
    # plt.hist(k_xx_pred, bins=60, density=True)
    # plt.xlabel(r'$k_{xx}$')
    # plt.ylabel("Frequency for relative")
    # plt.title("prediction")
    # plt.show()

    # plt.scatter(x_train, y_train, color="red")
    # plt.plot(x_train, lr.predict(x_train), color="green")
    # plt.title("Salary vs Experience (Training set)")
    # plt.xlabel("Years of Experience")
    # plt.ylabel("Salary")
    # plt.show()
    # exit()


def plot_hist(target, prediction, xlabel="k"):
    plt.hist(target, bins=60,  color="red", label="target")
    plt.hist(prediction, bins=60, color="blue", label="prediction")
    plt.xlabel(xlabel)
    #plt.ylabel("Frequency for relative")
    plt.legend()
    plt.show()


def plot_t_p(targets, predictions, title="k"):
    fig, ax = plt.subplots()
    ax.scatter(targets, predictions, edgecolors=(0, 0, 0))
    ax.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'k--', lw=4)
    ax.set_xlabel('Targets')
    ax.set_ylabel('Predictions')
    plt.title(title)
    plt.show()
