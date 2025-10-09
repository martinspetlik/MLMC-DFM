import os
import numpy as np
import mlmc
from mlmc.sample_storage_hdf import SampleStorageHDF
from mlmc.moments import Legendre, Monomial
from mlmc.quantity.quantity import make_root_quantity
from mlmc.quantity.quantity_estimate import estimate_mean
from mlmc import estimator
from mlmc.plot import diagnostic_plots as dp
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib.pyplot as plt
import zarr
import glob

from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit


def plot_bias(list_l_params, list_all_moments_mean_obj, ref_moments_mean_obj, levels=None, moment=1, err_l_vars=None):
    """
    Plot log2 variance vs level and fit slope to estimate beta.

    l_vars: array shape (n_levels, n_moments) with variance per level & moment
    levels: array of level indices (default = np.arange(n_levels))
    moments: list of moment indices to plot
    err_l_vars: optional errors for l_vars
    """
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    ref_moment_mean = ref_moments_mean_obj.mean[moment]

    print("ref_moment_mean ", ref_moment_mean)

    for i in range(len(list_l_params)):
        l_params = np.squeeze(list_l_params[i])
        print("l_params ", l_params)
        moment_mean = list_all_moments_mean_obj[i].mean[moment]

        bias = moment_mean - ref_moment_mean

        print("abs bias ", np.abs(bias))

    #     ax.plot(l_params, y, 'o-')
    #
    #     slope, intercept = np.polyfit(l_params, y, 1)
    #     beta = -slope
    #     ax.plot(l_params, slope*l_params + intercept, '--',
    #             label=r'$\beta≈$' + f'{beta:.2f}')
    #
    # ax.set_ylabel(r'$\log_2 \, V_l$')
    # ax.set_xlabel('mesh step $h$')
    # ax.legend()
    # #ax.grid(True, which="both")
    # plt.tight_layout()
    # plt.show()


def plot_variance_mesh_size(list_l_params, list_all_moments_mean_obj, levels=None, moments=[0], err_l_vars=None):
    """
    Plot log2 variance vs level and fit slope to estimate beta.

    l_vars: array shape (n_levels, n_moments) with variance per level & moment
    levels: array of level indices (default = np.arange(n_levels))
    moments: list of moment indices to plot
    err_l_vars: optional errors for l_vars
    """
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    for i in range(len(list_l_params)):
        l_params = np.squeeze(list_l_params[i])
        print("l_params ", l_params)
        l_vars = list_all_moments_mean_obj[i].l_vars
        # n_levels = l_vars.shape[0]
        # if levels is None:
        #     levels = np.arange(n_levels)

        for m in moments:
            y = np.log2(l_vars[:, m])
            ax.plot(l_params, y, 'o-')

            slope, intercept = np.polyfit(l_params, y, 1)
            beta = -slope
            ax.plot(l_params, slope*l_params + intercept, '--',
                    label=r'$\beta≈$' + f'{beta:.2f}')

    ax.set_ylabel(r'$\log_2 \, V_l$')
    ax.set_xlabel('mesh step $h$')
    ax.legend()
    #ax.grid(True, which="both")
    plt.tight_layout()
    plt.show()


def plot_cost_mesh_size(list_l_params, list_costs, levels=None):
    """
    Plot log2 cost per level and fit slope to estimate gamma.

    costs: array of cost per sample at each level
    levels: optional array of level indices (default = 0,1,...)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # n_levels = len(costs)
    # if levels is None:
    #     levels = np.arange(n_levels)

    fig, ax = plt.subplots(figsize=(8, 5))

    for i in range(len(list_l_params)):
        l_params = np.squeeze(list_l_params[i])
        costs = list_costs[i]
        y = np.log2(costs)
        slope, intercept = np.polyfit(l_params, y, 1)
        gamma = slope

        ax.plot(l_params, y, 'o-')
        ax.plot(l_params, slope * l_params + intercept, '--',
                label=r'$\gamma≈$' + f'{gamma:.2f}')

    ax.set_ylabel(r'$\log_2 \, C_l (sec)$')
    ax.set_xlabel('mesh step $h$')
    ax.legend()
    #ax.grid(True, which="both")
    plt.tight_layout()
    plt.show()

    return gamma



def compare_means(samples_a, samples_b, title=""):
    # #####
    # # remove outliers
    # #####
    # samples_a = self._remove_outliers(samples_a)
    # samples_b = self._remove_outliers(samples_b)
    #
    # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    # axes.hist(samples_a, bins=100, density=True, label="samples A")
    # axes.hist(samples_b, bins=100, density=True, label="samples B", alpha=0.5)
    # fig.suptitle("Samples A, Samples B distr")
    # plt.show()

    # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    # axes.hist(samples_b, bins=100, density=True)
    # fig.suptitle("Samples B")
    # plt.show()
    ########
    #######

    statistic, p_value = stats.ttest_ind(np.squeeze(samples_a), np.squeeze(samples_b), equal_var=False)
    # statistic, p_value = self.two_sample_ztest(np.squeeze(samples_a), np.squeeze(samples_b))
    # statistic, p_value = stats.chisquare(np.squeeze(samples_a), np.squeeze(samples_b))
    alpha = 0.05
    print("p value ", p_value)
    # Check if the p-value is less than alpha
    if p_value < alpha:
        print("{} are significantly different.".format(title))
    else:
        print("There is no significant difference between {}".format(title))


def process(mlmc_file_path, target_var):
    n_moments = 3
    sample_storage = SampleStorageHDF(file_path=mlmc_file_path)
    n_levels = sample_storage.get_n_levels()
    level_parameters = sample_storage.get_level_parameters()
    print("n_levels ", n_levels)
    sample_storage.chunk_size = 1e8
    #target_var = 1e-6  # 5e-9
    root_quantity = make_root_quantity(storage=sample_storage,
                                       q_specs=sample_storage.load_result_format())

    cond_tn_quantity = root_quantity["cond_tn"]
    time_mean = cond_tn_quantity[1]  # times: [1]
    location_mean = time_mean['0']  # locations: ['0']
    k_xx = location_mean[0]
    q_value = k_xx  # cond_tn_quantity#k_xx
    # q_value = np.array([location_mean[0], location_mean[1], location_mean[2]])

    print("result format ", sample_storage.load_result_format()[0].shape[0])

    q_idx = 0

    # for q_idx in range(sample_storage.load_result_format()[0].shape[0]):
    #     print("q_idx ", q_idx)

    q_value = location_mean[q_idx]
    print("q_value ", q_value)

    # @TODO: How to estimate true_domain?
    quantile = 1e-3
    true_domain = mlmc.estimator.Estimate.estimate_domain(q_value, sample_storage, quantile=quantile)
    print("calculated true domain ", true_domain)
    # true_domain = (0.0001033285847, 0.15110546971700062)
    true_domain = (0.0001033285847, 0.15110546971700062)  # outflow
    # true_domain = (9.585054483236782e-05, 0.01004395027124762)  # cond tn
    print("true domain ", true_domain)
    # true_domain = (3.61218304e-07, 3.645725840000398e-05)
    # true_domain = (4.1042996300000003e-07, 2.9956777900000792e-05)
    # true_domain = (3.61218304e-07, 2.9956777900000792e-05)
    # moments_fn = Legendre(self.n_moments, true_domain)
    moments_fn = Monomial(n_moments, true_domain)  # , ref_domain=true_domain)

    # n_ops = np.array(sample_storage.get_n_ops())
    # print("n ops ", n_ops)
    # print("n ops ", n_ops[:, 0] / n_ops[:, 1])

    print("sample storage n collected ", sample_storage.get_n_collected())

    estimate_obj = mlmc.estimator.Estimate(quantity=q_value, sample_storage=sample_storage, moments_fn=moments_fn)
    estimate_obj_squared = mlmc.estimator.Estimate(quantity=np.power(q_value, 2), sample_storage=sample_storage,
                                                   moments_fn=moments_fn)

    # cons_check_val = mlmc.estimator.consistency_check(q_value, sample_storage=sample_storage)
    # print("consistency check val ", cons_check_val)
    # exit()

    # ###################
    # ###################
    # # Central moments #
    # ###################
    # ###################
    # means = estimate_mean(root_quantity)
    #
    # # moments_fn = Legendre(n_moments, true_domain)
    # moments_fn = Monomial(self.n_moments, true_domain)
    #
    # moments_quantity = moments(root_quantity, moments_fn=moments_fn, mom_at_bottom=True)
    # moments_mean = estimate_mean(moments_quantity)
    #
    # cond_tn_quantity = moments_mean["cond_tn"]
    # time_mean = cond_tn_quantity[1]  # times: [1]
    # location_mean = time_mean['0']  # locations: ['0']
    # # print("location mean ", location_mean)
    # k_xx = location_mean[2]
    # q_value = k_xx
    # print("q_value ", q_value.mean)
    # values_mean = q_value  # result shape: (1, 1)
    # value_mean = values_mean[0]
    # assert value_mean.mean == 1
    #
    # true_domain = [-10, 10]  # keep all values on the original domain
    # central_moments_fn = Monomial(self.n_moments, true_domain, ref_domain=true_domain)
    # central_moments_quantity = moments(root_quantity, moments_fn=central_moments_fn, mom_at_bottom=True)
    # central_moments_mean = estimate_mean(central_moments_quantity)
    #
    # print("central moments mean ", central_moments_mean.mean)
    #
    # ############
    # ############
    # ############

    ###############################
    ###############################
    # Pure data MEAN and VARIANCE #
    ###############################
    ###############################
    squared_root_quantity = np.power(q_value, 2)
    means_root_quantity = estimate_mean(q_value)
    means_squared_root_quantity = estimate_mean(squared_root_quantity)

    fine_means_root_quantity = estimate_mean(q_value, form="fine")
    coarse_means_root_quantity = estimate_mean(q_value, form="coarse")

    print("means_root_quantity.l_means ", means_root_quantity.l_means)
    print("fine_means_root_quantity.l_means ", fine_means_root_quantity.l_means)
    print("coarse_means_root_quantity.l_means ", coarse_means_root_quantity.l_means)

    variance = means_squared_root_quantity.mean - np.power(means_root_quantity.mean, 2)

    print("COLLECTED VALUES MEAN: {}, VARIANCE: {}".format(means_root_quantity.mean, variance))

    # means, vars = estimate_obj.estimate_moments(moments_fn)
    # means_squared, vars = estimate_obj_squared.estimate_moments(moments_fn)
    #
    # print('means ', means)
    # print("means squared ", means_squared)
    #
    # variance = np.squeeze(means_squared)[1] - (np.squeeze(means)[1] ** 2)
    #
    # print("variance ", variance)
    #
    ############
    ############
    ############

    #############
    # subsample #
    #############
    import mlmc.quantity.quantity_estimate as qe
    sample_vector = [10, 10, 10]
    # root_quantity_subsamples = root_quantity.subsample(sample_vector)  # out of [100, 80, 50, 30, 10]
    # moments_quantity = qe.moments(root_quantity_subsamples, moments_fn=moments_fn, mom_at_bottom=True)
    # mult_chunks_moments_mean = estimate_mean(moments_quantity)
    # mult_chunks_length_mean = mult_chunks_moments_mean['length']
    # mult_chunks_time_mean = mult_chunks_length_mean[1]
    # mult_chunks_location_mean = mult_chunks_time_mean['10']
    # mult_chunks_value_mean = mult_chunks_location_mean[0]
    # quantity_subsample = estimate_obj.quantity.select(estimate_obj.quantity.subsample(sample_vec=sample_vector))
    # moments_quantity = qe.moments(quantity_subsample, moments_fn=moments_fn, mom_at_bottom=False)
    # q_mean = qe.estimate_mean(moments_quantity)
    means, vars = estimate_obj.estimate_moments(moments_fn)
    moments_quantity = qe.moments(q_value, moments_fn)
    estimate_obj_moments_quantity = mlmc.estimator.Estimate(quantity=moments_quantity,
                                                            sample_storage=sample_storage, moments_fn=moments_fn)

    print("vars ", vars)
    print('means ', means)

    # moments_fn = self.set_moments(q_value, sample_storage, n_moments=self.n_moments, quantile=quantile)
    # estimate_obj = estimator.Estimate(q_value, sample_storage=sample_storage,
    #                                   moments_fn=moments_fn)

    # New estimation according to already finished samples
    variances, n_ops = estimate_obj.estimate_diff_vars_regression(sample_storage.get_n_collected())

    # n_ops[2] = 200
    # n_ops[1] = 100
    #
    # print("n ops ", n_ops)
    n_estimated = estimator.estimate_n_samples_for_target_variance(target_var, variances, n_ops,
                                                                   n_levels=n_levels)

    # n_ops = [64, 150, 300]
    # n_ops[0] = 85

    # n_ops = [36, 75, 120]

    # n_ops = [36.23067711011784, 848.5677567292186, 3115.1620610555015]

    # n_ops = [50.74231049047417, 150.8056385226611, 300.4481985032301]

    print("level variances ", variances)
    print("n ops ", n_ops)

    print("n estimated ", n_estimated)

    # n_ops = [2.1530759945526734, 424.25148745817296, 2510.9868827356786]
    # n_ops = [10, 25, 120]
    # n_ops = [130, 150]
    print("total cost ", np.sum(n_ops * n_estimated))  # np.array(sample_storage.get_n_collected())))

    # moments_quantity = moments(root_quantity, moments_fn=moments_fn, mom_at_bottom=True)
    # moments_mean = estimate_mean(moments_quantity)
    # conductivity_mean = moments_mean['cond_tn']
    # time_mean = conductivity_mean[1]  # times: [1]
    # location_mean = time_mean['0']  # locations: ['0']
    # values_mean = location_mean[0]  # result shape: (1,)
    # value_mean = values_mean[0]
    # assert value_mean.mean == 1

    l_0_samples = estimate_obj.get_level_samples(level_id=0, n_samples=sample_storage.get_n_collected()[0])

    if n_levels == 1:
        print("l0 var", np.var(l_0_samples))

        print("L0 mean {}".format(np.mean(l_0_samples)))

    elif n_levels == 2:
        l_1_samples = estimate_obj.get_level_samples(level_id=1, n_samples=sample_storage.get_n_collected()[1])
        l_1_samples = np.squeeze(l_1_samples)  # / div_coef

        print("mean l_1_samples fine ", np.mean(l_1_samples[:, 0]))
        print("mean l_1_samples coarse ", np.mean(l_1_samples[:, 1]))

        l_0_samples_moments = estimate_obj_moments_quantity.get_level_samples(level_id=0, n_samples=
        sample_storage.get_n_collected()[0])

        mask = np.any(np.isnan(l_0_samples_moments), axis=0).any(axis=1)
        l_0_samples_moments = l_0_samples_moments[..., ~mask, :]
        l_0_samples_moments = l_0_samples_moments[1:, :, 0]

        # print("l0 samples moments ", l_0_samples_moments.shape)

        l_1_samples_moments = estimate_obj_moments_quantity.get_level_samples(level_id=1, n_samples=
        sample_storage.get_n_collected()[1])
        mask = np.any(np.isnan(l_1_samples_moments), axis=0).any(axis=1)
        l_1_samples_moments = l_1_samples_moments[..., ~mask, :]

        # print("l1 sample momoments shape ", l_1_samples_moments.shape)

        l1_fine_moments = l_1_samples_moments[1:, :, 0]
        l1_coarse_moments = l_1_samples_moments[1:, :, 1]

        # print("l1 fine moments ", l1_fine_moments.shape)
        # print("l1 coarse moments ", l1_coarse_moments)

        l1_fine_samples = l_1_samples[:, 0]
        l1_coarse_samples = l_1_samples[:, 1]

        log = False
        if log:
            l1_fine_samples = np.log10(l1_fine_samples)
            l1_fine_samples = l1_fine_samples[~np.isinf(l1_fine_samples)]

            l1_coarse_samples = np.log10(l1_coarse_samples)
            l1_coarse_samples = l1_coarse_samples[~np.isinf(l1_coarse_samples)]

            l_0_samples = np.log10(l_0_samples)
            l_0_samples = l_0_samples[~np.isinf(l_0_samples)]

        l1_diff = l1_fine_samples - l1_coarse_samples

        # print("var l1_fine_moments ", np.var(l1_fine_moments, axis=1))
        # print("var l1_coarse_moments ", np.var(l1_coarse_moments, axis=1))

        moment_1_cov_matrix = np.cov(l1_fine_moments[0, :], l1_coarse_moments[0, :])
        moment_2_cov_matrix = np.cov(l1_fine_moments[1, :], l1_coarse_moments[1, :])
        print(" moment_1_cov_matrix  ", moment_1_cov_matrix)
        print(" moment_2_cov_matrix  ", moment_2_cov_matrix)
        moment_1_tot_var = moment_1_cov_matrix[0, 0] + moment_1_cov_matrix[1, 1] - 2 * moment_1_cov_matrix[1, 0]
        moment_2_tot_var = moment_2_cov_matrix[0, 0] + moment_2_cov_matrix[1, 1] - 2 * moment_2_cov_matrix[1, 0]

        print("moment 1 tot var ", moment_1_tot_var)
        print("moment 2 tot var ", moment_2_tot_var)

        # print("l1 diff ", l1_diff)
        print("l1 diff var", np.var(l1_diff))
        print("l0 var", np.var(l_0_samples))

        print("L0 mean {}, L1 coarse mean: {}".format(np.mean(l_0_samples), np.mean(l1_coarse_samples)))
        compare_means(l_0_samples, l1_coarse_samples, title="L0 fine mean and L1 coarse mean")

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        axes.hist(l1_coarse_samples, bins=100, density=True, label="L1 coarse")
        axes.hist(np.squeeze(l_0_samples), bins=100, density=True, label="L0 samples", alpha=0.5)
        fig.suptitle("L1 coarse L0 samples, log: {}".format(log))
        fig.legend()
        fig.savefig("L1_coarse_L0_samples.pdf")
        plt.show()
        # print("L1 fine mean {}, L2 coarse mean: {}".format(np.mean(l1_fine_samples), np.mean(l2_coarse_samples)))
    else:
        l_1_samples = estimate_obj.get_level_samples(level_id=1, n_samples=sample_storage.get_n_collected()[1])
        l_1_samples = np.squeeze(l_1_samples)  # / div_coef

        l_0_samples_moments = estimate_obj_moments_quantity.get_level_samples(level_id=0, n_samples=
        sample_storage.get_n_collected()[0])

        mask = np.any(np.isnan(l_0_samples_moments), axis=0).any(axis=1)
        l_0_samples_moments = l_0_samples_moments[..., ~mask, :]
        l_0_samples_moments = l_0_samples_moments[1:, :, 0]

        # print("l0 samples moments ", l_0_samples_moments.shape)

        l_1_samples_moments = estimate_obj_moments_quantity.get_level_samples(level_id=1, n_samples=
        sample_storage.get_n_collected()[1])
        mask = np.any(np.isnan(l_1_samples_moments), axis=0).any(axis=1)
        l_1_samples_moments = l_1_samples_moments[..., ~mask, :]

        # print("l1 sample momoments shape ", l_1_samples_moments.shape)

        l1_fine_moments = l_1_samples_moments[1:, :, 0]
        l1_coarse_moments = l_1_samples_moments[1:, :, 1]

        l_2_samples = estimate_obj.get_level_samples(level_id=2, n_samples=sample_storage.get_n_collected()[2])
        l_2_samples = np.squeeze(l_2_samples)  # / div_coef
        # print("squeezed l1 samples ", l_1_samples)
        # print("l_1_samples[:, 0] ", l_1_samples[:, 0])
        # print("l_1_samples[:, 1] ", l_1_samples[:, 1])

        l_2_samples_moments = estimate_obj_moments_quantity.get_level_samples(level_id=2, n_samples=
        sample_storage.get_n_collected()[2])
        mask = np.any(np.isnan(l_2_samples_moments), axis=0).any(axis=1)
        l_2_samples_moments = l_2_samples_moments[..., ~mask, :]

        # print("l1 sample momoments shape ", l_2_samples_moments.shape)

        l2_fine_moments = l_2_samples_moments[1:, :, 0]
        l2_coarse_moments = l_2_samples_moments[1:, :, 1]

        if n_levels == 4:
            l_3_samples = estimate_obj.get_level_samples(level_id=3, n_samples=sample_storage.get_n_collected()[3])
            l_3_samples = np.squeeze(l_3_samples)

            l3_fine_samples = l_3_samples[:, 0]
            l3_coarse_samples = l_3_samples[:, 1]

            l_3_samples_moments = estimate_obj_moments_quantity.get_level_samples(level_id=3, n_samples=
            sample_storage.get_n_collected()[3])
            mask = np.any(np.isnan(l_3_samples_moments), axis=0).any(axis=1)
            l_3_samples_moments = l_3_samples_moments[..., ~mask, :]
            l3_fine_moments = l_3_samples_moments[1:, :, 0]
            l3_coarse_moments = l_3_samples_moments[1:, :, 1]

        # l_3_samples = estimate_obj.get_level_samples(level_id=3)
        # l_3_samples = np.squeeze(l_3_samples)

        # l1_fine_samples = l_1_samples[1:10, 0]
        # l1_coarse_samples = l_1_samples[1:10, 1]

        l1_fine_samples = l_1_samples[:, 0]
        l1_coarse_samples = l_1_samples[:, 1]

        l2_fine_samples = l_2_samples[:, 0]
        l2_coarse_samples = l_2_samples[:, 1]

        # l3_fine_samples = l_3_samples[:, 0]
        # l3_coarse_samples = l_3_samples[:, 1]

        # l3_diff = l3_fine_samples - l3_coarse_samples
        l2_diff = l2_fine_samples - l2_coarse_samples
        l1_diff = l1_fine_samples - l1_coarse_samples

        log = True
        if log:
            l2_coarse_samples = np.log10(l2_coarse_samples)
            l2_coarse_samples = l2_coarse_samples[~np.isinf(l2_coarse_samples)]

            l1_fine_samples = np.log10(l1_fine_samples)
            l1_fine_samples = l1_fine_samples[~np.isinf(l1_fine_samples)]

            l1_coarse_samples = np.log10(l1_coarse_samples)
            l1_coarse_samples = l1_coarse_samples[~np.isinf(l1_coarse_samples)]

            l_0_samples = np.log10(l_0_samples)
            l_0_samples = l_0_samples[~np.isinf(l_0_samples)]

            # l_0_samples += 0.03

            # l1_coarse_samples = l1_coarse_samples[:100]
            # l_0_samples = l_0_samples[:100]

        print("min l1_fine_samples ", np.min(l1_fine_samples))

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        axes.hist(l2_diff, bins=100, density=True)
        fig.suptitle("L2 diff")
        fig.legend()
        plt.show()

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        axes.hist(l2_coarse_samples, bins=100, density=True, label="L2 coarse")
        axes.hist(l1_fine_samples, bins=100, density=True, label="L1 fine", alpha=0.5)
        fig.suptitle("L2 coarse L1 fine samples, log: {}".format(log))
        fig.legend()
        fig.savefig("L2_coarse_L1_fine_samples.pdf")
        plt.show()

        # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        # axes.hist(l1_fine_samples, bins=100, density=True)
        # fig.suptitle("L1 fine samples, log: {}".format(log))
        # plt.show()

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        axes.hist(l1_coarse_samples, bins=100, density=True, label="L1 coarse")
        axes.hist(np.squeeze(l_0_samples), bins=100, density=True, label="L0 samples", alpha=0.5)
        fig.suptitle("L1 coarse L0 samples, log: {}".format(log))
        fig.legend()
        fig.savefig("L1_coarse_L0_samples.pdf")
        plt.show()

        # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        # axes.hist(np.squeeze(l_0_samples), bins=100, density=True)
        # fig.suptitle("L0 samples, log: {}".format(log))
        # plt.show()

        # print("l1 diff ", l1_diff)
        # print("l2 diff ", l2_diff)

        print("l1 diff var", np.var(l1_diff))
        print("l2 diff var", np.var(l2_diff))

        print("l0 var", np.var(l_0_samples))

        l_0_samples = np.squeeze(l_0_samples)
        l_0_samples = l_0_samples[:np.min([len(l_0_samples), len(l1_coarse_samples)])]
        l1_coarse_samples = l1_coarse_samples[:np.min([len(l_0_samples), len(l1_coarse_samples)])]

        print("L0 mean {}, L1 coarse mean: {}".format(np.mean(l_0_samples), np.mean(l1_coarse_samples)))
        print("L0 std {}, L1 coarse std: {}".format(np.std(l_0_samples), np.std(l1_coarse_samples)))
        print("L1 fine mean {}, L2 coarse mean: {}".format(np.mean(l1_fine_samples), np.mean(l2_coarse_samples)))
        print("L1 fine std {}, L2 coarse std: {}".format(np.std(l1_fine_samples), np.std(l2_coarse_samples)))

        print("l0 samples shape ", l_0_samples.shape)

        print("np.min([len(l_0_samples), len(l1_coarse_samples)]) ",
              np.min([len(l_0_samples), len(l1_coarse_samples)]))

        compare_means(l_0_samples, l1_coarse_samples, title="L0 fine mean and L1 coarse mean")
        compare_means(l1_fine_samples, l2_coarse_samples, title="L1 fine mean and L2 coarse mean")

        if n_levels == 4:
            compare_means(l2_fine_samples, l3_coarse_samples, title="L2 fine mean and L3 coarse mean")

        print("============== MOMENTS MEAN COMPARISON ===================")

        compare_means(l_0_samples_moments[0], l1_coarse_moments[0], title="Moments 0 L0 fine and L1 coarse ")
        compare_means(l_0_samples_moments[1], l1_coarse_moments[1], title="Moments 1 L0 fine and L1 coarse ")
        compare_means(l1_fine_moments[0], l2_coarse_moments[0], title="Moments 0 L1 fine and L2 coarse ")
        compare_means(l1_fine_moments[1], l2_coarse_moments[1], title="Moments 1 L1 fine and L2 coarse ")

        if log:
            m0_l1_coarse_moments = np.log10(l1_coarse_moments[0])
            m0_l0_samples_moments = np.log10(np.squeeze(l_0_samples_moments[0]))

            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
            axes.hist(m0_l1_coarse_moments, bins=100, density=True, label="moment 0 L1 coarse")
            axes.hist(m0_l0_samples_moments, bins=100, density=True, label="moment 0 L0 samples", alpha=0.5)
            fig.suptitle("moments 0 L1 coarse L0 samples, log: {}".format(log))
            fig.legend()
            fig.savefig("moments_0_L1_coarse_L0_samples.pdf")
            plt.show()

            print("moments 0 L0 mean {}, L1 coarse mean: {}".format(np.mean(m0_l0_samples_moments),
                                                                    np.mean(m0_l1_coarse_moments)))
            print("moments 0 L0 std {}, L1 coarse std: {}".format(np.std(m0_l0_samples_moments),
                                                                  np.std(m0_l1_coarse_moments)))
            compare_means(m0_l0_samples_moments, m0_l1_coarse_moments,
                               title="Moments 0 L0 fine and L1 coarse ")

    sample_vec = [100, 75, 50, 25]
    # sample_vec = [10, 7, 5]
    # sample_vec = [4462,  213,  164]
    # sample_vec = [250, 100]
    # sample_vec = [10000]
    # sample_vec = [1785,   80,   21]
    bs_n_estimated = estimate_obj.bs_target_var_n_estimated(target_var=target_var, sample_vec=sample_vec,
                                                            n_subsamples=10)
    print("mean bs l vars ", estimate_obj.mean_bs_l_vars)
    print("bs n estimated ", bs_n_estimated)
    print("bs total cost ", np.sum(n_ops * bs_n_estimated))

    ####################
    # Diagnostic plots #
    ####################
    # print("estimate_obj.moments_mean_obj.l_vars ", estimate_obj.moments_mean_obj.l_vars.shape)
    # print("bs means ", estimate_obj.bs_mean)
    # print("mean bs mean ", estimate_obj.mean_bs_mean)
    # print("mean bs var ", estimate_obj.mean_bs_var)
    # dp.log_var_per_level(estimate_obj.moments_mean_obj.l_vars, moments=range(1, n_moments))
    # dp.log_mean_per_level(estimate_obj.moments_mean_obj.l_means, moments=range(1, n_moments))
    # dp.sample_cost_per_level(n_ops)
    # dp.variance_to_cost_ratio(estimate_obj.moments_mean_obj.l_vars, n_ops, moments=range(1, n_moments))
    # dp.kurtosis_per_level(estimate_obj.moments_mean_obj.mean, estimate_obj.moments_mean_obj.l_means,
    #                       moments=range(n_moments))

    # from mlmc.plot.plots import Distribution
    # distr_obj, result, _, _ = estimate_obj.construct_density()
    # distr_plot = Distribution(title="distributions", error_plot=None)
    # distr_plot.add_distribution(distr_obj)
    #
    # samples = estimate_obj.get_level_samples(level_id=0)[..., 0]
    # #print("samples ", samples)
    # distr_plot.add_raw_samples(np.squeeze(samples))  # add histogram
    # distr_plot.show()

    return level_parameters, estimate_obj.moments_mean_obj, n_ops


##########
## 2LMC ##
##########
mlmc_file_paths = [
"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_75_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept/mlmc_2.hdf5",
"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_8_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept/mlmc_2.hdf5",
"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_10_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept/mlmc_2.hdf5",
"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_10_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_2/mlmc_2.hdf5",
"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_125_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept/mlmc_2.hdf5",
"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_15_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept/mlmc_2.hdf5",
"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_20_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept/mlmc_2.hdf5",
"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_40_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept/mlmc_2.hdf5",
"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_40_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_2/mlmc_2.hdf5"
]

##########
## 3LMC ##
##########
mlmc_file_paths = ["/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_75_1125_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest/mlmc_3.hdf5",
    "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_8_16_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest/mlmc_3.hdf5",
                   "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_10_20_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest/mlmc_3.hdf5",
                   "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_10_30_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest/mlmc_3.hdf5"
                   "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_125_3125_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest/mlmc_3.hdf5",

                   ]


reference_mc = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/1LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ref_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_h_5_seed_12345_SRF_direct_gen/mlmc_1.hdf5"


if __name__ == "__main__":
    all_level_parameters = []
    all_moments_mean_obj = []
    all_n_ops = []

    ref_level_parameters, ref_moments_mean_obj, ref_n_ops = process(reference_mc, target_var=5e-8)



    for mlmc_file_path in mlmc_file_paths:
        print("##################################################")
        print("##################################################")
        print("##################################################")
        level_parameters, moments_mean_obj, n_ops = process(mlmc_file_path, target_var=1e-6)

        all_level_parameters.append(level_parameters)
        all_moments_mean_obj.append(moments_mean_obj)
        all_n_ops.append(n_ops)

    plot_variance_mesh_size(all_level_parameters, all_moments_mean_obj, all_n_ops, moments=[1])
    plot_cost_mesh_size(all_level_parameters, all_n_ops)
    plot_bias(all_level_parameters,all_moments_mean_obj, ref_moments_mean_obj)

