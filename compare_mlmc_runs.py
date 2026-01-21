import os
import numpy as np
import mlmc
from mlmc.sample_storage_hdf import SampleStorageHDF
from mlmc.moments import Legendre, Monomial
from mlmc.quantity.quantity import make_root_quantity
from mlmc.quantity.quantity_estimate import estimate_mean, cache_clear, moments
from mlmc import estimator
from mlmc.plot import diagnostic_plots as dp
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib.pyplot as plt
from bgem.upscale import voigt_to_tn

import zarr
import glob
import seaborn as sns
from matplotlib import ticker

from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit

#rng = np.random.default_rng(123)
np.random.seed(11) # 42

# Style
sns.set_style("white")
sns.set_context("paper", font_scale=1.2)


def plot_bias(list_l_params, list_all_moments_mean_obj, ref_moments_mean_obj, levels=None, moment=1, err_l_vars=None, label_names=[], fontsize=17, fontsize_ticks=15):
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


def plot_variance_mesh_size(list_l_params, list_all_l_vars, levels=None, moments=[0], err_l_vars=None, label_names=[], fontsize=17, fontsize_ticks=15):
    """
    Plot log2 variance vs level and fit slope to estimate beta.

    l_vars: array shape (n_levels, n_moments) with variance per level & moment
    levels: array of level indices (default = np.arange(n_levels))
    moments: list of moment indices to plot
    err_l_vars: optional errors for l_vars
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    print("font size ", fontsize)
    print("fontsize_ticks ", fontsize_ticks)


    palette = sns.color_palette("Set2", n_colors=len(list_l_params))

    for i in range(len(list_l_params)):
        l_params = np.squeeze(list_l_params[i])
        print("l_params ", l_params)
        l_vars = list_all_l_vars[i]
        print("l vars ", l_vars)
        # n_levels = l_vars.shape[0]
        # if levels is None:
        #     levels = np.arange(n_levels)

        print("label_names ", label_names)

        #for m in moments:
        y = np.log2(l_vars)#[:, m])
        print("y ", y)
        ax.plot(l_params, y, 'o', color=palette[i], markersize=6)

        slope, intercept = np.polyfit(l_params, y, 1)
        beta = slope
        ax.plot(l_params, slope*l_params + intercept, '--', color=palette[i],
                label=label_names[i] + ", " + r'$\beta≈$' + f'{beta:.2f}', linewidth=2.1)

    ax.set_ylabel(r'$\log_2 \, V_l$', fontsize=fontsize)
    ax.set_xlabel('$h_l$', fontsize=fontsize)

    ax.yaxis.get_offset_text().set_fontsize(fontsize_ticks)
    ax.xaxis.get_offset_text().set_fontsize(fontsize_ticks)
    ax.legend(fontsize=fontsize_ticks)
    #ax.grid(True, which="both")
    plt.tight_layout()
    plt.savefig("variance_mesh_size.pdf", dpi=300, bbox_inches="tight")
    plt.show()


def plot_cost_mesh_size(list_l_params, list_costs, levels=None, label_names=[], fontsize=17, fontsize_ticks=15):
    """
    Plot log2 cost per level and fit slope to estimate gamma.

    costs: array of cost per sample at each level
    levels: optional array of level indices (default = 0,1,...)
    """

    # n_levels = len(costs)
    # if levels is None:
    #     levels = np.arange(n_levels)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Style
    # sns.set_style("white")
    # sns.set_context("paper", font_scale=1.2)
    palette = sns.color_palette("Set2", n_colors=len(list_l_params))

    for i in range(len(list_l_params)):
        l_params = np.squeeze(list_l_params[i])
        costs = list_costs[i]
        y = np.log2(costs)
        slope, intercept = np.polyfit(l_params, y, 1)
        gamma = -slope

        ax.plot(l_params, y, 'o', color=palette[i], markersize=8)
        ax.plot(l_params, slope * l_params + intercept, '--', color=palette[i],
                label=label_names[i] + ", " + r'$\gamma≈$' + f'{gamma:.2f}', linewidth=2.2)

    ax.set_ylabel(r'$\log_2 \, C_l [sec]$', fontsize=fontsize)
    ax.set_xlabel('$h_l$', fontsize=fontsize)

    ax.yaxis.get_offset_text().set_fontsize(fontsize_ticks)
    ax.xaxis.get_offset_text().set_fontsize(fontsize_ticks)
    ax.legend(fontsize=fontsize_ticks)
    # ax.grid(True, which="both")
    plt.tight_layout()
    plt.savefig("cost_mesh_size.pdf", dpi=300, bbox_inches="tight")
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


def get_n_ops(mlmc_hdf_file):
    sample_storage = SampleStorageHDF(file_path=mlmc_hdf_file)
    return sample_storage.get_n_ops()


def get_level_samples(quantity, sample_storage, n_estimated, rnd_idx=None):
    ###############
    # GET SAMPLES #
    ###############
    samples_level_0 = []
    samples_level_1 = []
    samples_level_2 = []
    samples_level_3 = []

    write_rnd_idx = False
    if rnd_idx is None:
        rnd_idx = []
        write_rnd_idx = True

    if len(n_estimated) == 1:
        n_est_level_0 = int(n_estimated[0])
        chunk_spec = next(sample_storage.chunks(level_id=0, n_samples=sample_storage.get_n_collected()[0]))
        samples_level_0 = np.squeeze(quantity.samples(chunk_spec=chunk_spec))

        print("np.max(samples_level_0) ", np.max(samples_level_0))
        max_idx = np.argmax(samples_level_0)
        print("index of max:", max_idx)


        if samples_level_0.shape[-1] < n_est_level_0:
            print("INSUFFICIENT NUMBER OF SAMPLES: len(samples_level_0) {} < n_est_level_0 {}".format(
                len(samples_level_0), n_est_level_0))
            n_est_level_0 = np.min([samples_level_0.shape[-1], n_est_level_0])

        # start = np.random.randint(0, len(samples_level_0) - n_est_level_0 + 1)
        # samples_level_0 = samples_level_0[start:start + n_est_level_0]

        if not write_rnd_idx:
            idx = rnd_idx[0]
        else:
            idx = np.random.choice(samples_level_0.shape[-1], size=n_est_level_0, replace=False)
            rnd_idx.append(idx)



        samples_level_0 = samples_level_0[..., idx]


    if len(n_estimated) in [2, 3, 4]:
        n_est_level_0 = int(n_estimated[0])
        chunk_spec = next(sample_storage.chunks(level_id=0, n_samples=sample_storage.get_n_collected()[0]))
        samples_level_0 = np.squeeze(quantity.samples(chunk_spec=chunk_spec))

        if len(samples_level_0) < n_est_level_0:
            print("INSUFFICIENT NUMBER OF SAMPLES: len(samples_level_0) {} < n_est_level_0 {}".format(
                len(samples_level_0), n_est_level_0))
            n_est_level_0 = np.min([len(samples_level_0), n_est_level_0])

        # start = np.random.randint(0, len(samples_level_0) - n_est_level_0 + 1)
        # samples_level_0 = samples_level_0[start:start + n_est_level_0]

        if not write_rnd_idx:
            idx = rnd_idx[0]
        else:
            idx = np.random.choice(len(samples_level_0), size=n_est_level_0, replace=False)
            rnd_idx.append(idx)

        samples_level_0 = samples_level_0[idx]

        ####
        # Level 1
        ####
        n_est_level_1 = int(n_estimated[1])
        chunk_spec = next(sample_storage.chunks(level_id=1, n_samples=sample_storage.get_n_collected()[1]))
        samples_level_1 = np.squeeze(quantity.samples(chunk_spec=chunk_spec))

        if len(samples_level_1) < n_est_level_1:
            print("INSUFFICIENT NUMBER OF SAMPLES: len(samples_level_1) {} < n_est_level_1 {}".format(
                len(samples_level_1), n_est_level_1))
        n_est_level_1 = np.min([len(samples_level_1), n_est_level_1])

        # start = np.random.randint(0, len(samples_level_1) - n_est_level_1 + 1)
        # samples_level_1 = samples_level_1[start:start + n_est_level_1]
        if not write_rnd_idx:
            idx = rnd_idx[1]
        else:
            idx = np.random.choice(len(samples_level_1), size=n_est_level_1, replace=False)
            rnd_idx.append(idx)
        samples_level_1 = samples_level_1[idx]

    if len(n_estimated) in [3, 4]:
        ####
        # Level 2
        ####
        n_est_level_2 = int(n_estimated[2])
        chunk_spec = next(sample_storage.chunks(level_id=2, n_samples=sample_storage.get_n_collected()[2]))
        samples_level_2 = np.squeeze(quantity.samples(chunk_spec=chunk_spec))

        if len(samples_level_2) < n_est_level_2:
            print("INSUFFICIENT NUMBER OF SAMPLES: len(samples_level_2) {} < n_est_level_2 {}".format(
                len(samples_level_2), n_est_level_2))
            n_est_level_2 = np.min([len(samples_level_2), n_est_level_2])

        # start = np.random.randint(0, len(samples_level_2) - n_est_level_2 + 1)
        # samples_level_2 = samples_level_2[start:start + n_est_level_2]
        if not write_rnd_idx:
            idx = rnd_idx[2]
        else:
            idx = np.random.choice(len(samples_level_2), size=n_est_level_2, replace=False)
            rnd_idx.append(idx)
        samples_level_2 = samples_level_2[idx]

    if len(n_estimated) in [4]:
        ####
        # Level 3
        ####
        n_est_level_3 = int(n_estimated[3])
        chunk_spec = next(sample_storage.chunks(level_id=3, n_samples=sample_storage.get_n_collected()[3]))
        samples_level_3 = np.squeeze(quantity.samples(chunk_spec=chunk_spec))

        if len(samples_level_3) < n_est_level_3:
            print("INSUFFICIENT NUMBER OF SAMPLES: len(samples_level_3) {} < n_est_level_3 {}".format(
                len(samples_level_3), n_est_level_3))
            n_est_level_3 = np.min([len(samples_level_3), n_est_level_3])

        # start = np.random.randint(0, len(samples_level_3) - n_est_level_3 + 1)
        # samples_level_3 = samples_level_3[start:start + n_est_level_3]

        if not write_rnd_idx:
            idx = rnd_idx[3]
        else:
            idx = np.random.choice(len(samples_level_3), size=n_est_level_3, replace=False)
            rnd_idx.append(idx)
        samples_level_3 = samples_level_3[idx]

    return rnd_idx, [samples_level_0, samples_level_1, samples_level_2, samples_level_3]

def process(mlmc_file_path, mlmc_info={}, target_var=1e-6):
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

    num_quantities = 6

    rnd_idx = None
    initial_guess_n_estimated = []

    print("----------------------------------")
    print(" INITIAL GUESS")
    print("----------------------------------")
    for q_idx in range(num_quantities):
        print("-----------------------")
        print(" QUANTITY {}".format(q_idx))
        print("-----------------------")
        q_value = location_mean[q_idx]
        print("q_value ", q_value)

        print("sample storage n collected ", sample_storage.get_n_collected())

        cost_file_n_ops = get_n_ops(mlmc_info["cost_file"])
        print("cost_file_n_ops ", cost_file_n_ops)

        n0 = 100
        nL = 10
        sample_vec_0 = np.round(np.exp2(np.linspace(np.log2(n0), np.log2(nL), n_levels))).astype(int)
        if "init_sample_vec" in mlmc_info:
            print("sample vec 0 ", sample_vec_0)
            print('init sample vec ', np.array(mlmc_info["init_sample_vec"]).astype(int))
            sample_vec = np.max([np.array(mlmc_info["init_sample_vec"]).astype(int), sample_vec_0], axis=0)
            print("sample vec ", sample_vec)
        else:
            sample_vec = [500]

        rnd_idx, level_samples = get_level_samples(q_value, sample_storage, sample_vec, rnd_idx)

        var_est, level_variances, _, _, _,  = mean_var_at_level_quantity(level_samples, n_estimated=sample_vec)
        print("level variances ", level_variances)
        print("level variances ", level_variances.shape)

        print("var est ", var_est)
        initial_guess_n_estimated_quantity = estimator.estimate_n_samples_for_target_variance(target_var, level_variances,
                                                                                     cost_file_n_ops, n_levels=n_levels)

        initial_guess_n_estimated.append(initial_guess_n_estimated_quantity)


    print("initial_guess_n_estimated ", initial_guess_n_estimated)


    initial_guess_n_estimated = np.max([sample_vec] + initial_guess_n_estimated, axis=0)
    print("initial_guess_n_estimated ", initial_guess_n_estimated)

    rnd_idx = None
    refined_guess_n_estimated = []
    for q_idx in range(num_quantities):
        print("-----------------------")
        print(" QUANTITY {}".format(q_idx))
        print("-----------------------")
        q_value = location_mean[q_idx]

        rnd_idx, level_samples = get_level_samples(q_value, sample_storage, initial_guess_n_estimated, rnd_idx)

        var_est, level_variances, _, _, _, = mean_var_at_level_quantity(level_samples, n_estimated=initial_guess_n_estimated)
        print("level variances ", level_variances.shape)

        print("var est ", var_est)
        refined_guess_n_estimated_quantity = estimator.estimate_n_samples_for_target_variance(target_var,
                                                                                              level_variances,
                                                                                              cost_file_n_ops,
                                                                                              n_levels=n_levels)

        refined_guess_n_estimated.append(refined_guess_n_estimated_quantity)

    print("refined guess n estimated ", refined_guess_n_estimated)
    refined_guess_n_estimated = np.max([sample_vec] + refined_guess_n_estimated, axis=0)
    print("refined_guess_n_estimated ", refined_guess_n_estimated)

    rnd_idx = None
    final_guess_n_estimated = []
    for q_idx in range(num_quantities):
        print("-----------------------")
        print(" QUANTITY {}".format(q_idx))
        print("-----------------------")
        q_value = location_mean[q_idx]

        rnd_idx, level_samples = get_level_samples(q_value, sample_storage, refined_guess_n_estimated, rnd_idx)

        var_est, level_variances, _, _, _, = mean_var_at_level_quantity(level_samples,
                                                                        n_estimated=refined_guess_n_estimated)
        print("level variances ", level_variances)

        print("var est ", var_est)
        final_guess_n_estimated_quantity = estimator.estimate_n_samples_for_target_variance(target_var,
                                                                                              level_variances,
                                                                                              cost_file_n_ops,
                                                                                              n_levels=n_levels)
        final_guess_n_estimated.append(final_guess_n_estimated_quantity)

    print("final_guess_n_estimated ", final_guess_n_estimated)
    final_guess_n_estimated = np.max([sample_vec] + final_guess_n_estimated, axis=0)
    print("final_guess_n_estimated ", final_guess_n_estimated)


    print("----------------------------------")
    print(" FINAl GUESS")
    print("----------------------------------")

    rnd_idx = None
    final_total_mean = []
    final_total_var = []
    final_estimator_variance = []
    final_mean_diff_on_levels = []
    final_var_diff_on_levels = []
    final_finest_moments_mean = []
    final_finest_moments_var = []
    for q_idx in range(num_quantities):
        print("-----------------------")
        print(" QUANTITY {}".format(q_idx))
        print("-----------------------")
        q_value = location_mean[q_idx]

        rnd_idx, level_samples = get_level_samples(q_value, sample_storage, final_guess_n_estimated, rnd_idx)

        final_variances, final_level_variances, moments_diffs_on_levels, finest_moments, total_moments = mean_var_at_level_quantity(level_samples,
                                                                        n_estimated=final_guess_n_estimated)


        final_estimator_variance.append(final_variances)

        final_mean_diff_on_levels.append(moments_diffs_on_levels[0])
        final_var_diff_on_levels.append(moments_diffs_on_levels[1])

        final_finest_moments_mean.append(finest_moments[0])
        final_finest_moments_var.append(finest_moments[1])

        # print("FINAL skewness diffs on levels ", moments_diffs_on_levels[2])
        # print("FINAL kurtosis diffs on levels ", moments_diffs_on_levels[3])
        # print("finest moments skewness ", finest_moments[2])
        # print("finest moments kurtosis ", finest_moments[3])


        #print("level variances ", final_level_variances)


        final_total_mean.append(np.sum(moments_diffs_on_levels[0], axis=0) + finest_moments[0])
        final_total_var.append(np.sum(moments_diffs_on_levels[1], axis=0) + finest_moments[1])

    print("FINAL mean diffs on levels ", final_mean_diff_on_levels)
    print("FINAL var diffs on levels ", final_var_diff_on_levels)

    print("FINAL finest moments mean ", final_finest_moments_mean)
    print("FINAL finest moments var ", final_finest_moments_var)

    print("FINAL ESTIMATOR VARIANCE ", final_estimator_variance)

    if len(final_guess_n_estimated) == 1:
        eq_cond_tn = voigt_to_tn(np.array([final_finest_moments_mean]))
    else:
        eq_cond_tn = voigt_to_tn(np.array([final_total_mean]))

    print("EVALS ", np.linalg.eigvals(eq_cond_tn))

    print("FINAL total MEAN ", final_total_mean)
    print("FINAL total VAR ", final_total_var)

    print("FINAL n estimated ", final_guess_n_estimated)
    print("cost_file_n_ops ", cost_file_n_ops)
    print("FINAL total cost ",
          np.sum(cost_file_n_ops * final_guess_n_estimated))  # np.array(sample_storage.get_n_collected())))



    #final_variances, final_level_variances, moments_diffs_on_levels, finest_moments, total_moments = mean_var_at_level_quantity(q_value, sample_storage, final_n_estimated)


    # variances, n_ops = estimate_obj_subsample.estimate_diff_vars_regression(n_estimated)
    # print("final n estimated ", final_n_estimated)
    # q_value_subsample = q_value.subsample(np.array(final_n_estimated))
    # true_domain_subsample = mlmc.estimator.Estimate.estimate_domain(q_value_subsample, sample_storage, quantile=quantile)
    # moments_fn_subsample = Monomial(n_moments, true_domain_subsample)
    # moments_fn_subsample = moments_fn
    # estimate_obj_subsample = mlmc.estimator.Estimate(quantity=q_value_subsample, sample_storage=sample_storage,
    #                                                  moments_fn=moments_fn_subsample)
    # means, vars = estimate_obj_subsample.estimate_moments(moments_fn_subsample)
    # moments_quantity = qe.moments(q_value_subsample, moments_fn_subsample)
    # estimate_obj_moments_quantity = mlmc.estimator.Estimate(quantity=moments_quantity,
    #                                                         sample_storage=sample_storage,
    #                                                         moments_fn=moments_fn_subsample)

    # print("level variances ", final_level_variances)
    # print("FINAL n estimated ", final_n_estimated)
    # print("cost_file_n_ops ", cost_file_n_ops)
    # print("FINAL total cost ", np.sum(cost_file_n_ops * final_n_estimated))  # np.array(sample_storage.get_n_collected())))
    # print("FINAL total MEAN ", np.sum(moments_diffs_on_levels[0], axis=0) + finest_moments[0])
    # print("FINAL total VAR ", np.sum(moments_diffs_on_levels[1], axis=0) + finest_moments[1])
    # print("FINAL total SKEWNESS ", np.sum(moments_diffs_on_levels[2], axis=0) + finest_moments[2])
    # print("FINAL total KURTOSIS ", np.sum(moments_diffs_on_levels[3], axis=0) + finest_moments[3])
    # print("FINAL MOMENTS: [{}, {}, {}, {}]".format(np.sum(moments_diffs_on_levels[0], axis=0) + finest_moments[0],
    #                                                np.sum(moments_diffs_on_levels[1], axis=0) + finest_moments[1],
    #                                                np.sum(moments_diffs_on_levels[2], axis=0) + finest_moments[2],
    #                                                np.sum(moments_diffs_on_levels[3], axis=0) + finest_moments[3]
    #                                                ))

    #variances, n_ops = estimate_obj_subsample.estimate_diff_vars_regression(n_estimated)

    print("------------------------------------")
    print("------------------------------------")

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

    return level_parameters, final_level_variances, cost_file_n_ops


def mean_at_level(estimate_obj_moments_quantity, n_estimated):
    print("n estimated ", n_estimated)
    cache_clear()

    if len(n_estimated) == 2:
        n_est_level_0 = int(n_estimated[0])
        sample_moments_level_0 = estimate_obj_moments_quantity.get_level_samples(level_id=0, n_samples=n_est_level_0)
        print("sample_moments_level_0.shape ", sample_moments_level_0.shape)
        sample_moments_mean_level_0 = np.squeeze(np.nanmean(sample_moments_level_0, axis=1))

        n_est_level_1 = int(n_estimated[1])
        print("n_est_level_1 ", n_est_level_1)
        sample_moments_level_1 = estimate_obj_moments_quantity.get_level_samples(level_id=1, n_samples=n_est_level_1)
        print("sample moments level 1 ", sample_moments_level_1)
        print("sample_moments_level_1.shape ", sample_moments_level_1.shape)
        sample_moments_mean_level_1 = np.nanmean(sample_moments_level_1, axis=1)
        print("sample_moments_mean_level_1 ", sample_moments_mean_level_1)
        sample_moments_mean_level_1_fine = sample_moments_mean_level_1[..., 0]
        sample_moments_mean_level_1_coarse = sample_moments_mean_level_1[..., 1]

        print("sample_moments_mean_level_0 ", sample_moments_mean_level_0)
        print("sample_moments_mean_level_1_fine ", sample_moments_mean_level_1_fine)
        print("sample_moments_mean_level_1_coarse ", sample_moments_mean_level_1_coarse)

        print("sample_moments_mean_level_0 - sample_moments_mean_level_1_coarse ", sample_moments_mean_level_0 - sample_moments_mean_level_1_coarse)


def estimate_domain(all_samples, quantile=None):
    """
    Estimate moments domain from MLMC samples.
    :param quantity: mlmc.quantity.Quantity instance, represents the real quantity
    :param sample_storage: mlmc.sample_storage.SampleStorage instance, provides all the samples
    :param quantile: float in interval (0, 1), None means whole sample range
    :return: lower_bound, upper_bound
    """
    ranges = []
    if quantile is None:
        quantile = 0.01

    for level_samples in all_samples:
        if len(level_samples) == 0:
            continue
        fine_samples = level_samples[..., 0]  # Fine samples at level 0
        fine_samples = np.squeeze(fine_samples)
        #print("fine samples ", fine_samples)
        fine_samples = fine_samples[~np.isnan(fine_samples)]  # remove NaN
        print("fine samples.shape ", fine_samples.shape)
        ranges.append(np.percentile(fine_samples, [100 * quantile, 100 * (1 - quantile)]))

    ranges = np.array(ranges)
    return np.min(ranges[:, 0]), np.max(ranges[:, 1])


def mean_at_level_quantity(quantity, sample_storage, n_estimated):

    print("n estimated ", n_estimated)

    ###############
    # GET SAMPLES #
    ###############
    samples_level_0 = []
    samples_level_1 = []
    samples_level_2 = []
    samples_level_3 = []
    if len(n_estimated) in [2, 3, 4]:
        n_est_level_0 = int(n_estimated[0])
        chunk_spec = next(sample_storage.chunks(level_id=0, n_samples=sample_storage.get_n_collected()[0]))
        samples_level_0 = np.squeeze(quantity.samples(chunk_spec=chunk_spec))

        if len(samples_level_0) < n_est_level_0:
            print("INSUFFICIENT NUMBER OF SAMPLES: len(samples_level_0) {} < n_est_level_0 {}".format(
                len(samples_level_0), n_est_level_0))
            n_est_level_0 = np.min([len(samples_level_0), n_est_level_0])

        start = np.random.randint(0, len(samples_level_0) - n_est_level_0 + 1)
        samples_level_0 = samples_level_0[start:start + n_est_level_0]

        ####
        # Level 1
        ####
        n_est_level_1 = int(n_estimated[1])
        chunk_spec = next(sample_storage.chunks(level_id=1, n_samples=sample_storage.get_n_collected()[1]))
        samples_level_1 = np.squeeze(quantity.samples(chunk_spec=chunk_spec))

        if len(samples_level_1) < n_est_level_1:
            print("INSUFFICIENT NUMBER OF SAMPLES: len(samples_level_1) {} < n_est_level_1 {}".format(
                len(samples_level_1), n_est_level_1))
        n_est_level_1 = np.min([len(samples_level_1), n_est_level_1])

        start = np.random.randint(0, len(samples_level_1) - n_est_level_1 + 1)
        samples_level_1 = samples_level_1[start:start + n_est_level_1]

    if len(n_estimated) in [3, 4]:
        ####
        # Level 2
        ####
        n_est_level_2 = int(n_estimated[2])
        chunk_spec = next(sample_storage.chunks(level_id=2, n_samples=sample_storage.get_n_collected()[2]))
        samples_level_2 = np.squeeze(quantity.samples(chunk_spec=chunk_spec))

        if len(samples_level_2) < n_est_level_2:
            print("INSUFFICIENT NUMBER OF SAMPLES: len(samples_level_2) {} < n_est_level_2 {}".format(len(samples_level_2), n_est_level_2))
            n_est_level_2 = np.min([len(samples_level_2), n_est_level_2])

        start = np.random.randint(0, len(samples_level_2) - n_est_level_2 + 1)
        samples_level_2 = samples_level_2[start:start + n_est_level_2]

    if len(n_estimated) in [4]:
        ####
        # Level 3
        ####
        n_est_level_3 = int(n_estimated[3])
        chunk_spec = next(sample_storage.chunks(level_id=3, n_samples=sample_storage.get_n_collected()[3]))
        samples_level_3 = np.squeeze(quantity.samples(chunk_spec=chunk_spec))

        if len(samples_level_3) < n_est_level_3:
            print("INSUFFICIENT NUMBER OF SAMPLES: len(samples_level_3) {} < n_est_level_3 {}".format(len(samples_level_3), n_est_level_3))
            n_est_level_3 = np.min([len(samples_level_3), n_est_level_3])

        start = np.random.randint(0, len(samples_level_3) - n_est_level_3 + 1)
        samples_level_3 = samples_level_3[start:start + n_est_level_3]

    ########################
    # GET MOMENTS FUNCTION #
    ########################
    quantile = 1e-3
    n_moments = 3
    all_samples = [samples_level_0, samples_level_1, samples_level_2, samples_level_3]
    true_domain = estimate_domain(all_samples, quantile=quantile)
    print("true domain ", true_domain)
    moments_fn = Monomial(n_moments, true_domain, ref_domain=true_domain)

    total_var = 0
    level_variances = []
    mean_diffs_on_levels = []
    if len(n_estimated) in [2, 3, 4]:
        ####
        # Level 0
        ####
        moments_level_0 = moments_fn(samples_level_0)
        sample_moments_mean_level_0 = np.nanmean(moments_level_0, axis=0)
        sample_moments_var_level_0 = np.nanvar(moments_level_0, axis=0)

        ####
        # Level 1
        ####
        moments_level_1 = moments_fn(samples_level_1)
        moments_mean_level_1 = np.nanmean(moments_level_1, axis=0)
        moments_mean_level_1_fine = moments_mean_level_1[0]
        moments_mean_level_1_coarse = moments_mean_level_1[1]

        print("moments_mean_level_1_fine ", moments_mean_level_1_fine.shape)
        print("moments_mean_level_1_coarse ", moments_mean_level_1_coarse)

        level_variances.append(sample_moments_var_level_0)
        level_variances.append(np.nanvar(moments_level_1[:, 0,: ]-moments_level_1[:, 1, :], axis=0))

        print("sample_moments_var_level_0 ", sample_moments_var_level_0)

        total_var = sample_moments_var_level_0 / len(moments_level_0) + \
                    (np.nanvar(moments_level_1[:, 0, : ]-moments_level_1[:, 1, :], axis=0)) / len(moments_level_1)

        moments_mean_level_0_fine_1_coarse_diff = sample_moments_mean_level_0 - moments_mean_level_1_coarse

        mean_diffs_on_levels.append(moments_mean_level_0_fine_1_coarse_diff)

        total_mean = moments_mean_level_0_fine_1_coarse_diff
        finest_mean = moments_mean_level_1_fine

        print("#########################")
        print("Covariance between levels")
        print("#########################")

        # level_1_diff = moments_level_1[:, 0,: ] - moments_level_1[:, 1,: ]
        # print("level 1 diff", level_1_diff.shape)
        # print("moments_level_0 ", moments_level_0.shape)
        #
        # cov_level_0_level_1 = np.cov(moments_level_0[:, 0], level_1_diff[:, 0])
        # print("cov level 0 level 1 ", cov_level_0_level_1)

    if len(n_estimated) in [3, 4]:
        ####
        # Level 2
        ####
        # n_est_level_2 = int(n_estimated[2])
        # chunk_spec = next(sample_storage.chunks(level_id=2, n_samples=sample_storage.get_n_collected()[2]))
        # samples_level_2 = np.squeeze(quantity.samples(chunk_spec=chunk_spec))
        #
        # if len(samples_level_2) < n_est_level_2:
        #     print("INSUFFICIENT NUMBER OF SAMPLES: len(samples_level_2) {} < n_est_level_2 {}".format(len(samples_level_2), n_est_level_2))
        #     n_est_level_2 = np.min([len(samples_level_2), n_est_level_2])
        #
        # start = np.random.randint(0, len(samples_level_2) - n_est_level_2 + 1)
        # samples_level_2 = samples_level_2[start:start + n_est_level_2]

        moments_level_2 = moments_fn(samples_level_2)

        moments_mean_level_2 = np.nanmean(moments_level_2, axis=0)
        moments_mean_level_2_fine = moments_mean_level_2[0]
        moments_mean_level_2_coarse = moments_mean_level_2[1]


        total_var += (np.nanvar(moments_level_2[:, 0,: ] - moments_level_2[:, 1,: ],axis=0)) / len(moments_level_2)

        level_variances.append(np.nanvar(moments_level_2[:, 0, :]-moments_level_2[:, 1, :], axis=0))

        moments_mean_level_1_fine_2_coarse_diff = moments_mean_level_1_fine - moments_mean_level_2_coarse

        mean_diffs_on_levels.append(moments_mean_level_1_fine_2_coarse_diff)

        total_mean += moments_mean_level_1_fine_2_coarse_diff
        finest_mean = moments_mean_level_2_fine

    if len(n_estimated) in [4]:
        ####
        # Level 3
        ####
        # n_est_level_3 = int(n_estimated[3])
        # chunk_spec = next(sample_storage.chunks(level_id=3, n_samples=sample_storage.get_n_collected()[3]))
        # samples_level_3 = np.squeeze(quantity.samples(chunk_spec=chunk_spec))
        #
        # if len(samples_level_3) < n_est_level_3:
        #     print("INSUFFICIENT NUMBER OF SAMPLES: len(samples_level_3) {} < n_est_level_3 {}".format(len(samples_level_3), n_est_level_3))
        #     n_est_level_3 = np.min([len(samples_level_3), n_est_level_3])
        #
        # start = np.random.randint(0, len(samples_level_3) - n_est_level_3 + 1)
        # samples_level_3 = samples_level_3[start:start + n_est_level_3]

        moments_level_3 = moments_fn(samples_level_3)
        moments_mean_level_3 = np.nanmean(moments_level_3, axis=0)
        moments_mean_level_3_fine = moments_mean_level_3[0]
        moments_mean_level_3_coarse = moments_mean_level_3[1]

        total_var += (np.nanvar(moments_level_3[:, 0, :] - moments_level_3[:, 1, :], axis=0)) / len(moments_level_3)

        level_variances.append(np.nanvar(moments_level_3[:, 0, :] - moments_level_3[:, 1, :], axis=0))

        moments_mean_level_2_fine_3_coarse_diff = moments_mean_level_2_fine - moments_mean_level_3_coarse

        total_mean += moments_mean_level_2_fine_3_coarse_diff
        finest_mean = moments_mean_level_3_fine

        mean_diffs_on_levels.append(moments_mean_level_2_fine_3_coarse_diff)

    print("total var ", total_var)
    print("level variances ", level_variances)

    return total_var, np.array(level_variances), mean_diffs_on_levels, finest_mean


def mean_var_at_level_quantity(level_samples, n_estimated):
    print("n estimated ", n_estimated)

    ###############
    # GET SAMPLES #
    ###############
    samples_level_0 = level_samples[0]
    samples_level_1 = level_samples[1]
    samples_level_2 = level_samples[2]
    samples_level_3 = level_samples[3]

    ########################
    # GET MOMENTS FUNCTION #
    ########################
    quantile = 1e-3
    n_moments = 3
    #all_samples = [samples_level_0, samples_level_1, samples_level_2, samples_level_3]
    #true_domain = estimate_domain(all_samples, quantile=quantile)
    #print("true domain ", true_domain)
    #moments_fn = Monomial(n_moments, true_domain, ref_domain=true_domain)

    total_var = [0, 0]
    level_variances = []
    mean_diffs_on_levels = []
    var_diffs_on_levels = []
    skewness_diffs_on_levels = []
    kurtosis_diffs_on_levels = []
    if len(n_estimated) == 1:
        ####
        # Level 0
        ####
        #print("samples_level_0.shape ", samples_level_0.shape)
        #print("samples level 0 ", samples_level_0)

        #print("np.max(samples_level_0) ", np.max(samples_level_0))
        # moments_mean_level_0 = np.nanmean(samples_level_0, axis=1)
        # moments_var_level_0 = np.nanvar(samples_level_0, axis=1)



        moments_mean_level_0 = np.nanmean(samples_level_0, axis=0)
        moments_var_level_0 = np.nanvar(samples_level_0, axis=0)



        print("moments_var_level_0 ", moments_var_level_0)

        mean_diffs_on_levels.append([])
        var_diffs_on_levels.append([])

        total_moments_mean = 0
        total_moments_var = 0
        total_moments_skewness = 0
        total_moments_kurtosis = 0

        finest_moments_mean = moments_mean_level_0

        second_moment = (samples_level_0 - moments_mean_level_0) ** 2
        third_moment = ((samples_level_0 - moments_mean_level_0) / np.sqrt(moments_var_level_0)) ** 3
        fourth_moment = ((samples_level_0 - moments_mean_level_0) / np.sqrt(moments_var_level_0)) ** 4

        finest_moments_var = np.mean(second_moment)
        finest_moments_skewness = np.mean(third_moment)
        finest_moments_kurtosis = np.mean(fourth_moment)

        print("finest_moments_skewness ", finest_moments_skewness)
        print("finest_moments_kurtosis ", finest_moments_kurtosis)
        level_variances.append([moments_var_level_0, np.nanvar(second_moment, axis=0), np.nanvar(third_moment, axis=0), np.nanvar(fourth_moment, axis=0)])

        print("moments_var_level_0 ", moments_var_level_0)

        total_var[0] = moments_var_level_0 / len(samples_level_0)
        total_var[1] = np.nanvar(second_moment, axis=0) / len(samples_level_0)

    if len(n_estimated) in [2, 3, 4]:
        ####
        # Level 0
        ####
        moments_mean_level_0 = np.nanmean(samples_level_0, axis=0)
        moments_var_level_0 = np.nanvar(samples_level_0, axis=0)

        second_moment_level_0 = (samples_level_0 - moments_mean_level_0) ** 2
        third_moment_level_0 = ((samples_level_0 - moments_mean_level_0) / np.sqrt(moments_var_level_0)) ** 3
        fourth_moment_level_0 = ((samples_level_0 - moments_mean_level_0) / np.sqrt(moments_var_level_0)) ** 4

        moments_skewness_level_0 = np.mean(third_moment_level_0, axis=0)
        moments_kurtosis_level_0 = np.mean(fourth_moment_level_0, axis=0)

        print("moments_var_level_0 ", moments_var_level_0)

        calc_moments_var_level_0 = np.mean((samples_level_0 - moments_mean_level_0)**2)
        print("calc_moments_var_level_0 ", calc_moments_var_level_0)

        ####
        # Level 1
        ####
        print("samples_level_1.shape ", samples_level_1.shape)
        moments_mean_level_1 = np.nanmean(samples_level_1, axis=0)
        moments_mean_level_1_fine = moments_mean_level_1[0]
        moments_mean_level_1_coarse = moments_mean_level_1[1]

        moments_var_level_1 = np.nanvar(samples_level_1, axis=0)
        moments_var_level_1_fine = moments_var_level_1[0]
        moments_var_level_1_coarse = moments_var_level_1[1]

        second_moment_level_1 = (samples_level_1 - moments_mean_level_1) ** 2
        third_moment_level_1 = ((samples_level_1 - moments_mean_level_1) / np.sqrt(moments_var_level_1)) ** 3
        fourth_moment_level_1 = ((samples_level_1 - moments_mean_level_1) / np.sqrt(moments_var_level_1)) ** 4

        moments_skewness_level_1 = np.mean(third_moment_level_1, axis=0)
        moments_skewness_level_1_fine = moments_skewness_level_1[0]
        moments_skewness_level_1_coarse = moments_skewness_level_1[1]
        moments_kurtosis_level_1 = np.mean(fourth_moment_level_1, axis=0)
        moments_kurtosis_level_1_fine = moments_kurtosis_level_1[0]
        moments_kurtosis_level_1_coarse = moments_kurtosis_level_1[1]

        # print("mean_level_1_fine ", mean_level_1_fine)
        # print("mean_level_1_coarse ", mean_level_1_coarse)

        level_variances.append([moments_var_level_0, np.nanvar(second_moment_level_0, axis=0), np.nanvar(third_moment_level_0, axis=0), np.nanvar(fourth_moment_level_0, axis=0)])
        level_variances.append([np.nanvar(samples_level_1[:, 0] - samples_level_1[:, 1], axis=0),
                                np.nanvar(second_moment_level_1[:, 0] - second_moment_level_1[:, 1], axis=0),
                                np.nanvar(third_moment_level_1[:, 0] - third_moment_level_1[:, 1], axis=0),
                                np.nanvar(fourth_moment_level_1[:, 0] - fourth_moment_level_1[:, 1], axis=0),
                                ])

        total_var[0] = moments_var_level_0 / len(samples_level_0) + \
                    (np.nanvar(samples_level_1[:, 0]-samples_level_1[:, 1], axis=0)) / len(samples_level_1)
        total_var[1] = np.nanvar(second_moment_level_0, axis=0) / len(samples_level_0) + \
                       (np.nanvar(second_moment_level_1[:, 0] - second_moment_level_1[:, 1], axis=0)) / len(samples_level_1)

        mean_level_0_fine_1_coarse_diff = moments_mean_level_0 - moments_mean_level_1_coarse
        var_level_0_fine_1_coarse_diff = moments_var_level_0 - moments_var_level_1_coarse
        skewness_level_0_fine_1_coarse_diff = moments_skewness_level_0 - moments_skewness_level_1_coarse
        kurtosis_level_0_fine_1_coarse_diff = moments_kurtosis_level_0 - moments_kurtosis_level_1_coarse

        mean_diffs_on_levels.append(mean_level_0_fine_1_coarse_diff)
        var_diffs_on_levels.append(var_level_0_fine_1_coarse_diff)
        skewness_diffs_on_levels.append(skewness_level_0_fine_1_coarse_diff)
        kurtosis_diffs_on_levels.append(kurtosis_level_0_fine_1_coarse_diff)

        total_moments_mean = mean_level_0_fine_1_coarse_diff
        total_moments_var = var_level_0_fine_1_coarse_diff
        total_moments_skewness = skewness_level_0_fine_1_coarse_diff
        total_moments_kurtosis = kurtosis_level_0_fine_1_coarse_diff

        finest_moments_mean = moments_mean_level_1_fine
        finest_moments_var = moments_var_level_1_fine
        finest_moments_skewness = moments_skewness_level_1_fine
        finest_moments_kurtosis = moments_kurtosis_level_1_fine

        print("#########################")
        print("Covariance between levels")
        print("#########################")

        # level_1_diff = moments_level_1[:, 0,: ] - moments_level_1[:, 1,: ]
        # print("level 1 diff", level_1_diff.shape)
        # print("moments_level_0 ", moments_level_0.shape)
        #
        # cov_level_0_level_1 = np.cov(moments_level_0[:, 0], level_1_diff[:, 0])
        # print("cov level 0 level 1 ", cov_level_0_level_1)

    if len(n_estimated) in [3, 4]:
        ####
        # Level 2
        ####
        moments_mean_level_2 = np.nanmean(samples_level_2, axis=0)
        moments_mean_level_2_fine = moments_mean_level_2[0]
        moments_mean_level_2_coarse = moments_mean_level_2[1]

        moments_var_level_2 = np.nanvar(samples_level_2, axis=0)
        moments_var_level_2_fine = moments_var_level_2[0]
        moments_var_level_2_coarse = moments_var_level_2[1]

        second_moment_level_2 = (samples_level_2 - moments_mean_level_2) ** 2
        third_moment_level_2 = ((samples_level_2 - moments_mean_level_2) / np.sqrt(moments_var_level_2)) ** 3
        fourth_moment_level_2 = ((samples_level_2 - moments_mean_level_2) / np.sqrt(moments_var_level_2)) ** 4

        moments_skewness_level_2 = np.mean(third_moment_level_2, axis=0)
        moments_skewness_level_2_fine = moments_skewness_level_2[0]
        moments_skewness_level_2_coarse = moments_skewness_level_2[1]
        moments_kurtosis_level_2 = np.mean(fourth_moment_level_2, axis=0)
        moments_kurtosis_level_2_fine = moments_kurtosis_level_2[0]
        moments_kurtosis_level_2_coarse = moments_kurtosis_level_2[1]

        # moments_skewness_level_2 = np.mean(((samples_level_2 - moments_mean_level_2) / np.sqrt(moments_var_level_2)) ** 3)
        # moments_skewness_level_2_fine = moments_skewness_level_2[0]
        # moments_skewness_level_2_coarse = moments_skewness_level_2[1]
        # moments_kurtosis_level_2 = np.mean(((samples_level_2 - moments_mean_level_2) / np.sqrt(moments_var_level_2)) ** 4)
        # moments_kurtosis_level_2_fine = moments_kurtosis_level_2[0]
        # moments_kurtosis_level_2_coarse = moments_kurtosis_level_2[1]

        total_var[0] += (np.nanvar(samples_level_2[:, 0] - samples_level_2[:, 1],axis=0)) / len(samples_level_2)
        total_var[1] += (np.nanvar(second_moment_level_2[:, 0] - second_moment_level_2[:, 1], axis=0)) / len(samples_level_2)

        level_variances.append([np.nanvar(samples_level_2[:, 0] - samples_level_2[:, 1], axis=0),
                                np.nanvar(second_moment_level_2[:, 0] - second_moment_level_2[:, 1], axis=0),
                                np.nanvar(third_moment_level_2[:, 0] - third_moment_level_2[:, 1], axis=0),
                                np.nanvar(fourth_moment_level_2[:, 0] - fourth_moment_level_2[:, 1], axis=0),
                                ])

        mean_level_1_fine_2_coarse_diff = moments_mean_level_1_fine - moments_mean_level_2_coarse
        var_level_1_fine_2_coarse_diff = moments_var_level_1_fine - moments_var_level_2_coarse
        skewness_level_1_fine_2_coarse_diff = moments_skewness_level_1_fine - moments_skewness_level_2_coarse
        kurtosis_level_1_fine_2_coarse_diff = moments_kurtosis_level_1_fine - moments_kurtosis_level_2_coarse

        mean_diffs_on_levels.append(mean_level_1_fine_2_coarse_diff)
        var_diffs_on_levels.append(var_level_1_fine_2_coarse_diff)
        skewness_diffs_on_levels.append(skewness_level_1_fine_2_coarse_diff)
        kurtosis_diffs_on_levels.append(kurtosis_level_1_fine_2_coarse_diff)

        total_moments_mean += mean_level_1_fine_2_coarse_diff
        total_moments_var += var_level_1_fine_2_coarse_diff
        total_moments_skewness += skewness_level_1_fine_2_coarse_diff
        total_moments_kurtosis += kurtosis_level_1_fine_2_coarse_diff

        finest_moments_mean = moments_mean_level_2_fine
        finest_moments_var = moments_var_level_2_fine
        finest_moments_skewness = moments_skewness_level_2_fine
        finest_moments_kurtosis = moments_kurtosis_level_2_fine

    if len(n_estimated) in [4]:
        ####
        # Level 3
        ####
        moments_mean_level_3 = np.nanmean(samples_level_3, axis=0)
        moments_mean_level_3_fine = moments_mean_level_3[0]
        moments_mean_level_3_coarse = moments_mean_level_3[1]

        moments_var_level_3 = np.nanvar(samples_level_3, axis=0)
        moments_var_level_3_fine = moments_var_level_3[0]
        moments_var_level_3_coarse = moments_var_level_3[1]

        second_moment_level_3 = (samples_level_3 - moments_mean_level_3) ** 2
        third_moment_level_3 = ((samples_level_3 - moments_mean_level_3) / np.sqrt(moments_var_level_3)) ** 3
        fourth_moment_level_3 = ((samples_level_3 - moments_mean_level_3) / np.sqrt(moments_var_level_3)) ** 4

        moments_skewness_level_3 = np.mean(third_moment_level_3, axis=0)
        moments_skewness_level_3_fine = moments_skewness_level_3[0]
        moments_skewness_level_3_coarse = moments_skewness_level_3[1]
        moments_kurtosis_level_3 = np.mean(fourth_moment_level_3, axis=0)
        moments_kurtosis_level_3_fine = moments_kurtosis_level_3[0]
        moments_kurtosis_level_3_coarse = moments_kurtosis_level_3[1]

        # moments_skewness_level_3 = np.mean(((samples_level_3 - moments_mean_level_3) / np.sqrt(moments_var_level_3)) ** 3)
        # moments_skewness_level_3_fine = moments_skewness_level_3[0]
        # moments_skewness_level_3_coarse = moments_skewness_level_3[1]
        # moments_kurtosis_level_3 = np.mean(((samples_level_3 - moments_mean_level_3) / np.sqrt(moments_var_level_3)) ** 4)
        # moments_kurtosis_level_3_fine = moments_kurtosis_level_3[0]
        # moments_kurtosis_level_3_coarse = moments_kurtosis_level_3[1]

        total_var[0] += (np.nanvar(samples_level_3[:, 0] - samples_level_3[:, 1], axis=0)) / len(samples_level_3)
        total_var[1] += (np.nanvar(second_moment_level_3[:, 0] - second_moment_level_3[:, 1], axis=0)) / len(samples_level_3)

        level_variances.append([np.nanvar(samples_level_3[:, 0] - samples_level_3[:, 1], axis=0),
                                np.nanvar(second_moment_level_3[:, 0] - second_moment_level_3[:, 1], axis=0),
                                np.nanvar(third_moment_level_3[:, 0] - third_moment_level_3[:, 1], axis=0),
                                np.nanvar(fourth_moment_level_3[:, 0] - fourth_moment_level_3[:, 1], axis=0),
                                ])

        mean_level_2_fine_3_coarse_diff = moments_mean_level_2_fine - moments_mean_level_3_coarse
        var_level_2_fine_3_coarse_diff = moments_var_level_2_fine - moments_var_level_3_coarse
        skewness_level_2_fine_3_coarse_diff = moments_skewness_level_2_fine - moments_skewness_level_3_coarse
        kurtosis_level_2_fine_3_coarse_diff = moments_kurtosis_level_2_fine - moments_kurtosis_level_3_coarse

        mean_diffs_on_levels.append(mean_level_2_fine_3_coarse_diff)
        var_diffs_on_levels.append(var_level_2_fine_3_coarse_diff)
        skewness_diffs_on_levels.append(skewness_level_2_fine_3_coarse_diff)
        kurtosis_diffs_on_levels.append(kurtosis_level_2_fine_3_coarse_diff)

        total_moments_mean += mean_level_2_fine_3_coarse_diff
        total_moments_var += var_level_2_fine_3_coarse_diff
        total_moments_skewness += skewness_level_2_fine_3_coarse_diff
        total_moments_kurtosis += kurtosis_level_2_fine_3_coarse_diff

        finest_moments_mean = moments_mean_level_3_fine
        finest_moments_var = moments_var_level_3_fine
        finest_moments_skewness = moments_skewness_level_3_fine
        finest_moments_kurtosis = moments_kurtosis_level_3_fine


    print("total var ", total_var)
    #print("level variances ", level_variances)

    total_moments = [total_moments_mean, total_moments_var, total_moments_skewness, total_moments_kurtosis]
    diffs_on_levels = [mean_diffs_on_levels, var_diffs_on_levels, skewness_diffs_on_levels, kurtosis_diffs_on_levels]
    finest_moments = [finest_moments_mean, finest_moments_var, finest_moments_skewness, finest_moments_kurtosis]

    # total_moments = [total_moments_mean, total_moments_var]
    # diffs_on_levels = [mean_diffs_on_levels, var_diffs_on_levels]
    # finest_moments = [finest_moments_mean, finest_moments_var]
    level_variances = np.array(level_variances)[:, :2]

    return total_var, np.array(level_variances), diffs_on_levels, finest_moments, total_moments




########
## MC ##
########
mlmc_file_paths = {"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/1LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ref_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_h_5_seed_12345_SRF_direct_gen/mlmc_1.hdf5":   {"name": 'MC', "n_subdomains": [1728], "init_sample_vec": [500], "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/1LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ref_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_h_5_seed_12345_SRF_direct_gen/mlmc_1.hdf5"}}
##########
## 2LMC ##
##########
#mlmc_file_paths = [
# #"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_75_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept/mlmc_2.hdf5",
# #"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_8_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept/mlmc_2.hdf5",
# "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_10_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept/mlmc_2.hdf5",
# #"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_10_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_2/mlmc_2.hdf5",
# #"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_125_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept/mlmc_2.hdf5",
# "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_15_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept/mlmc_2.hdf5",
# "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_20_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept/mlmc_2.hdf5",
# "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_40_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept/mlmc_2.hdf5",
# #"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_40_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_2/mlmc_2.hdf5"
# ]

# mlmc_file_paths = {
# #"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_75_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5": {"name": '2LMC-1', "n_subdomains": [1728], "init_sample_vec": [500, 10],
# #                                                                                                                                                                                                                                                                                                                                           "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_75_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5"},
# "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_10_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5": {"name": '2LMC-2', "n_subdomains": [729], "init_sample_vec": [500, 14],
#                                                                                                                                                                                                                                                                                                                                            "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_10_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5"},
# #"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_125_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5": {"name": '2LMC-3', "n_subdomains": [343],  "init_sample_vec": [500, 30],
# #                                                                                                                                                                                                                                                                                                                                            "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_125_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5"},
# #"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_15_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5": {"name": '2LMC-4', "n_subdomains": [216], "init_sample_vec": [500, 47],
# #                                                                                                                                                                                                                                                                                                                                           "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_15_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5"},
# #"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_20_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5": {"name": '2LMC-5', "n_subdomains": [125], "init_sample_vec": [500, 80],
# #                                                                                                                                                                                                                                                                                                                                           "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_20_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5"},
# # "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_40_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5": {"name": '2LMC-6', "n_subdomains": [27], "init_sample_vec": [500, 370],
# #                                                                                                                                                                                                                                                                                                                                            "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_40_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5"},
# # "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_40_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_2/mlmc_2.hdf5": {"name": '2LMC-6B', "n_subdomains": [0], "init_sample_vec": [500, 370],
# #                                                                                                                                                                                                                                                                                                                                            "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_40_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed_2/mlmc_2.hdf5"}
# }

##########
## 3LMC ##
##########
#mlmc_file_paths = [
# #"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_10_20_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest/mlmc_3.hdf5",
# "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_10_20_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_fixed/mlmc_3.hdf5",
# "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_10_30_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest_fixed/mlmc_3.hdf5",
# #"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_10_30_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest_fixed/mlmc_3.hdf5",
# #"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_12_24_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest/mlmc_3.hdf5",
# #"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_12_24_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest_fixed/mlmc_3.hdf5",
# #"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_75_1125_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest/mlmc_3.hdf5",
# "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_75_1125_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest_fixed/mlmc_3.hdf5",
# "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_125_3125_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest_fixed/mlmc_3.hdf5"
# ]
#
mlmc_file_paths = {
#"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_75_1125_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest_fixed/mlmc_3.hdf5": {"name": '3LMC-1', "n_subdomains": [512, 1728], "init_sample_vec": [500, 20, 10],
#                                                                                                                                                                                                                                                                                                                                              "cost_file":"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_75_1125_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest_fixed/mlmc_3_cost.hdf5"},
"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_10_20_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_fixed/mlmc_3.hdf5": {"name": '3LMC-2', "n_subdomains": [125, 729], "init_sample_vec": [500, 80, 10],
                                                                                                                                                                                                                                                                                                                                    "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_10_20_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_fixed/mlmc_3.hdf5"},
"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_10_20_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_fixed_no_nn/mlmc_3.hdf5": {"name": '3LMC-2', "n_subdomains": [729], "init_sample_vec": [500, 20, 10],
                                                                                                                                                                                                                                                                                                                                    "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_10_20_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_fixed_no_nn/mlmc_3.hdf5"},

#"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_12_24_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest_fixed/mlmc_3.hdf5": {"name": '3LMC-3', "n_subdomains": [64, 512], "init_sample_vec": [500, 156, 20],
#                                                                                                                                                                                                                                                                                                                                    "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_12_24_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest_fixed/mlmc_3.hdf5"},
#"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_10_30_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest_fixed/mlmc_3.hdf5": {"name": '3LMC-3', "n_subdomains": [729], "init_sample_vec": [500, 20, 10],
#                                                                                                                                                                                                                                                                                                                                           "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_10_30_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest_fixed/mlmc_3.hdf5"},
#"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_8_16_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest_fixed/mlmc_3.hdf5",
#"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_12_24_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest_fixed/mlmc_3.hdf5"
#"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_125_3125_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest_fixed/mlmc_3.hdf5": {"name": '3LMC-4', "n_subdomains": [343], "init_sample_vec": [100, 75, 75],                                                                                                                                                                                                                                                                                                                                          "cost_file":"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_125_3125_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest_fixed/mlmc_3.hdf5"}
}


############
##  4LMC  ##
# ##########
# mlmc_file_paths = {
# "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/4LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_75_1125_16875_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_fixed/mlmc_4.hdf5": {"name": '4LMC-1', "n_subdomains": [216, 512, 1728], "init_sample_vec": [500, 47, 20, 10], "cost_file":"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/4LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_75_1125_16875_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_fixed/mlmc_4.hdf5"},
# #"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/4LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_875_153_268_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_fixed/mlmc_4.hdf5": {"name": '4LMC-2', "n_subdomains": [1000], "init_sample_vec": [500, 10],
#     # "cost_file":"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/4LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_875_153_268_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_fixed/mlmc_4.hdf5"},
# #"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/4LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_8_16_27_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_fixed/mlmc_4.hdf5": {"name": '4LMC-3', "n_subdomains": [9, 125, 729], "init_sample_vec": [500, 1111, 80, 10],  "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/4LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_10_20_40_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_fixed/mlmc_4.hdf5"},
# #"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/4LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_10_20_40_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_fixed_2/mlmc_4.hdf5": {"name": '4LMC-3B', "n_subdomains": [1728], "init_sample_vec": [500, 1111, 80, 10], "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/4LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_10_20_40_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_fixed_2/mlmc_4.hdf5"},
# }

# mlmc_file_paths = [#"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_75_1125_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest/mlmc_3.hdf5",
#                    #"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_8_16_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest/mlmc_3.hdf5",
#                    "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_10_20_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest/mlmc_3.hdf5",
# #"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_12_24_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest/mlmc_3.hdf5",
# #                   "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_10_30_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest/mlmc_3.hdf5",
# #                   "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_125_3125_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest/mlmc_3.hdf5",
#                    ]
#
# mlmc_file_paths = ["/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/4LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_75_1125_16875_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population/mlmc_4.hdf5",
#                    "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/4LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_875_153_268_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population/mlmc_4.hdf5"]


#########
## ALL ##
#########
# mlmc_file_paths = {
# "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_75_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5": {"name": '2LMC-1', "n_subdomains": [1728], "init_sample_vec": [500, 10],
#                                                                                                                                                                                                                                                                                                                                            "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_75_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5"},
# "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_10_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5": {"name": '2LMC-2', "n_subdomains": [729], "init_sample_vec": [500, 14],
#                                                                                                                                                                                                                                                                                                                                            "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_10_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5"},
# "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_125_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5": {"name": '2LMC-3', "n_subdomains": [343],  "init_sample_vec": [500, 30],
#                                                                                                                                                                                                                                                                                                                                             "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_125_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5"},
# "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_15_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5": {"name": '2LMC-4', "n_subdomains": [216], "init_sample_vec": [500, 47],
#                                                                                                                                                                                                                                                                                                                                            "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_15_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5"},
# "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_20_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5": {"name": '2LMC-5', "n_subdomains": [125], "init_sample_vec": [500, 80],
#                                                                                                                                                                                                                                                                                                                                            "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_20_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5"},
# "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_75_1125_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest_fixed/mlmc_3.hdf5": {
#         "name": '3LMC-1', "n_subdomains": [512, 1728], "init_sample_vec": [500, 20, 10],
#         "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_75_1125_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest_fixed/mlmc_3_cost.hdf5"},
# "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_10_20_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_fixed/mlmc_3.hdf5": {
#         "name": '3LMC-2', "n_subdomains": [125, 729], "init_sample_vec": [500, 80, 14],
#         "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_10_20_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_fixed/mlmc_3.hdf5"},
#
# #"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_12_24_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest_fixed/mlmc_3.hdf5": {
# #        "name": '3LMC-3', "n_subdomains": [64, 512], "init_sample_vec": [500, 156, 20],
# #        "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_12_24_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest_fixed/mlmc_3.hdf5"},
# "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/4LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_75_1125_16875_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_fixed/mlmc_4.hdf5": {"name": '4LMC-1', "n_subdomains": [216, 512, 1728], "init_sample_vec": [500, 47, 20, 10], "cost_file":"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/4LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_75_1125_16875_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_fixed/mlmc_4.hdf5"},
# }

reference_mc_path = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/1LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ref_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_h_5_seed_12345_SRF_direct_gen/mlmc_1.hdf5"
reference_mc_info = {"name": 'MC_ref', "n_subdomains": [1728], "init_sample_vec": [500], "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/1LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ref_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_h_5_seed_12345_SRF_direct_gen/mlmc_1.hdf5"}

##############
## COND TN  ##
##############
##########
## 1LMC ##
# ##########
#
#mlmc_file_paths = {"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/1LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ref_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_h_5_seed_12345_SRF_direct_gen/mlmc_1.hdf5": {"name": 'MC', "n_subdomains": [1728], "init_sample_vec": [500],"cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/1LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ref_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_h_5_seed_12345_SRF_direct_gen/mlmc_1.hdf5"}}
#
# ##########
# ## 2LMC ##
# # ##########
# mlmc_file_paths = {
# "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_75_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5": {"name": '2LMC-1', "n_subdomains": [1728], "init_sample_vec": [500, 10],
#                                                                                                                                                                                                                                                                                                                                    "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_75_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5"},
# "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_10_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5": {"name": '2LMC-2', "n_subdomains": [729], "init_sample_vec": [500, 14],
#                                                                                                                                                                                                                                                                                                                                    "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_10_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5"},
# "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_125_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5": {"name": '2LMC-3', "n_subdomains": [343],  "init_sample_vec": [500, 30],
#                                                                                                                                                                                                                                                                                                                                     "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_125_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5"},
# "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_15_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5": {"name": '2LMC-5', "n_subdomains": [125], "init_sample_vec": [500, 80],
#                                                                                                                                                                                                                                                                                                                                    "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_15_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed_cost/mlmc_2_cost.hdf5"},
# "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_20_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5": {"name": '2LMC-5', "n_subdomains": [125], "init_sample_vec": [500, 80],
#                                                                                                                                                                                                                                                                                                                                    "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_20_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5"},
# # "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_40_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5": {"name": '2LMC-6', "n_subdomains": [0], "init_sample_vec": [500, 370],
# #                                                                                                                                                                                                                                                                                                                                 "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_40_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5"},
# }

# # ##########
# # ## 3LMC ##
# # ##########
# mlmc_file_paths = {
# "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_75_1125_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest_fixed/mlmc_3.hdf5": {"name": '3LMC-1', "n_subdomains": [729, 1728], "init_sample_vec": [500, 20, 10],
#                                                                                                                                                                                                                                                                                                                                               "cost_file":"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_75_1125_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest_fixed/mlmc_3_cost.hdf5"},
# "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_10_20_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_fixed/mlmc_3.hdf5": {"name": '3LMC-2', "n_subdomains": [729], "init_sample_vec": [500, 20, 14],
#                                                                                                                                                                                                                                                                                                                                     "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_10_20_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_fixed/mlmc_3.hdf5"},
# # "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_12_24_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest_fixed/mlmc_3.hdf5": {"name": '3LMC-2', "n_subdomains": [729], "init_sample_vec": [500, 20, 10],
# #                                                                                                                                                                                                                                                                                                                                     "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_12_24_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest_fixed/mlmc_3_cost.hdf5"},
# # "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_12_24_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest_fixed/mlmc_3.hdf5": {"name": '3LMC-3', "n_subdomains": [729], "init_sample_vec": [500, 20, 10],
# #                                                                                                                                                                                                                                                                                                                                    "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_12_24_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest_fixed/mlmc_3.hdf5"},
# # "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_10_30_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest_fixed/mlmc_3.hdf5": {"name": '3LMC-3', "n_subdomains": [729], "init_sample_vec": [500, 20, 10],
# #                                                                                                                                                                                                                                                                                                                                            "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_10_30_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest_fixed/mlmc_3.hdf5"},
# #"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_8_16_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest_fixed/mlmc_3.hdf5",
# #"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_12_24_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest_fixed/mlmc_3.hdf5"
# #"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_125_3125_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest_fixed/mlmc_3.hdf5": {"name": '3LMC-4', "n_subdomains": [343], "init_sample_vec": [100, 75, 75],                                                                                                                                                                                                                                                                                                                                          "cost_file":"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/outflow/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_125_3125_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest_fixed/mlmc_3.hdf5"}
# }

############
##  4LMC  ##
# ##########
# mlmc_file_paths = {
# "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/4LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_75_1125_16875_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_fixed/mlmc_4.hdf5": {"name": '4LMC-1', "n_subdomains": [216, 512, 1728], "init_sample_vec": [500, 47, 20, 10], "cost_file":"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/4LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_75_1125_16875_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_fixed/mlmc_4_cost.hdf5"},
# }

# ##########
# ## ALL ##
# # ##########
mlmc_file_paths = {
# "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_75_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5": {"name": '2LMC-1', "n_subdomains": [1728], "init_sample_vec": [500, 10],
#                                                                                                                                                                                                                                                                                                                                    "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_75_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5"},
# "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_10_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5": {"name": '2LMC-2', "n_subdomains": [729], "init_sample_vec": [500, 14],
#                                                                                                                                                                                                                                                                                                                                    "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_10_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5"},
# "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_125_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5": {"name": '2LMC-3', "n_subdomains": [343],  "init_sample_vec": [500, 30],
#                                                                                                                                                                                                                                                                                                                                     "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_125_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5"},
# "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_15_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5": {"name": '2LMC-4', "n_subdomains": [216], "init_sample_vec": [500, 47],
#                                                                                                                                                                                                                                                                                                                                    "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_15_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed_cost/mlmc_2_cost.hdf5"},
# "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_20_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5": {"name": '2LMC-5', "n_subdomains": [125], "init_sample_vec": [500, 80],
#                                                                                                                                                                                                                                                                                                                                    "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/2LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_H_20_h_5_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_extended_domain_orig_kept_fixed/mlmc_2.hdf5"},
# "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_75_1125_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest_fixed/mlmc_3.hdf5": {"name": '3LMC-1', "n_subdomains": [729, 1728], "init_sample_vec": [500, 20, 10],
#                                                                                                                                                                                                                                                                                                                                               "cost_file":"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_75_1125_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_latest_fixed/mlmc_3_cost.hdf5"},
"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_10_20_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_fixed/mlmc_3.hdf5": {"name": '3LMC-2', "n_subdomains": [729], "init_sample_vec": [500, 20, 14],
                                                                                                                                                                                                                                                                                                                                 "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/3LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_10_20_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_fixed/mlmc_3.hdf5"},
# "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/4LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_75_1125_16875_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_fixed/mlmc_4.hdf5": {"name": '4LMC-1', "n_subdomains": [216, 512, 1728], "init_sample_vec": [500, 47, 20, 10], "cost_file":"/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/4LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ratio_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_fine_coarse_h_5_75_1125_16875_regular_grid_interp_seed_12345_L0_test_SRF_gen_gpu_samples_population_fixed/mlmc_4_cost.hdf5"},
#
}

# #
reference_mc_path = "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/1LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ref_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_h_5_seed_12345_SRF_direct_gen/mlmc_1.hdf5"
reference_mc_info = {"name": 'MC_ref', "n_subdomains": [1728], "init_sample_vec": [500], "cost_file": "/home/martin/Documents/MLMC-DFM_3D_data/MLMC-DFM_3D_experiments/MLMC/1LMC/cond_frac_1_5/fractures_diag_cond/domain_60/ref_MLMC-DFM_3D_n_voxels_64_save_bulk_avg_eq_tn_hom_domain_60_max_frac_limit_cl_10_h_5_seed_12345_SRF_direct_gen/mlmc_1.hdf5"}
# # #

if __name__ == "__main__":
    all_level_parameters = []
    all_moments_mean_obj = []
    all_n_ops = []
    all_l_vars = []

    #ref_level_parameters, ref_moments_mean_obj, ref_n_ops = process(reference_mc_path, reference_mc_info, target_var=1e-9)

    #ref_level_parameters, ref_moments_mean_obj, ref_n_ops = process(reference_mc_path, reference_mc_info, target_var=1e-11)


    # for i in range(0, 25, 1):
    #     #np.random.seed(i)  # 42

    mlmc_info_names = []
    for mlmc_file_path, mlmc_info in mlmc_file_paths.items():
        #np.random.seed(42) # general seed
        #np.random.seed(1) # different seed for 2LMC-2, used for cond_tn cases
        #np.random.seed(20)  # different seed for 2LMC-4
        np.random.seed(2) # different seed for 3LMC-2 for COND TN case to meet target variance
        print("##################################################")
        print("##################################################")
        print("##################################################")
        print("mlmc file path ", mlmc_file_path)
        print("mlmc_info ", mlmc_info)
        mlmc_info_names.append(mlmc_info["name"])
        #level_parameters, l_vars, n_ops = process(mlmc_file_path, mlmc_info, target_var=5e-8)
        level_parameters, l_vars, n_ops = process(mlmc_file_path, mlmc_info, target_var=2.5e-10)

        all_level_parameters.append(level_parameters)
        #all_moments_mean_obj.append(moments_mean_obj)
        print("l vars ", l_vars)
        print("max l vars ", np.max(l_vars, axis=1))
        print("l vars shape ", l_vars.shape)
        all_l_vars.append(np.max(l_vars, axis=1))
        all_n_ops.append(n_ops)

    print("all_level_parameters ", all_level_parameters)

    plot_variance_mesh_size(all_level_parameters, all_l_vars, all_n_ops, moments=[0], label_names=mlmc_info_names, fontsize=17, fontsize_ticks=15)
    plot_cost_mesh_size(all_level_parameters, all_n_ops, label_names=mlmc_info_names, fontsize=17, fontsize_ticks=15)
    #plot_cost_mesh_size(all_level_parameters, all_n_ops, label_names=mlmc_info_names, fontsize=17, fontsize_ticks=15)
    #plot_bias(all_level_parameters,all_moments_mean_obj, ref_moments_mean_obj, label_names=["MLMC-ref"] + mlmc_info_names,  fontsize=17, fontsize_ticks=15)
