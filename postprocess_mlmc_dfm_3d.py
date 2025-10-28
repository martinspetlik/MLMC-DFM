import os
import sys
import numpy as np
import argparse
import mlmc
from mlmc.sample_storage_hdf import SampleStorageHDF
from mlmc.moments import Monomial
from mlmc.quantity.quantity import make_root_quantity
from mlmc.quantity.quantity_estimate import estimate_mean, moments
from mlmc import estimator
from mlmc.plot import diagnostic_plots as dp
import scipy.stats as stats
import matplotlib.pyplot as plt


class PostProcess:
    """
    Steps:
        1. Load storage & root quantity
        2. Select observable q_value
        3. Determine domain & build moments
        4. Compute mean/variance
        5. Estimate moments + regression
        6. Compute optimal n_samples
        7. Diagnostics & plots
    """

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('work_dir', help='Work directory')
        args = parser.parse_args(sys.argv[1:])
        self.work_dir = os.path.abspath(args.work_dir)

        self.n_levels = 2
        self.n_moments = 3
        self.target_var = 1e-6

        self.process()

    def compare_means(self, samples_a, samples_b, title=""):
        statistic, p_value = stats.ttest_ind(np.squeeze(samples_a), np.squeeze(samples_b), equal_var=False)
        print(f"{title} - p value: {p_value}")
        if p_value < 0.05:
            print(f"❗ {title} are significantly different.")
        else:
            print(f"✅ No significant difference between {title}")

    # -------------------------------------------------------------------------
    # MAIN PIPELINE
    # -------------------------------------------------------------------------
    def process(self):
        storage = SampleStorageHDF(file_path=os.path.join(self.work_dir, f"mlmc_{self.n_levels}.hdf5"))
        storage.chunk_size = int(1e8)

        root_quantity = make_root_quantity(storage=storage, q_specs=storage.load_result_format())
        cond = root_quantity["cond_tn"][1]["0"]  # quantity → time idx 1 → location "0"

        dim = storage.load_result_format()[0].shape[0]
        print(f"→ Components found: {dim}")

        for q_idx in range(dim):
            print("\n============================================================")
            print(f"              Processing component q_idx = {q_idx}")
            print("============================================================")

            q_value = cond[q_idx]
            self._process_component(q_value, storage)

    def _bootstrap_sizing(self, estimate_obj, n_ops):
        """
        Compute bootstrap-based estimate of required sample counts to reach target_var.
        Safe for 1+ levels.
        """
        # Default bootstrap sampling plan (truncate if fewer levels)
        sample_vec = [250, 125, 75, 50][:self.n_levels]

        try:
            bs_n_est = estimate_obj.bs_target_var_n_estimated(
                target_var=self.target_var,
                sample_vec=sample_vec,
                n_subsamples=10
            )
            total_bs_cost = float(np.sum(n_ops * bs_n_est))

            print("\n🧪 Bootstrap sample sizing:")
            print("  bs n_estimated:", bs_n_est)
            print("  bs total cost:", f"{total_bs_cost:.2f}")

            if hasattr(estimate_obj, "mean_bs_var"):
                print("  mean bootstrap variance:", estimate_obj.mean_bs_var)
            if hasattr(estimate_obj, "mean_bs_mean"):
                print("  mean bootstrap mean:", estimate_obj.mean_bs_mean)

            return bs_n_est

        except Exception as e:
            print("Bootstrap sizing skipped:", e)
            return None

    # -------------------------------------------------------------------------
    # PROCESS EACH COMPONENT
    # -------------------------------------------------------------------------
    def _process_component(self, q_value, storage):
        true_domain = self._estimate_domain(q_value, storage)
        moments_fn = Monomial(self.n_moments, true_domain)

        self._print_basic_statistics(q_value)

        estimate_obj = mlmc.estimator.Estimate(quantity=q_value, sample_storage=storage, moments_fn=moments_fn)

        # Moments
        m_quantity = moments(q_value, moments_fn)
        m_estimate = mlmc.estimator.Estimate(quantity=m_quantity, sample_storage=storage, moments_fn=moments_fn)

        # Regression
        n_ops = self._regression_and_optimal_samples(estimate_obj, storage)

        bs_n_est = self._bootstrap_sizing(estimate_obj, n_ops)

        # Diagnostics
        self._diagnostics(estimate_obj, m_estimate, storage, n_ops)

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------
    def _estimate_domain(self, q_value, storage):
        try:
            dom = mlmc.estimator.Estimate.estimate_domain(q_value, storage, quantile=1e-3)
            print("Estimated domain:", dom)
        except:
            dom = (0.000103, 0.1511)
            print("⚠ Domain estimation failed, using default:", dom)
        return dom

    def _print_basic_statistics(self, q_value):
        m = estimate_mean(q_value)
        m2 = estimate_mean(np.power(q_value, 2))

        mean_val = np.squeeze(m.mean)
        mean_sq_val = np.squeeze(m2.mean)
        var = mean_sq_val - mean_val ** 2

        if np.isscalar(mean_val):
            print(f"Collected MEAN: {float(mean_val):.6e}, VARIANCE: {float(var):.6e}")
        else:
            print("Collected MEAN:", mean_val)
            print("VARIANCE:", var)

    def _regression_and_optimal_samples(self, estimate_obj, storage):
        n_col = storage.get_n_collected()
        print("Collected samples per level:", n_col)

        variances, n_ops = estimate_obj.estimate_diff_vars_regression(n_col)
        print("Level variances:", variances)
        print("Ops per level:", n_ops)

        n_est = estimator.estimate_n_samples_for_target_variance(
            self.target_var, variances, n_ops, n_levels=self.n_levels
        )
        print("Optimal n_samples:", n_est)
        print("Total cost:", np.sum(n_ops * n_est))

        return n_ops

    def _plot_hist(self, data_a, data_b=None, label_a="A", label_b="B",
                   title="", bins=100, log=False, save_as=None):
        """
        Plot histograms for one or two datasets.
        """
        a = np.squeeze(data_a)

        if log:
            a = np.log10(a)
            a = a[~np.isinf(a)]

        plt.figure(figsize=(8, 6))
        plt.hist(a, bins=bins, density=True, alpha=0.6, label=label_a)

        if data_b is not None:
            b = np.squeeze(data_b)
            if log:
                b = np.log10(b)
                b = b[~np.isinf(b)]
            plt.hist(b, bins=bins, density=True, alpha=0.6, label=label_b)

        plt.title(title)
        plt.legend()
        plt.tight_layout()

        if save_as:
            plt.savefig(save_as, bbox_inches='tight')
        plt.show()

    def _diagnostics(self, est, est_moments, storage, n_ops):
        print("\n📊 Level diagnostics:")

        n_col = storage.get_n_collected()
        l0 = np.squeeze(est.get_level_samples(level_id=0, n_samples=n_col[0]))

        if self.n_levels >= 2:
            l1 = np.squeeze(est.get_level_samples(level_id=1, n_samples=n_col[1]))
            l1_fine, l1_coarse = l1[:, 0], l1[:, 1]

            self.compare_means(l0, l1_coarse, "L0 fine vs L1 coarse")

            # Histogram plot
            self._plot_hist(l0, l1_coarse, label_a="L0", label_b="L1 coarse",
                            title="Histogram: L0 vs L1 coarse", save_as="L0_L1_hist.png")

        if self.n_levels >= 3:
            l2 = np.squeeze(est.get_level_samples(level_id=2, n_samples=n_col[2]))
            l2_fine, l2_coarse = l2[:, 0], l2[:, 1]

            self.compare_means(l1_fine, l2_coarse, "L1 fine vs L2 coarse")

            self._plot_hist(l1_fine, l2_coarse, label_a="L1 fine", label_b="L2 coarse",
                            title="Histogram: L1 fine vs L2 coarse", save_as="L1_L2_hist.png")

        # Moments histograms (first 2 non-zero moments)
        try:
            m0 = est_moments.get_level_samples(level_id=0, n_samples=n_col[0])
            m1 = est_moments.get_level_samples(level_id=1, n_samples=n_col[1])

            # select non-constant moments & fine stream
            m0 = m0[1:, :, 0]
            m1 = m1[1:, :, 1]

            # Plot moment 0 histogram
            self._plot_hist(m0[0], m1[0], label_a="L0 m0", label_b="L1 m0",
                            title="Histogram: Moment 0 L0 vs L1 coarse")

        except Exception as e:
            print("Moment histogram skipped:", e)

        est.estimate_moments()

        # Core MLMC diagnostics (only meaningful with >= 2 levels)
        try:
            if self.n_levels >= 2 and getattr(est, "moments_mean_obj", None) is not None:
                dp.log_var_per_level(est.moments_mean_obj.l_vars, moments=range(1, self.n_moments))
                dp.log_mean_per_level(est.moments_mean_obj.l_means, moments=range(1, self.n_moments))
                dp.sample_cost_per_level(n_ops)
                dp.variance_to_cost_ratio(est.moments_mean_obj.l_vars, n_ops, moments=range(1, self.n_moments))
                dp.kurtosis_per_level(est.moments_mean_obj.mean, est.moments_mean_obj.l_means,
                                      moments=range(self.n_moments))
            else:
                print("Only one level available — skipping per-level log diagnostics.")
        except Exception as e:
            print("Per-level diagnostics skipped:", e)


if __name__ == "__main__":
    PostProcess()
