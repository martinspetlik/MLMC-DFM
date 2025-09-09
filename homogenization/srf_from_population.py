import os
import numpy as np


class SRFFromTensorPopulation:
    def __init__(self, config_dict):
        self._cond_tns = None
        self._config_dict = config_dict
        self._get_tensors(config_dict)
        self._srf_gstools_model = None
        self._svd_model = {}

        self.mean_log_conductivity = None
        if "mean_log_conductivity" in config_dict:
            self.mean_log_conductivity = config_dict["mean_log_conductivity"]

        self._rf_sample = None

        fine_step = config_dict["fine"]["step"]
        print("fine step ", fine_step)
        print("coarse step ", config_dict["coarse"]["step"])

        coarse_step = config_dict["coarse"]["step"]

        hom_block_size = int(fine_step * 3)

        level_parameters = list(np.squeeze(config_dict["sim_config"]["level_parameters"], axis=1))
        current_level_index = list(np.squeeze(config_dict["sim_config"]["level_parameters"], axis=1)).index(fine_step)
        previous_level_fine_step = level_parameters[current_level_index + 1]

        previous_level_hom_block_size = int(previous_level_fine_step * 3)

        if "hom_box_fine_step_mult" in config_dict["sim_config"]:
            hom_block_size = int(fine_step * config_dict["sim_config"]["hom_box_fine_step_mult"])
            previous_level_hom_block_size = int(previous_level_fine_step * config_dict["sim_config"]["hom_box_fine_step_mult"])

        if coarse_step == 0:
            orig_domain_box = config_dict["sim_config"]["geometry"]["orig_domain_box"]
            print("orig_domain_box ", orig_domain_box)
            larger_domain_size = orig_domain_box[0] + previous_level_hom_block_size + previous_level_fine_step
        else:
            orig_domain_box = config_dict["sim_config"]["geometry"]["orig_domain_box"]
            print("orig_domain_box ", orig_domain_box)
            larger_domain_size = orig_domain_box[0] + hom_block_size + fine_step

        # larger_domain_reduced_by_homogenization = larger_domain_size
        # for hb_size in hom_block_sizes:
        #     larger_domain_reduced_by_homogenization -= hb_size

        print("new larger_domain_size ", larger_domain_size)
        #print("larger_domain_reduced_by_homogenization ", larger_domain_reduced_by_homogenization)
        print("hom block size ", hom_block_size)
        print("previous level hom block size ", previous_level_hom_block_size)

        print("self._cond_tns_coords", self._cond_tns_coords)


        centers_1d = self.expand_domain_centers_1d(self._cond_tns_coords, new_domain=larger_domain_size)
        print("centers_1d shape:", centers_1d.shape)
        print("centers_1d ", centers_1d)

        z, y, x = np.meshgrid(centers_1d, centers_1d, centers_1d, indexing='ij')
        self.centers_3d = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

        print("self.centers_3d ", self.centers_3d)

        #self.centers_3d = SRFFromTensorPopulation.calculate_all_centers(larger_domain_size, previous_level_hom_block_size, overlap=previous_level_hom_block_size/2)

    def expand_domain_centers_1d(self, orig_points, new_domain, tol=1e-9, step_override=None):
        """
        Compute 1D centers of homogenization blocks for an expanded cubic domain.

        :param orig_points: array-like, shape (N, 3)
            Original coordinates of block centers.
        :param new_domain: float
            Side length of the new cubic domain.
        :param tol: float, optional
            Tolerance for floating-point comparisons. Default is 1e-9.
        :param step_override: float or None, optional
            If provided, use this step directly instead of inferring it.
        :return: np.ndarray
            Sorted 1D array of block centers along one axis.
        """
        pts = np.asarray(orig_points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("orig_points must be array-like with shape (N, 3).")

        L_new = float(new_domain) / 2.0

        # --- infer principal step from x-axis by frequency (ignores rare offsets) ---
        if step_override is None:
            # take x-axis; use absolute values and ignore near-zero
            x = np.abs(pts[:, 0])
            x = x[np.isfinite(x)]
            x = x[np.abs(x) > tol]

            if x.size == 0:
                raise ValueError("Cannot infer step: no nonzero coordinates found.")

            # round to stabilize duplicates, then choose the most frequent value
            xv, cnt = np.unique(np.round(x, 8), return_counts=True)
            # pick the positive value with the highest count
            # (on ties, pick the largest — typical for boundary-aligned coarse levels)
            max_count = cnt.max()
            candidates = xv[cnt == max_count]
            step = float(np.max(candidates))
        else:
            step = float(step_override)

        if step <= 0 or step > 2 * L_new + tol:
            raise ValueError(f"Inferred/override step looks invalid: {step}")

        # --- build centers on one side using the principal step, then clamp to boundary ---
        centers_pos = list(np.arange(step, L_new - tol, step))
        # ensure boundary is included
        if len(centers_pos) == 0 or abs(centers_pos[-1] - L_new) > tol:
            centers_pos.append(L_new)

        # mirror and include zero
        centers = sorted(set([-v for v in centers_pos] + [0.0] + centers_pos))
        return np.array(centers, dtype=float)

    @staticmethod
    def calculate_all_centers(domain_size, block_size, overlap):
        start = -domain_size / 2 #+ block_size / 2
        end = domain_size / 2 #- block_size / 2

        length = end - start
        stride = block_size - overlap

        # Calculate number of intervals, rounding up to cover full domain
        num_intervals = int(np.ceil(length / stride))
        n_centers = num_intervals + 1  # total centers

        # Recalculate exact stride to cover full domain exactly
        exact_stride = length / (n_centers - 1)

        centers_1d = start + exact_stride * np.arange(n_centers)
        print("centers_1d ", centers_1d)

        # Generate 3D centers meshgrid
        z, y, x = np.meshgrid(centers_1d, centers_1d, centers_1d, indexing='ij')
        centers_3d = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
        print("centers_3d.shape ", centers_3d.shape)

        return centers_3d

    @staticmethod
    def symmetrize_cond_tns(cond_tns):
        sym_values = (cond_tns[:, 1] + cond_tns[:, 2])/2
        cond_tns = np.delete(cond_tns, 2, 1)
        cond_tns[:, 1] = sym_values
        return cond_tns

    def _get_tensors(self, config_dict):
        cond_pop_file = config_dict["fine"]['cond_tn_pop_file']
        cond_pop_coords_file = config_dict["fine"]['cond_tn_pop_coords_file']
        self._cond_tns = np.load(cond_pop_file)
        self._cond_tns_coords = np.load(cond_pop_coords_file)

    def generate_field(self, reuse_sample=False):
        # Sample N indices independently with replacement from the M tensor samples
        indices = np.random.choice(self._cond_tns.shape[0], size=self.centers_3d.shape[0], replace=True)
        # Select sampled tensors
        sampled_tensors = self._cond_tns[indices]  # shape: (N, 3, 3)

        if reuse_sample:
            import glob
            import random
            # Original file path
            name, ext = os.path.splitext("cond_tensors_values.npz") # @TODO: use a variable

            # Build the search pattern
            pattern = os.path.join(self._config_dict["fine"]["common_files_dir"], f"{name}_*{ext}")
            # Find all matching files
            files = glob.glob(pattern)
            # Randomly select one if available
            if files:
                selected_file = random.choice(files)
                print("Selected file:", selected_file)
            else:
                print("No matching files found.")

            loaded_cond_tns_for_fine_sample = np.load(selected_file)['data']

            # Step 1: build a mapping from coordinate -> index in large array
            coord_to_index = {tuple(c): i for i, c in enumerate(self.centers_3d)}

            # Step 2: scatter values from small_tensors into large_tensors
            for tensor, coord in zip(loaded_cond_tns_for_fine_sample, self._cond_tns_coords):
                idx = coord_to_index.get(tuple(coord))
                if idx is not None:
                    sampled_tensors[idx] = tensor

        return sampled_tensors, self.centers_3d
