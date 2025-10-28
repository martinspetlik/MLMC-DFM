import os
import numpy as np


class SRFFromTensorPopulation:
    """
    Generates stochastic random fields (SRF) of conductivity tensors
    from a given tensor population.

    Responsibilities:
      - Load conductivity tensors and their coordinates from disk.
      - Expand original domain to include homogenization padding.
      - Compute homogenization block centers in 1D and 3D.
      - Provide sampling functions for conductivity fields.
    """

    def __init__(self, config_dict):
        """
        Initialize the SRF generator by loading conductivity tensors and
        computing homogenization block centers for an expanded domain.

        :param config_dict: dict
            Configuration dictionary. Expected keys:
            - "fine": {"step", "cond_tn_pop_file", "cond_tn_pop_coords_file"}
            - "coarse": {"step"}
            - "sim_config": {"geometry", "level_parameters", optional: "hom_box_fine_step_mult"}
            - optional: "mean_log_conductivity"
        """
        self._config_dict = config_dict
        self._cond_tns = None                # conductivity tensors
        self._cond_tns_coords = None         # tensor coordinates
        self._get_tensors(config_dict)       # load from files

        self._srf_gstools_model = None       # placeholder for gstools SRF model
        self._svd_model = {}                 # placeholder for SVD models

        # Optional mean log conductivity
        self.mean_log_conductivity = config_dict.get("mean_log_conductivity")

        self._rf_sample = None               # cached sample (optional reuse)

        # --------------------------------------------------------
        # Compute homogenization block sizes and expanded domain
        # --------------------------------------------------------
        larger_domain_size, _, _ = SRFFromTensorPopulation.get_larger_domain_size(config_dict)
        print("self._cond_tns_coords", self._cond_tns_coords)

        # --------------------------------------------------------
        # Compute centers of homogenization blocks
        # --------------------------------------------------------
        centers_1d = self.expand_domain_centers_1d(
            self._cond_tns_coords, new_domain=larger_domain_size
        )
        print("centers_1d shape:", centers_1d.shape)
        print("centers_1d ", centers_1d)

        # Build full 3D Cartesian product of centers
        z, y, x = np.meshgrid(centers_1d, centers_1d, centers_1d, indexing="ij")
        self.centers_3d = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
        print("self.centers_3d ", self.centers_3d)

        # Alternative way (kept for reference):
        # self.centers_3d = SRFFromTensorPopulation.calculate_all_centers(
        #     larger_domain_size, previous_level_hom_block_size, overlap=previous_level_hom_block_size/2
        # )

    @staticmethod
    def get_larger_domain_size(config_dict):
        fine_step = config_dict["fine"]["step"]
        coarse_step = config_dict["coarse"]["step"]
        print("fine step ", fine_step)
        print("coarse step ", coarse_step)

        # Default homogenization block sizes
        hom_block_size = coarse_step * 1.5
        level_parameters = list(np.squeeze(config_dict["sim_config"]["level_parameters"], axis=1))

        # Identify fine level index and previous level parameters
        current_level_index = level_parameters.index(fine_step)
        previous_level_fine_step = level_parameters[current_level_index + 1]
        previous_level_hom_block_size = fine_step * 1.5 # Coarse step on the previous level * 1.5

        # Override with multiplier if provided
        if "hom_box_fine_step_mult" in config_dict["sim_config"]:
            mult = config_dict["sim_config"]["hom_box_fine_step_mult"]
            hom_block_size = fine_step * mult
            previous_level_hom_block_size = previous_level_fine_step * mult

        # Compute expanded domain size
        orig_domain_box = config_dict["sim_config"]["geometry"]["orig_domain_box"]
        if coarse_step == 0:
            larger_domain_size = orig_domain_box[0] + previous_level_hom_block_size + previous_level_fine_step
        else:
            larger_domain_size = orig_domain_box[0] + hom_block_size + fine_step

        print("orig_domain_box ", orig_domain_box)
        print("new larger_domain_size ", larger_domain_size)
        print("hom block size ", hom_block_size)
        print("previous level hom block size ", previous_level_hom_block_size)

        return larger_domain_size, hom_block_size, previous_level_hom_block_size

    @staticmethod
    def expand_domain_centers_1d(orig_points, new_domain, tol=1e-9, step_override=None):
        """
        Keep all original centers and extend outward to fill the new cubic domain.

        :param orig_points: array-like, shape (N, 3)
        :param new_domain: float
        :param tol: float, optional
        :param step_override: float or None, optional
        :return: np.ndarray of extended 1D centers
        """
        pts = np.asarray(orig_points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("orig_points must be array-like with shape (N, 3).")

        x = np.sort(np.unique(pts[:, 0]))
        L_new = float(new_domain) / 2.0

        # infer step
        if step_override is None:
            diffs = np.diff(x)
            diffs = diffs[np.abs(diffs) > tol]
            if len(diffs) == 0:
                raise ValueError("Cannot infer step size.")
            step = float(np.median(diffs))
        else:
            step = float(step_override)

        if step <= 0:
            raise ValueError(f"Invalid step size: {step}")

        # extend to left
        xmin = x[0]
        while xmin - step >= -L_new - tol:
            xmin -= step
            x = np.insert(x, 0, xmin)

        # extend to right
        xmax = x[-1]
        while xmax + step <= L_new + tol:
            xmax += step
            x = np.append(x, xmax)

        # snap exact boundaries if close
        if abs(x[0] + L_new) > tol and -L_new > x[0] - step / 2:
            x[0] = -L_new
        if abs(x[-1] - L_new) > tol and L_new < x[-1] + step / 2:
            x[-1] = L_new

        centers = np.round(x, 8)
        return np.array(centers, dtype=float)

    @staticmethod
    def calculate_all_centers(domain_size, block_size, overlap):
        """
        Compute 3D centers for homogenization blocks with given overlap.

        :param domain_size: float
            Size of the domain along one axis.
        :param block_size: float
            Homogenization block size.
        :param overlap: float
            Overlap between adjacent blocks.
        :return: np.ndarray
            Array of shape (N, 3) containing 3D centers.
        """
        start = -domain_size / 2
        end = domain_size / 2
        stride = block_size - overlap

        num_intervals = int(np.ceil((end - start) / stride))
        n_centers = num_intervals + 1
        exact_stride = (end - start) / (n_centers - 1)

        centers_1d = start + exact_stride * np.arange(n_centers)
        z, y, x = np.meshgrid(centers_1d, centers_1d, centers_1d, indexing="ij")
        centers_3d = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

        print("centers_1d ", centers_1d)
        print("centers_3d.shape ", centers_3d.shape)
        return centers_3d

    @staticmethod
    def symmetrize_cond_tns(cond_tns):
        """
        Enforce symmetry in conductivity tensors by averaging the off-diagonal terms.

        :param cond_tns: np.ndarray, shape (N, 3)
            Conductivity tensor values in Voigt notation.
        :return: np.ndarray, shape (N, 2)
            Symmetrized conductivity tensors (Voigt notation reduced).
        """
        sym_values = (cond_tns[:, 1] + cond_tns[:, 2]) / 2
        cond_tns = np.delete(cond_tns, 2, axis=1)
        cond_tns[:, 1] = sym_values
        return cond_tns

    def _get_tensors(self, config_dict):
        """
        Load conductivity tensor population and their coordinates from files.

        :param config_dict: dict
            Configuration dictionary containing file paths for the tensor population.
        """
        cond_pop_file = config_dict["fine"]["cond_tn_pop_file"]
        print("cond_pop_file ", cond_pop_file)
        cond_pop_coords_file = config_dict["fine"]["cond_tn_pop_coords_file"]
        self._cond_tns = np.load(cond_pop_file)
        self._cond_tns_coords = np.load(cond_pop_coords_file)

    # ------------------------------------------------------------------
    # Field generation
    # ------------------------------------------------------------------

    def generate_field(self, reuse_sample=False, location_population=False):
        """
        Generate a stochastic random field by sampling conductivity tensors.

        :param reuse_sample: bool, optional
            If True, reuse an existing sample from disk instead of resampling.
        :param location_population: bool, optional
            If True, generate value at each location within original area from a population of tensors at that particular location
        :return: tuple
            sampled_tensors: np.ndarray, shape (N, 3, 3)
                Sampled conductivity tensors.
            centers: np.ndarray, shape (N, 3)
                Corresponding center coordinates.
        """
        # Randomly sample indices with replacement
        indices = np.random.choice(
            self._cond_tns.shape[0], size=self.centers_3d.shape[0], replace=True
        )
        sampled_tensors = self._cond_tns[indices]

        if location_population:
            print("location population")
            coord_values_dir = os.path.join(self._config_dict["fine"]["common_files_dir"], "coord_values")

            coord_to_index = {tuple(c): i for i, c in enumerate(self.centers_3d)}
            for coord in self._cond_tns_coords:
                filename = "coord_" + "_".join(str(c) for c in coord) + ".npy"
                path = os.path.join(coord_values_dir, filename)
                cond_tns_for_coord = np.load(path, mmap_mode="r")
                print("cond_tns_for_coord.shape ", cond_tns_for_coord.shape)
                choice_idx = np.random.randint(cond_tns_for_coord.shape[0])

                idx = coord_to_index.get(tuple(coord))
                if idx is not None:
                    sampled_tensors[idx] = cond_tns_for_coord[choice_idx]

        if reuse_sample:
            import glob, random
            # TODO: replace hard-coded filename with config variable
            name, ext = os.path.splitext("cond_tensors_values.npz")
            pattern = os.path.join(self._config_dict["fine"]["common_files_dir"], f"{name}_*{ext}")
            files = glob.glob(pattern)
            if files:
                selected_file = random.choice(files)
                print("Selected file:", selected_file)
                loaded_cond_tns_for_fine_sample = np.load(selected_file)["data"]
                # Map coords -> index and scatter into sampled_tensors
                coord_to_index = {tuple(c): i for i, c in enumerate(self.centers_3d)}
                for tensor, coord in zip(loaded_cond_tns_for_fine_sample, self._cond_tns_coords):
                    idx = coord_to_index.get(tuple(coord))
                    if idx is not None:
                        sampled_tensors[idx] = tensor
            else:
                print("No matching files found.")

        return sampled_tensors, self.centers_3d
