import os
import shutil
import sys
import numpy as np
import argparse
import mlmc
from mlmc.sampler import Sampler
from mlmc.sample_storage_hdf import SampleStorageHDF
from mlmc.sample_storage import Memory
from mlmc.sampling_pool import OneProcessPool
from mlmc.sampling_pool_pbs import SamplingPoolPBS
from homogenization.sim_sample_3d import DFMSim3D
from mlmc.moments import Legendre, Monomial
from mlmc.quantity.quantity import make_root_quantity
from mlmc.quantity.quantity_estimate import estimate_mean, moments
from mlmc import estimator
from mlmc.plot import diagnostic_plots as dp
import yaml
import scipy.stats as stats
import matplotlib.pyplot as plt
import zarr
import glob

from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit


class ProcessSimple:
    """
    Command-line interface for running MLMC simulations.

    This class handles execution commands such as running new simulations,
    renewing failed samples, collecting results, and post-processing.
    It configures simulation settings, storage, sampling pools, and MLMC parameters.
    """

    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('command', choices=['run', 'collect', 'renew', 'process'],
                            help='run - create new execution, '
                                 'collect - keep collected results and append to existing HDF file, '
                                 'renew - rerun failed samples with the same sample IDs, '
                                 'process - post-process stored results')
        parser.add_argument('work_dir', help='Work directory')
        parser.add_argument('scratch_dir', help='Scratch directory')
        parser.add_argument("-c", "--clean", default=False, action='store_true',
                            help="Clean before run (applies only to the run command)")
        parser.add_argument("-d", "--debug", default=False, action='store_true',
                            help="Keep temporary sample directories for debugging")

        args = parser.parse_args(sys.argv[1:])

        self.work_dir = os.path.abspath(args.work_dir)
        self.scratch_dir = os.path.abspath(args.scratch_dir)
        self.clean = args.clean  # Remove existing result files when running
        self.debug = args.debug  # Keep temporary sample directories
        self.use_pbs = False
        self.generate_samples_per_level = True
        self.n_levels = 3  # Number of MLMC levels
        self._levels_fine_srf_from_population = [0]
        self.n_moments = 3

        step_range = [12.5, 5]
        self.level_parameters = estimator.determine_level_parameters(self.n_levels, step_range)

        print("self.level_parameters ", self.level_parameters)

        self.n_samples = estimator.determine_n_samples(self.n_levels)

        if args.command == 'run':
            self.run()
        elif args.command == 'recollect':
            self.run(recollect=True)
        elif args.command == "process":
            self.process()
        else:
            self.clean = False
            self.run(renew=True) if args.command == 'renew' else self.run()

    def run(self, renew=False, recollect=False):
        """
        Run MLMC sampling workflow.

        :param renew: If True, reruns only failed samples with the same sample IDs.
        :param recollect: If True, recollects results and appends to existing HDF5 file (not supported).
        :return: None
        """
        os.makedirs(self.work_dir, mode=0o775, exist_ok=True)

        if self.clean:
            hdf_path = os.path.join(self.work_dir, f"mlmc_{self.n_levels}.hdf5")
            if os.path.exists(hdf_path):
                os.remove(hdf_path)

        sampler = self.setup_config(clean=self.clean)

        if recollect:
            raise NotImplementedError("Recollection mode not supported in this version")
        else:
            self.generate_jobs(sampler, n_samples=[3, 3], renew=renew, target_var=1e-4)
            self.all_collect(sampler)

    def setup_config(self, clean):
        """
        Prepare simulation configuration and create the MLMC sampler.

        :param clean: If True, remove existing files before starting.
        :return: Configured mlmc.Sampler instance.
        """
        self.set_environment_variables()
        sampling_pool = self.create_sampling_pool()

        with open(os.path.join(self.work_dir, "sim_config_3D.yaml"), "r") as f:
            sim_config_dict = yaml.load(f, Loader=yaml.FullLoader)

        if "nn_path_levels" in sim_config_dict:
            print("nn path levels dict ", sim_config_dict["nn_path_levels"])
            expected_keys = set(range(self.n_levels - 1, 0, -1))
            actual_keys = set(sim_config_dict["nn_path_levels"].keys())
            assert expected_keys.issubset(actual_keys), "'nn_path_levels' keys do not correspond to the set number of levels"

            new_dict = {}
            for level in range(1, self.n_levels):
                param_value = self.level_parameters[level][0]
                if level in sim_config_dict["nn_path_levels"]:
                    path = sim_config_dict["nn_path_levels"][level]
                    new_dict[param_value] = path
            sim_config_dict["nn_path_levels"] = new_dict

        sim_config_dict['fields_params'] = dict(model='exp', corr_length=0.1, sigma=1, mode_no=10000)
        sim_config_dict['field_template'] = "!FieldFE {mesh_data_file: \"$INPUT_DIR$/%s\", field_name: %s}"
        sim_config_dict['env'] = dict(flow123d=self.flow123d, gmsh=self.gmsh, gmsh_version=1)
        sim_config_dict['work_dir'] = self.work_dir
        sim_config_dict['scratch_dir'] = self.scratch_dir
        sim_config_dict['yaml_file'] = os.path.join(self.work_dir, '01_conductivity.yaml')
        sim_config_dict['yaml_file_homogenization'] = os.path.join(self.work_dir, 'flow_templ.yaml')
        sim_config_dict['yaml_file_homogenization_vtk'] = os.path.join(self.work_dir, 'flow_templ_vtk.yaml')
        sim_config_dict['level_parameters'] = self.level_parameters
        sim_config_dict['levels_fine_srf_from_population'] = self._levels_fine_srf_from_population

        simulation_factory = DFMSim3D(config=sim_config_dict, clean=clean)

        sample_storage = SampleStorageHDF(
            file_path=os.path.join(self.work_dir, f"mlmc_{self.n_levels}.hdf5"),
        )

        sampler = Sampler(sample_storage=sample_storage, sampling_pool=sampling_pool, sim_factory=simulation_factory,
                          level_parameters=self.level_parameters)

        return sampler


    def set_environment_variables(self):
        """
        Configure environment variables for running simulations.

        Automatically detects whether running on a cluster or locally
        to set paths and execution settings.
        """
        root_dir = os.path.abspath(self.work_dir)
        while root_dir != '/':
            root_dir, tail = os.path.split(root_dir)

        if tail == 'storage' or tail == 'auto':
            # Metacentrum cluster configuration
            self.sample_sleep = 30
            self.init_sample_timeout = 600
            self.sample_timeout = 60
            self.adding_samples_coef = 0.1
            self.flow123d = 'flow123d'
            self.gmsh = "/storage/liberec3-tul/home/martin_spetlik/astra/gmsh/bin/gmsh"
        else:
            # Local machine configuration
            self.sample_sleep = 1
            self.init_sample_timeout = 60
            self.sample_timeout = 60
            self.adding_samples_coef = 0.1
            self.flow123d = ["flow123d/ci-gnu:4.0.0a01_94c428", "flow123d"]
            self.gmsh = "/home/martin/gmsh/bin/gmsh"

    def create_sampling_pool(self):
        """
        Create and return a sampling pool.

        :return: Local or PBS-based sampling pool instance.
        """
        if not self.use_pbs:
            return OneProcessPool(work_dir=self.work_dir, debug=self.debug)

        sampling_pool = SamplingPoolPBS(work_dir=self.work_dir, debug=self.debug)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        pbs_config = dict(
            n_cores=1,
            n_nodes=1,
            select_flags={'cgroups': 'cpuacct', 'scratch_local': '10gb'},
            mem='2Gb',
            queue='charon',
            pbs_name='flow123d',
            walltime='48:00:00',
            optional_pbs_requests=[],
            home_dir='/storage/liberec3-tul/home/martin_spetlik/',
            python='python3',
            env_setting=['cd $MLMC_WORKDIR',
                         'source venv_torch/bin/activate',
                         f'export PYTHONPATH={script_dir}/:$PYTHONPATH'],
            scratch_dir=None
        )

        sampling_pool.pbs_common_setting(flow_3=True, **pbs_config)

        return sampling_pool

    def create_sampling_pool_cpu(self):
        """
        Create a PBS sampling pool configured for CPU execution.

        :return: PBS-based sampling pool instance for CPU jobs.
        """
        if not self.use_pbs:
            return OneProcessPool(work_dir=self.work_dir, debug=self.debug)

        output_dir = os.path.join(self.work_dir, "output")
        output_dir_gpu = os.path.join(self.work_dir, "output_gpu")
        print("output_dir ", output_dir)
        print("output_dir_gpu ", output_dir_gpu)
        shutil.move(output_dir, output_dir_gpu)

        sampling_pool = SamplingPoolPBS(work_dir=self.work_dir, debug=self.debug)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        pbs_config = dict(
            n_cores=1,
            n_nodes=1,
            select_flags={'cgroups': 'cpuacct', 'scratch_local': '24gb'},
            mem='24Gb',
            queue='charon',
            pbs_name='flow123d',
            walltime='24:00:00',
            optional_pbs_requests=[],
            home_dir='/storage/liberec3-tul/home/martin_spetlik/',
            python='python3',
            env_setting=['cd $MLMC_WORKDIR',
                         "trap 'clean_scratch' EXIT TERM",
                         'source venv_torch/bin/activate',
                         f'export PYTHONPATH={script_dir}/:$PYTHONPATH'],
            scratch_dir=None
        )

        sampling_pool.pbs_common_setting(flow_3=True, **pbs_config)

        return sampling_pool

    @staticmethod
    def create_zarr_file(n_samples, sampler):
        """
        Create a Zarr file for storing homogenization inputs and outputs.

        :param n_samples: Number of samples.
        :param sampler: Sampler instance.
        """
        if len(n_samples) > 1:
            raise Exception("Use single level, len(n_samples) should be 1")

        level_instance_obj = sampler._level_sim_objects[0]
        config_dict = level_instance_obj.config_dict

        input_shape_n_voxels = config_dict["sim_config"]["geometry"]["n_voxels"]
        input_shape_n_channels = 6  # 6 channels for cond_tn
        n_cond_tn_channels = 6  # 1 channel for cross_section
        #n_cross_section_channels = 1
        output_shape = (6,)
        n_samples = n_samples[0]

        zarr_file_path = os.path.join(config_dict["fine"]["common_files_dir"], DFMSim3D.ZARR_FILE)

        if not os.path.exists(zarr_file_path):
            zarr_file = zarr.open(zarr_file_path, mode='w')

            # # Create the 'inputs' dataset with the specified shape
            inputs = zarr_file.create_dataset('inputs',
                                              shape=(n_samples,) + (input_shape_n_channels, *input_shape_n_voxels),
                                              dtype='float32',
                                              chunks=(1, n_cond_tn_channels, *input_shape_n_voxels),
                                              fill_value=0)

            # Create the 'outputs' dataset with the specified shape
            outputs = zarr_file.create_dataset('outputs', shape=(n_samples,) + output_shape, dtype='float32',
                                               chunks=(1,  n_cond_tn_channels), fill_value=0)
            bulk_avg = zarr_file.create_dataset('bulk_avg', shape=(n_samples,) + output_shape, dtype='float32',
                                               chunks=(1, n_cond_tn_channels), fill_value=0)

            # Assign metadata to indicate channel names
            inputs.attrs['channel_names'] = ['cond_tn_0', 'cond_tn_1', 'cond_tn_2', 'cond_tn_3', 'cond_tn_4', 'cond_tn_5']
            outputs.attrs['channel_names'] = ['cond_tn_0', 'cond_tn_1', 'cond_tn_2', 'cond_tn_3', 'cond_tn_4', 'cond_tn_5']
            bulk_avg.attrs['channel_names'] = ['cond_tn_0', 'cond_tn_1', 'cond_tn_2', 'cond_tn_3', 'cond_tn_4', 'cond_tn_5']

    @staticmethod
    def save_cond_tns_per_coord(level_instance_obj):
        """
        Save conductivity tensor samples grouped by spatial coordinate.

        Loads sample values and coordinates, then stores one file per coordinate.
        """
        directory = level_instance_obj.config_dict["coarse"]["common_files_dir"]
        sample_cond_tns = []

        name, ext = os.path.splitext(DFMSim3D.COND_TN_VALUES_FILE)
        pattern = os.path.join(directory, f"{name}_*{ext}")

        for file_path in glob.glob(pattern):
            print("Found value file:", file_path)
            loaded_cond_tns = np.load(file_path)["data"]
            sample_cond_tns.append(list(loaded_cond_tns))

        name, ext = os.path.splitext(DFMSim3D.COND_TN_COORDS_FILE)
        pattern = os.path.join(directory, f"{name}_*{ext}")

        for file_path in glob.glob(pattern):
            print("Found coords file:", file_path)
            loaded_coords = np.load(file_path)["data"]
            break

        output_dir = os.path.join(directory, "coord_values")
        os.makedirs(output_dir, exist_ok=True)

        sample_cond_tns = np.array(sample_cond_tns).transpose(1, 0, 2, 3)
        print("sample_cond_tns.shape ", sample_cond_tns.shape)

        for coord, values in zip(loaded_coords, sample_cond_tns):
            coord_tuple = tuple(coord)
            filename = "coord_" + "_".join(str(c) for c in coord_tuple) + ".npy"
            path = os.path.join(output_dir, filename)
            np.save(path, np.array(values))
            print(f"Saved {path}")

    def generate_jobs(self, sampler, n_samples=None, renew=False, target_var=None):
        """
        Generate and schedule MLMC samples.

        :param sampler: Sampler instance used to schedule samples.
        :param n_samples: List defining number of samples per level, or None for automatic scheduling.
        :param renew: If True, rerun only failed samples.
        :param target_var: If provided, estimate required samples to reach target variance.
        :return: None
        """
        if len(n_samples) == 1:
            ProcessSimple.create_zarr_file(n_samples, sampler)

        if renew:
            sampler.ask_sampling_pool_for_samples()
            sampler.renew_failed_samples()
            sampler.ask_sampling_pool_for_samples(sleep=self.sample_sleep, timeout=self.sample_timeout)
        else:
            print("n_samples ", n_samples)
            if n_samples is not None:
                if self.use_pbs or self.generate_samples_per_level:
                    for level_id in reversed(range(self.n_levels)):
                        level_instance_obj = sampler._level_sim_objects[level_id]

                        if self.use_pbs and level_id == 0:
                            sampler._sampling_pool = self.create_sampling_pool_cpu()

                        sampler.schedule_samples(level_id=level_id, n_samples=n_samples[level_id])
                        running = 1
                        while running > 0:
                            running = 0
                            running += sampler.ask_sampling_pool_for_samples(sleep=self.sample_sleep, timeout=1e-5)
                            print("N running: ", running)

                        print("level_id ", level_id)
                        if level_id > 0:
                            cond_tn_pop_file = os.path.join(level_instance_obj.config_dict["coarse"]["common_files_dir"],
                                                             DFMSim3D.COND_TN_POP_FILE)
                            cond_tn_pop_coords_file = os.path.join(level_instance_obj.config_dict["coarse"]["common_files_dir"],
                                DFMSim3D.COND_TN_POP_COORDS_FILE)

                            print("cond_tn_pop_file ", cond_tn_pop_file)
                            if cond_tn_pop_file is not None:

                                if "generate_srf_from_hom_location_population" in level_instance_obj.config_dict["sim_config"]:
                                    if level_id - 1 in level_instance_obj.config_dict["sim_config"]["generate_srf_from_hom_location_population"] and \
                                            level_instance_obj.config_dict["sim_config"]["generate_srf_from_hom_location_population"][level_id - 1]:
                                        ProcessSimple.save_cond_tns_per_coord(level_instance_obj)

                                sample_cond_tns = []

                                directory = level_instance_obj.config_dict["coarse"]["common_files_dir"]
                                name, ext = os.path.splitext(DFMSim3D.COND_TN_VALUES_FILE)
                                pattern = os.path.join(directory, f"{name}_*{ext}")

                                for file_path in glob.glob(pattern):
                                    print("Found file:", file_path)
                                    loaded_cond_tns = np.load(file_path)['data']
                                    if len(sample_cond_tns) < 15000:
                                        sample_cond_tns.extend(list(loaded_cond_tns))

                                name, ext = os.path.splitext(DFMSim3D.COND_TN_COORDS_FILE)
                                pattern = os.path.join(directory, f"{name}_*{ext}")

                                for file_path in glob.glob(pattern):
                                    print("Found file:", file_path)
                                    loaded_coords = np.load(file_path)['data']
                                    break

                                print("samples_cond_tns ", sample_cond_tns)
                                np.save(cond_tn_pop_file, sample_cond_tns)
                                np.save(cond_tn_pop_coords_file, loaded_coords)

                else:
                    sampler.set_initial_n_samples(n_samples)
                    sampler.schedule_samples()
            else:
                sampler.set_initial_n_samples()
                sampler.schedule_samples()

            sampler.ask_sampling_pool_for_samples(sleep=self.sample_sleep, timeout=self.sample_timeout)
            self.all_collect(sampler)

            if target_var is not None:
                root_quantity = make_root_quantity(storage=sampler.sample_storage,
                                                   q_specs=sampler.sample_storage.load_result_format())

                estimate_obj = self.get_estimate_obj(root_quantity, sampler.sample_storage)

                variances, n_ops = estimate_obj.estimate_diff_vars_regression(sampler._n_scheduled_samples)
                n_estimated = estimator.estimate_n_samples_for_target_variance(target_var, variances, n_ops,
                                                                               n_levels=sampler.n_levels)

                print("variances ", variances)
                print("n ops ", n_ops)
                print("n estimated ", n_estimated)
                exit()

    def get_estimate_obj(self, quantity, sample_storage, quantile=0.001):
        """
        Create an Estimate object for a given quantity.

        :param quantity: Quantity to estimate.
        :param sample_storage: Sample storage object containing MLMC samples.
        :param quantile: Quantile used for domain estimation.
        :return: Estimate instance.
        """
        true_domain = mlmc.estimator.Estimate.estimate_domain(quantity, sample_storage, quantile=quantile)
        moments_fn = Monomial(self.n_moments, true_domain)
        estimate_obj = mlmc.estimator.Estimate(quantity=quantity, sample_storage=sample_storage, moments_fn=moments_fn)

        return estimate_obj

    def all_collect(self, sampler):
        """
        Collect finished samples from the sampling pool.

        :param sampler: Sampler instance managing the jobs.
        :return: None
        """
        running = 1
        while running > 0:
            running = 0
            running += sampler.ask_sampling_pool_for_samples(sleep=self.sample_sleep, timeout=1e-5)
            print("N running: ", running)

    def _remove_outliers(self, data):
        """
        Remove statistical outliers using the IQR method.

        :param data: Input array.
        :return: Filtered array with outliers removed.
        """
        Q1 = np.percentile(data, 10)
        Q3 = np.percentile(data, 90)
        IQR = Q3 - Q1
        threshold = 1.5
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        cleaned_data = data[~((data < lower_bound) | (data > upper_bound))]
        return cleaned_data

    def two_sample_ztest(self, data1, data2):
        """
        Perform a two-sample z-test on two datasets.

        :param data1: First sample array.
        :param data2: Second sample array.
        :return: Tuple (z_statistic, p_value).
        """
        mean1, mean2 = np.mean(data1), np.mean(data2)
        std1, std2 = np.std(data1), np.std(data2)
        n1, n2 = len(data1), len(data2)
        sed = np.sqrt(std1 ** 2 / n1 + std2 ** 2 / n2)
        z = (mean1 - mean2) / sed
        p_value = stats.norm.cdf(z) * 2
        return z, p_value

    def compare_means(self, samples_a, samples_b, title=""):
        """
        Compare means of two samples using t-test.

        :param samples_a: First sample array.
        :param samples_b: Second sample array.
        :param title: Optional label for output message.
        :return: None
        """
        statistic, p_value = stats.ttest_ind(np.squeeze(samples_a), np.squeeze(samples_b), equal_var=False)
        alpha = 0.05
        print("p value ", p_value)
        if p_value < alpha:
            print(f"{title} are significantly different.")
        else:
            print(f"There is no significant difference between {title}")

    def process(self):
         raise NotImplementedError("Use postprocess_mlmc_dfm_3d.py code")


    @staticmethod
    def empirical_variogram(pos, values, nbins=30):

        D = squareform(pdist(pos))
        print("D ", D)
        V = squareform(pdist(values[:, None], metric='sqeuclidean')) / 2
        bins = np.linspace(0, D.max(), nbins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        variogram = []

        for i in range(nbins):
            mask = (D >= bins[i]) & (D < bins[i + 1])
            if np.any(mask):
                variogram.append(np.mean(V[mask]))
            else:
                variogram.append(np.nan)
        return bin_centers, np.array(variogram)

    @staticmethod
    def variogram_model(h, C, L):
        return C * (1 - np.exp(-h / L))  # Exponential model

    @staticmethod
    def fit_variogram_model(distances, gamma_vals):
        mask = ~np.isnan(gamma_vals)
        popt, _ = curve_fit(ProcessSimple.variogram_model, distances[mask], gamma_vals[mask], bounds=(0, np.inf))
        return popt  # C (sill), L (correlation length)

    @staticmethod
    def get_corr_lengths(coords, cond_tns, use_log=False):
        from bgem.upscale import fem_plot, fem, voigt_to_tn, tn_to_voigt

        components = tn_to_voigt(cond_tns)
        print("components.shape ", components.shape)

        if use_log:
            components[..., 0] = np.log(components[..., 0])
            components[..., 1] = np.log(components[..., 1])
            components[..., 2] = np.log(components[..., 2])

        results = {}
        for i in range(components.shape[-1]):
            values = np.squeeze(components[..., i])
            print("values.shape ", values.shape)
            print("coords.shape ", coords.shape)
            dists, gamma = ProcessSimple.empirical_variogram(coords, values)
            C, L = ProcessSimple.fit_variogram_model(dists, gamma)

            print("Component {}, still: {}, correlation_length: {}".format(i, C, L))
            results[i] = {'sill': C, 'correlation_length': L}

            # Optional plot
            plt.figure()
            plt.plot(dists, gamma, 'o', label='Empirical')
            plt.plot(dists, ProcessSimple.variogram_model(dists, C, L), '-', label=f'Model: L={L:.2f}')
            plt.xlabel("Distance")
            plt.ylabel("Semivariance")
            plt.title(f"Variogram of $T_{{{i}}}$")
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    ProcessSimple()
