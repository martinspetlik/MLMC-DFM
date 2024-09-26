import copy
import os
import sys
import numpy as np
import argparse
#import mlmc.tool.simple_distribution
import mlmc
from mlmc.sampler import Sampler
from mlmc.sample_storage_hdf import SampleStorageHDF
from mlmc.sample_storage import Memory
from mlmc.sampling_pool import OneProcessPool
from mlmc.sampling_pool_pbs import SamplingPoolPBS
from homogenization.sim_sample_3d import DFMSim3D
#from homogenization.both_sample import FineHomSRFGstools
from mlmc.moments import Legendre, Monomial
from mlmc.tool.process_base import ProcessBase
from mlmc.quantity.quantity import make_root_quantity
from mlmc.quantity.quantity_estimate import estimate_mean, moments
from mlmc import estimator
from mlmc.plot import diagnostic_plots as dp
import yaml
import ruamel.yaml
import scipy.stats as stats
import matplotlib.pyplot as plt
import zarr


class ProcessSimple:
    # @TODO: generate more samples, with new seed

    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('command', choices=['run', 'collect', 'renew', 'process'],
                            help='run - create new execution,'
                                 'collect - keep collected, append existing HDF file'
                                 'renew - renew failed samples, run new samples with failed sample ids (which determine random seed)')
        parser.add_argument('work_dir', help='Work directory')
        parser.add_argument('scratch_dir', help='Scratch directory')
        parser.add_argument("-c", "--clean", default=False, action='store_true',
                            help="Clean before run, used only with 'run' command")
        parser.add_argument("-d", "--debug", default=False, action='store_true', help="Keep sample directories")

        args = parser.parse_args(sys.argv[1:])

        self.work_dir = os.path.abspath(args.work_dir)
        self.scratch_dir = os.path.abspath(args.scratch_dir)
        # Add samples to existing ones
        self.clean = args.clean
        # Remove HDF5 file, start from scratch
        self.debug = args.debug
        # 'Debug' mode is on - keep sample directories
        self.use_pbs = False
        self.generate_samples_per_level = True
        # Use PBS sampling pool
        self.n_levels = 1
        self.n_moments = 3
        # Number of MLMC levels

        step_range = [10, 5]

        # step_range [simulation step at the coarsest level, simulation step at the finest level]

        # Determine level parameters at each level (In this case, simulation step at each level) are set automatically
        self.level_parameters = estimator.determine_level_parameters(self.n_levels, step_range)
        #self.level_parameters = [self.level_parameters[4]]

        #self.level_parameters = [12.3607, 5.3459, 2.3121, 1, 0.4325]
        #self.n_levels = 1
        print("self.level_parameters ", self.level_parameters)

        # Determine number of samples at each level
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
        Run MLMC
        :param renew: If True then rerun failed samples with same sample id
        :return: None
        """
        # Create working directory if necessary
        os.makedirs(self.work_dir, mode=0o775, exist_ok=True)

        if self.clean:
            # Remove HFD5 file
            if os.path.exists(os.path.join(self.work_dir, "mlmc_{}.hdf5".format(self.n_levels))):
                os.remove(os.path.join(self.work_dir, "mlmc_{}.hdf5".format(self.n_levels)))

        # Create sampler (mlmc.Sampler instance) - crucial class which actually schedule sample
        sampler = self.setup_config(clean=self.clean)

        # Schedule samples
        # self.generate_jobs(sampler, n_samples=[5], renew=renew)

        # self.generate_jobs(sampler, n_samples=None, renew=renew, target_var=1e-3)
        # self.generate_jobs(sampler, n_samples=[500, 500], renew=renew, target_var=1e-5)
        if recollect:
            raise NotImplementedError("Not supported in released version")
        else:
            self.generate_jobs(sampler, n_samples=[2], renew=renew)#[1, 1, 1, 3, 3, 3, 3], renew=renew)
            #self.generate_jobs(sampler, n_samples=[100, 2],renew=renew, target_var=1e-5)
            self.all_collect(sampler)  # Check if all samples are finished

    def setup_config(self, clean):
        """
        Simulation dependent configuration
        :param clean: bool, If True remove existing files
        :return: mlmc.sampler instance
        """
        # Set pbs config, flow123d, gmsh, ..., random fields are set in simulation class
        self.set_environment_variables()

        # Create Pbs sampling pool
        sampling_pool = self.create_sampling_pool()

        with open(os.path.join(self.work_dir, "sim_config_3D.yaml"), "r") as f:
            sim_config_dict = yaml.load(f, Loader=yaml.FullLoader)
            print("sim_config_dict ", sim_config_dict)

        sim_config_dict['fields_params'] = dict(model='exp', corr_length=0.1, sigma=1, mode_no=10000)
        sim_config_dict['field_template'] = "!FieldFE {mesh_data_file: \"$INPUT_DIR$/%s\", field_name: %s}"
        sim_config_dict['env'] = dict(flow123d=self.flow123d, gmsh=self.gmsh, gmsh_version=1)  # The Environment.
        sim_config_dict['work_dir'] = self.work_dir
        sim_config_dict['scratch_dir'] = self.scratch_dir
        sim_config_dict['yaml_file'] = os.path.join(self.work_dir, '01_conductivity.yaml')
        sim_config_dict['yaml_file_homogenization'] = os.path.join(self.work_dir, 'flow_templ.yaml')
        sim_config_dict['yaml_file_homogenization_vtk'] = os.path.join(self.work_dir, 'flow_templ_vtk.yaml')
        sim_config_dict['level_parameters'] = self.level_parameters

        print("yaml file hom ", os.path.join(self.work_dir, 'flow_templ.yaml'))

        # simulation_config = {
        #     'work_dir': self.work_dir,
        #     'env': dict(flow123d=self.flow123d, gmsh=self.gmsh, gmsh_version=1),  # The Environment.
        #     'yaml_file': os.path.join(self.work_dir, '01_conductivity.yaml'),
        #     'geo_file': os.path.join(self.work_dir, 'square_1x1.geo'),
        #     'fields_params': dict(model='exp', corr_length=0.1, sigma=1, mode_no=10000),
        #     'field_template': "!FieldFE {mesh_data_file: \"$INPUT_DIR$/%s\", field_name: %s}"
        # }

        # Create simulation factory
        simulation_factory = DFMSim3D(config=sim_config_dict, clean=clean)

        # Create HDF sample storage
        sample_storage = SampleStorageHDF(
            file_path=os.path.join(self.work_dir, "mlmc_{}.hdf5".format(self.n_levels)),
            # append=self.append
        )

        #sample_storage = Memory()

        # Create sampler, it manages sample scheduling and so on
        sampler = Sampler(sample_storage=sample_storage, sampling_pool=sampling_pool, sim_factory=simulation_factory,
                          level_parameters=self.level_parameters)

        return sampler

    def set_environment_variables(self):
        """
        Set pbs config, flow123d, gmsh
        :return: None
        """
        root_dir = os.path.abspath(self.work_dir)
        while root_dir != '/':
            root_dir, tail = os.path.split(root_dir)

        if tail == 'storage' or tail == 'auto':
            # Metacentrum
            self.sample_sleep = 30
            self.init_sample_timeout = 600
            self.sample_timeout = 60
            self.adding_samples_coef = 0.1
            self.flow123d = 'flow123d'  # "/storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d"
            self.gmsh = "/storage/liberec3-tul/home/martin_spetlik/astra/gmsh/bin/gmsh"
        else:
            # Local
            self.sample_sleep = 1
            self.init_sample_timeout = 60
            self.sample_timeout = 60
            self.adding_samples_coef = 0.1
            #self.flow123d = ["flow123d/flow123d-gnu:3.1.0", "flow123d"]  # "/home/martin/flow123d/bin/fterm --no-term rel run"
            self.flow123d = ["flow123d/ci-gnu:4.0.0a01_94c428", "flow123d"]
            self.gmsh = "/home/martin/gmsh/bin/gmsh"

    def create_sampling_pool(self):
        """
        Initialize sampling pool, object which
        :return: None
        """
        if not self.use_pbs:
            return OneProcessPool(work_dir=self.work_dir, debug=self.debug)  # Everything runs in one process

        # Create PBS sampling pool
        sampling_pool = SamplingPoolPBS(work_dir=self.work_dir, debug=self.debug)
        # sampling_pool = OneProcessPool(work_dir=self.work_dir, debug=self.debug)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        pbs_config = dict(
            n_cores=1,
            n_nodes=1,
            select_flags={'cgroups': 'cpuacct', 'scratch_local': '10gb'},
            mem='2Gb',
            queue='charon',
            pbs_name='flow123d',
            walltime='48:00:00',
            optional_pbs_requests=[],  # e.g. ['#PBS -m ae', ...]
            home_dir='/storage/liberec3-tul/home/martin_spetlik/',
            python='python3',
            env_setting=['cd $MLMC_WORKDIR',
                         # 'module load python/3.8.0-gcc',
                         'source venv_torch/bin/activate',
                         'export PYTHONPATH={}/:$PYTHONPATH'.format(script_dir)
                         # 'module use /storage/praha1/home/jan-hybs/modules',
                         # 'module load flow123d',
                         # 'module unload python-3.6.2-gcc',
                         # 'module unload python36-modules-gcc'
                         ],
            scratch_dir=None
        )

        sampling_pool.pbs_common_setting(flow_3=True, **pbs_config)

        return sampling_pool

    # @staticmethod
    # def prepare_distr_data(level_instance_obj):
    #     config_dict = level_instance_obj.config_dict
    #
    #     cond_tn_pop_file = os.path.join(
    #         level_instance_obj.config_dict["fine"]["common_files_dir"],
    #         DFMSim.COND_TN_POP_FILE)
    #     if os.path.exists(cond_tn_pop_file):
    #         config_dict["fine"]["cond_tn_pop_file"] = cond_tn_pop_file
    #
    #     if "nn_path" in config_dict["sim_config"]:
    #         pred_cond_tn_pop_file = os.path.join(
    #             level_instance_obj.config_dict["fine"]["common_files_dir"],
    #             DFMSim.PRED_COND_TN_POP_FILE)
    #         if os.path.exists(pred_cond_tn_pop_file):
    #             config_dict["fine"]["pred_cond_tn_pop_file"] = pred_cond_tn_pop_file
    #
    #     sample_cond_tns = os.path.join(config_dict["fine"]["common_files_dir"],
    #                                    DFMSim.SAMPLES_COND_TNS_DIR)
    #     if os.path.exists(sample_cond_tns):
    #         config_dict["fine"]["sample_cond_tns"] = sample_cond_tns
    #
    #     if "pred_cond_tn_pop_file" in config_dict["fine"]:
    #         print("load pred cond tn pop file")
    #         cond_pop_file = config_dict["fine"]["pred_cond_tn_pop_file"]
    #     else:
    #         print("load cond pop file")
    #         cond_pop_file = config_dict["fine"]["cond_tn_pop_file"]
    #
    #     print("cond pop file ", cond_pop_file)
    #     cond_tns = np.load(cond_pop_file)
    #     print("cond tns ", cond_tns)
    #
    #     if config_dict["sim_config"]["bulk_fine_sample_model"] == "srf_gstools":
    #         FineHomSRFGstools(cond_tns, config_dict)

    @staticmethod
    def create_zarr_file(n_samples, sampler):
        if len(n_samples) > 1:
            raise Exception

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
            #inputs[:, :, :, :, :] = np.zeros((n_samples, n_cond_tn_channels, *input_shape_n_voxels,
            #                                                         ))  # Populate the first 6 channels
            # inputs[:, :, :, :, n_cond_tn_channels] = np.random.rand(n_samples,
            #                                                         *input_shape_n_voxels)  # Populate the last channel

            # Create the 'outputs' dataset with the specified shape
            outputs = zarr_file.create_dataset('outputs', shape=(n_samples,) + output_shape, dtype='float32',
                                               chunks=(1,  n_cond_tn_channels), fill_value=0)
            #outputs[:, :] = np.zeros((n_samples, n_cond_tn_channels))  # Populate the first 6 channels
            # outputs[:, n_cond_tn_channels] = np.random.rand(n_samples)  # Populate the last channel

            # Assign metadata to indicate channel names
            inputs.attrs['channel_names'] = ['cond_tn_0', 'cond_tn_1', 'cond_tn_2', 'cond_tn_3', 'cond_tn_4', 'cond_tn_5']
            outputs.attrs['channel_names'] = ['cond_tn_0', 'cond_tn_1', 'cond_tn_2', 'cond_tn_3', 'cond_tn_4', 'cond_tn_5']

    def generate_jobs(self, sampler, n_samples=None, renew=False, target_var=None):
        """
        Generate level samples
        :param n_samples: None or list, number of samples for each level
        :param renew: rerun failed samples with same random seed (= same sample id)
        :return: None
        """

        # Create zarr dir only for homogenization samples generation
        if len(n_samples) == 1:
            ProcessSimple.create_zarr_file(n_samples, sampler)


        if renew:
            sampler.ask_sampling_pool_for_samples()
            sampler.renew_failed_samples()
            sampler.ask_sampling_pool_for_samples(sleep=self.sample_sleep, timeout=self.sample_timeout)
        else:
            if n_samples is not None:
                if self.use_pbs or self.generate_samples_per_level:

                    for level_id in reversed(range(self.n_levels)):
                        #level_instance_obj = sampler._level_sim_objects[level_id]

                        sampler.schedule_samples(level_id=level_id, n_samples=n_samples[level_id])
                        #l_n_scheduled = sampler._n_scheduled_samples[level_id]
                        running = 1
                        while running > 0:
                            running = 0
                            running += sampler.ask_sampling_pool_for_samples(sleep=self.sample_sleep, timeout=1e-5)
                            print("N running: ", running)
                            #print("l_n_scheduled - running ", l_n_scheduled - running)
                            # if l_n_scheduled - running > 20:
                            #     break
                        exit()
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

                moments_fn = self.set_moments(root_quantity, sampler.sample_storage, n_moments=self.n_moments)
                estimate_obj = estimator.Estimate(root_quantity, sample_storage=sampler.sample_storage,
                                                  moments_fn=moments_fn)

                # New estimation according to already finished samples
                variances, n_ops = estimate_obj.estimate_diff_vars_regression(sampler._n_scheduled_samples)
                n_estimated = estimator.estimate_n_samples_for_target_variance(target_var, variances, n_ops,
                                                                               n_levels=sampler.n_levels)

                print("variances ", variances)
                print("n ops ", n_ops)
                print("n estimated ", n_estimated)
                exit()

                # Loop until number of estimated samples is greater than the number of scheduled samples
                while not sampler.process_adding_samples(n_estimated, self.sample_sleep, self.adding_samples_coef,
                                                         timeout=self.sample_timeout):
                    # New estimation according to already finished samples
                    variances, n_ops = estimate_obj.estimate_diff_vars_regression(sampler._n_scheduled_samples)
                    n_estimated = estimator.estimate_n_samples_for_target_variance(target_var, variances, n_ops,
                                                                                   n_levels=sampler.n_levels)

    def set_moments(self, quantity, sample_storage, n_moments=5, quantile=0.01):
        true_domain = estimator.estimate_domain(quantity, sample_storage, quantile=quantile)
        return Legendre(n_moments, true_domain)

    def all_collect(self, sampler):
        """
        Collect samples
        :param sampler: mlmc.Sampler object
        :return: None
        """
        running = 1
        while running > 0:
            running = 0
            running += sampler.ask_sampling_pool_for_samples(sleep=self.sample_sleep, timeout=1e-5)
            print("N running: ", running)

    def _remove_outliers(self, data):
        Q1 = np.percentile(data, 10)
        Q3 = np.percentile(data, 90)

        # Calculate IQR
        IQR = Q3 - Q1

        # Define threshold for outlier detection
        threshold = 1.5

        # Define upper and lower bounds
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        # Find indices of outliers
        outlier_indices = np.where((data < lower_bound) | (data > upper_bound))

        # Remove outliers
        cleaned_data = data[~((data < lower_bound) | (data > upper_bound))]

        return cleaned_data

    def two_sample_ztest(self, data1, data2):
        # Calculate means and standard deviations
        mean1, mean2 = np.mean(data1), np.mean(data2)
        std1, std2 = np.std(data1), np.std(data2)

        # Calculate standard error of the difference between means
        n1, n2 = len(data1), len(data2)
        sed = np.sqrt(std1 ** 2 / n1 + std2 ** 2 / n2)

        # Calculate Z-statistic
        z = (mean1 - mean2) / sed

        # Calculate p-value
        p_value = stats.norm.cdf(z) * 2  # Two-tailed test

        return z, p_value

    def compare_means(self, samples_a, samples_b, title=""):
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
        #statistic, p_value = self.two_sample_ztest(np.squeeze(samples_a), np.squeeze(samples_b))
        #statistic, p_value = stats.chisquare(np.squeeze(samples_a), np.squeeze(samples_b))
        alpha = 0.05
        print("p value ", p_value)
        # Check if the p-value is less than alpha
        if p_value < alpha:
            print("{} are significantly different.".format(title))
        else:
            print("There is no significant difference between {}".format(title))

    def process(self):
        div_coef = 1e-6
        sample_storage = SampleStorageHDF(file_path=os.path.join(self.work_dir, "mlmc_{}.hdf5".format(self.n_levels)))
        sample_storage.chunk_size = 1e8
        result_format = sample_storage.load_result_format()
        #root_quantity = make_root_quantity(sample_storage, result_format)

        # conductivity = root_quantity['cond_tn']
        # time = conductivity[1]  # times: [1]
        # location = time['0']  # locations: ['0']
        # q_value = location[0, 0]


        #l_0_samples = estimate_obj.get_level_samples(level_id=0)
        #l_1_samples = estimate_obj.get_level_samples(level_id=1)

        #l_2_samples = estimator.get_level_samples(level_id=2)
        #l_3_samples = estimator.get_level_samples(level_id=3)
        #l_4_samples = estimator.get_level_samples(level_id=4)

        #print("l 0 samples shape ", np.squeeze(l_0_samples).shape)
        #print("l 1 samples shape ", np.squeeze(l_1_samples[..., 0]).shape)

        # print("l_0_samples.var ", np.var(np.squeeze(l_0_samples)[:10000]))
        # print("fine l_1_samples.var ", np.var(np.squeeze(l_1_samples[..., 0])))
        # print("fine l_2_samples.var ", np.var(np.squeeze(l_2_samples[..., 0])))
        # print("fine l_3_samples.var ", np.var(np.squeeze(l_3_samples[..., 0])))
        # print("fine l_4_samples.var ", np.var(np.squeeze(l_4_samples[..., 0])))

        target_var = 1e-6
        root_quantity = make_root_quantity(storage=sample_storage,
                                           q_specs=sample_storage.load_result_format())

        import mlmc.quantity.quantity_estimate as qe
        #sample_vector = [100, 100, 100]
        #root_quantity = root_quantity.subsample(sample_vector)  # out of [100, 80, 50, 30, 10]
        # moments_quantity = qe.moments(root_quantity_subsamples, moments_fn=moments_fn, mom_at_bottom=True)
        # mult_chunks_moments_mean = estimate_mean(moments_quantity)
        # mult_chunks_length_mean = mult_chunks_moments_mean['length']
        # mult_chunks_time_mean = mult_chunks_length_mean[1]
        # mult_chunks_location_mean = mult_chunks_time_mean['10']
        # mult_chunks_value_mean = mult_chunks_location_mean[0]

        # cond_tn_quantity = root_quantity["conductivity"]
        # time_mean = cond_tn_quantity[1]  # times: [1]
        # location_mean = time_mean['0']  # locations: ['0']
        # k_xx = location_mean[0]

        cond_tn_quantity = root_quantity["cond_tn"]
        time_mean = cond_tn_quantity[1]  # times: [1]
        location_mean = time_mean['0']  # locations: ['0']
        k_xx = location_mean[2]
        q_value = k_xx #cond_tn_quantity#k_xx
        #q_value = np.array([location_mean[0], location_mean[1], location_mean[2]])

        # @TODO: How to estimate true_domain?
        quantile = 1e-3
        true_domain = mlmc.estimator.Estimate.estimate_domain(q_value, sample_storage, quantile=quantile)
        print("true domain ", true_domain)
        #true_domain = (3.61218304e-07, 3.645725840000398e-05)
        #true_domain = (4.1042996300000003e-07, 2.9956777900000792e-05)
        #true_domain = (3.61218304e-07, 2.9956777900000792e-05)
        #moments_fn = Legendre(self.n_moments, true_domain)
        moments_fn = Monomial(self.n_moments, true_domain)# ref_domain=true_domain)

        # n_ops = np.array(sample_storage.get_n_ops())
        # print("n ops ", n_ops)
        # print("n ops ", n_ops[:, 0] / n_ops[:, 1])

        print("sample storage n collected ", sample_storage.get_n_collected())

        estimate_obj = mlmc.estimator.Estimate(quantity=q_value, sample_storage=sample_storage, moments_fn=moments_fn)

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
        estimate_obj_moments_quantity = mlmc.estimator.Estimate(quantity=moments_quantity, sample_storage=sample_storage, moments_fn=moments_fn)

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
                                                                       n_levels=self.n_levels)

        #n_ops = [64, 150, 300]
        #n_ops[0] = 85

        #n_ops = [36, 75, 120]

        #n_ops = [36.23067711011784, 848.5677567292186, 3115.1620610555015]

        #n_ops = [50.74231049047417, 150.8056385226611, 300.4481985032301]

        print("level variances ", variances)
        print("n ops ", n_ops)

        print("n estimated ", n_estimated)

        #n_ops = [2.1530759945526734, 424.25148745817296, 2510.9868827356786]



        #n_ops = [10, 25, 120]
        #n_ops = [130, 150]
        print("total cost ", np.sum(n_ops * n_estimated)) #np.array(sample_storage.get_n_collected())))

        # moments_quantity = moments(root_quantity, moments_fn=moments_fn, mom_at_bottom=True)
        # moments_mean = estimate_mean(moments_quantity)
        # conductivity_mean = moments_mean['cond_tn']
        # time_mean = conductivity_mean[1]  # times: [1]
        # location_mean = time_mean['0']  # locations: ['0']
        # values_mean = location_mean[0]  # result shape: (1,)
        # value_mean = values_mean[0]
        # assert value_mean.mean == 1

        l_0_samples = estimate_obj.get_level_samples(level_id=0, n_samples=sample_storage.get_n_collected()[0])

        if self.n_levels == 1:
            print("l0 var", np.var(l_0_samples))

            print("L0 mean {}".format(np.mean(l_0_samples)))

        elif self.n_levels == 2:
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

            print("l1 fine moments ", l1_fine_moments.shape)
            print("l1 coarse moments ", l1_coarse_moments)


            l1_fine_samples = l_1_samples[:, 0]
            l1_coarse_samples = l_1_samples[:, 1]


            log = True
            if log:
                l1_fine_samples = np.log10(l1_fine_samples)
                l1_fine_samples = l1_fine_samples[~np.isinf(l1_fine_samples)]

                l1_coarse_samples = np.log10(l1_coarse_samples)
                l1_coarse_samples = l1_coarse_samples[~np.isinf(l1_coarse_samples)]

                l_0_samples = np.log10(l_0_samples)
                l_0_samples = l_0_samples[~np.isinf(l_0_samples)]

            l1_diff = l1_fine_samples - l1_coarse_samples

            #print("var l1_fine_moments ", np.var(l1_fine_moments, axis=1))
            #print("var l1_coarse_moments ", np.var(l1_coarse_moments, axis=1))

            moment_1_cov_matrix = np.cov(l1_fine_moments[0, :], l1_coarse_moments[0, :])
            moment_2_cov_matrix = np.cov(l1_fine_moments[1, :], l1_coarse_moments[1, :])
            print(" moment_1_cov_matrix  ",  moment_1_cov_matrix )
            print(" moment_2_cov_matrix  ", moment_2_cov_matrix)
            moment_1_tot_var = moment_1_cov_matrix[0, 0] +  moment_1_cov_matrix[1, 1] - 2*moment_1_cov_matrix[1, 0]
            moment_2_tot_var = moment_2_cov_matrix[0, 0] + moment_2_cov_matrix[1, 1] - 2*moment_2_cov_matrix[1, 0]

            print("moment 1 tot var ", moment_1_tot_var)
            print("moment 2 tot var ", moment_2_tot_var)

            # print("l1 diff ", l1_diff)
            print("l1 diff var", np.var(l1_diff))
            print("l0 var", np.var(l_0_samples))

            print("L0 mean {}, L1 coarse mean: {}".format(np.mean(l_0_samples), np.mean(l1_coarse_samples)))
            self.compare_means(l_0_samples, l1_coarse_samples, title="L0 fine mean and L1 coarse mean")

            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
            axes.hist(l1_coarse_samples, bins=100, density=True, label="L1 coarse")
            axes.hist(np.squeeze(l_0_samples), bins=100, density=True, label="L0 samples", alpha=0.5)
            fig.suptitle("L1 coarse L0 samples, log: {}".format(log))
            fig.legend()
            fig.savefig("L1_coarse_L0_samples.pdf")
            plt.show()
            #print("L1 fine mean {}, L2 coarse mean: {}".format(np.mean(l1_fine_samples), np.mean(l2_coarse_samples)))
        else:
            l_1_samples = estimate_obj.get_level_samples(level_id=1, n_samples=sample_storage.get_n_collected()[1])
            l_1_samples = np.squeeze(l_1_samples)# / div_coef

            l_0_samples_moments = estimate_obj_moments_quantity.get_level_samples(level_id=0, n_samples=sample_storage.get_n_collected()[0])

            mask = np.any(np.isnan(l_0_samples_moments), axis=0).any(axis=1)
            l_0_samples_moments = l_0_samples_moments[..., ~mask, :]
            l_0_samples_moments = l_0_samples_moments[1:, :, 0]

            #print("l0 samples moments ", l_0_samples_moments.shape)

            l_1_samples_moments = estimate_obj_moments_quantity.get_level_samples(level_id=1, n_samples=sample_storage.get_n_collected()[1])
            mask = np.any(np.isnan(l_1_samples_moments), axis=0).any(axis=1)
            l_1_samples_moments = l_1_samples_moments[..., ~mask, :]

            #print("l1 sample momoments shape ", l_1_samples_moments.shape)

            l1_fine_moments = l_1_samples_moments[1:, :, 0]
            l1_coarse_moments = l_1_samples_moments[1:, :, 1]

            l_2_samples = estimate_obj.get_level_samples(level_id=2, n_samples=sample_storage.get_n_collected()[2])
            l_2_samples = np.squeeze(l_2_samples) #/ div_coef
            #print("squeezed l1 samples ", l_1_samples)
            # print("l_1_samples[:, 0] ", l_1_samples[:, 0])
            # print("l_1_samples[:, 1] ", l_1_samples[:, 1])

            l_2_samples_moments = estimate_obj_moments_quantity.get_level_samples(level_id=2, n_samples=sample_storage.get_n_collected()[2])
            mask = np.any(np.isnan(l_2_samples_moments), axis=0).any(axis=1)
            l_2_samples_moments = l_2_samples_moments[..., ~mask, :]

            #print("l1 sample momoments shape ", l_2_samples_moments.shape)

            l2_fine_moments = l_2_samples_moments[1:, :, 0]
            l2_coarse_moments = l_2_samples_moments[1:, :, 1]

            if self.n_levels == 4:
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


            #l_3_samples = estimate_obj.get_level_samples(level_id=3)
            #l_3_samples = np.squeeze(l_3_samples)

            # l1_fine_samples = l_1_samples[1:10, 0]
            # l1_coarse_samples = l_1_samples[1:10, 1]

            l1_fine_samples = l_1_samples[:, 0]
            l1_coarse_samples = l_1_samples[:, 1]

            l2_fine_samples = l_2_samples[:, 0]
            l2_coarse_samples = l_2_samples[:, 1]

            # l3_fine_samples = l_3_samples[:, 0]
            # l3_coarse_samples = l_3_samples[:, 1]

            #l3_diff = l3_fine_samples - l3_coarse_samples
            l2_diff = l2_fine_samples - l2_coarse_samples
            l1_diff = l1_fine_samples - l1_coarse_samples

            log = False
            if log:
                l2_coarse_samples = np.log10(l2_coarse_samples)
                l2_coarse_samples =l2_coarse_samples[~np.isinf(l2_coarse_samples)]

                l1_fine_samples = np.log10(l1_fine_samples)
                l1_fine_samples = l1_fine_samples[~np.isinf(l1_fine_samples)]

                l1_coarse_samples = np.log10(l1_coarse_samples)
                l1_coarse_samples = l1_coarse_samples[~np.isinf(l1_coarse_samples)]

                l_0_samples = np.log10(l_0_samples)
                l_0_samples = l_0_samples[~np.isinf(l_0_samples)]

                #l_0_samples += 0.03

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
            axes.hist(np.squeeze(l_0_samples), bins=100,  density=True, label="L0 samples", alpha=0.5)
            fig.suptitle("L1 coarse L0 samples, log: {}".format(log))
            fig.legend()
            fig.savefig("L1_coarse_L0_samples.pdf")
            plt.show()

            # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
            # axes.hist(np.squeeze(l_0_samples), bins=100, density=True)
            # fig.suptitle("L0 samples, log: {}".format(log))
            # plt.show()

            #print("l1 diff ", l1_diff)
            #print("l2 diff ", l2_diff)

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

            print("np.min([len(l_0_samples), len(l1_coarse_samples)]) ", np.min([len(l_0_samples), len(l1_coarse_samples)]))



            self.compare_means(l_0_samples, l1_coarse_samples, title="L0 fine mean and L1 coarse mean")
            self.compare_means(l1_fine_samples, l2_coarse_samples, title="L1 fine mean and L2 coarse mean")

            if self.n_levels == 4:
                self.compare_means(l2_fine_samples, l3_coarse_samples, title="L2 fine mean and L3 coarse mean")

            print("============== MOMENTS MEAN COMPARISON ===================")


            self.compare_means(l_0_samples_moments[0], l1_coarse_moments[0], title="Moments 0 L0 fine and L1 coarse ")
            self.compare_means(l_0_samples_moments[1], l1_coarse_moments[1], title="Moments 1 L0 fine and L1 coarse ")
            self.compare_means(l1_fine_moments[0], l2_coarse_moments[0], title="Moments 0 L1 fine and L2 coarse ")
            self.compare_means(l1_fine_moments[1], l2_coarse_moments[1], title="Moments 1 L1 fine and L2 coarse ")

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


            print("moments 0 L0 mean {}, L1 coarse mean: {}".format(np.mean(m0_l0_samples_moments), np.mean(m0_l1_coarse_moments)))
            print("moments 0 L0 std {}, L1 coarse std: {}".format(np.std(m0_l0_samples_moments), np.std(m0_l1_coarse_moments)))
            self.compare_means(m0_l0_samples_moments, m0_l1_coarse_moments, title="Moments 0 L0 fine and L1 coarse ")

            exit()

        sample_vec = [100, 75, 50]
        sample_vec = [4462,  213,  164]
        sample_vec = [250, 100]
        #sample_vec = [10000]
        #sample_vec = [1785,   80,   21]
        bs_n_estimated = estimate_obj.bs_target_var_n_estimated(target_var=target_var, sample_vec=sample_vec, n_subsamples=10)
        print("mean bs l vars ", estimate_obj.mean_bs_l_vars)
        print("bs n estimated ", bs_n_estimated)
        print("bs total cost ",  np.sum(n_ops * bs_n_estimated))

        ####################
        # Diagnostic plots #
        ####################
        print("estimate_obj.moments_mean_obj.l_vars ", estimate_obj.moments_mean_obj.l_vars.shape)
        print("bs means ", estimate_obj.bs_mean)
        print("mean bs mean ", estimate_obj.mean_bs_mean)
        print("mean bs var ", estimate_obj.mean_bs_var)
        dp.log_var_per_level(estimate_obj.moments_mean_obj.l_vars, moments=range(self.n_moments))
        dp.log_mean_per_level(estimate_obj.moments_mean_obj.l_means, moments=range(self.n_moments))
        dp.sample_cost_per_level(n_ops)
        dp.kurtosis_per_level(estimate_obj.moments_mean_obj.mean, estimate_obj.moments_mean_obj.l_means, moments=range(self.n_moments))

        # from mlmc.plot.plots import Distribution
        # distr_obj, result, _, _ = estimate_obj.construct_density()
        # distr_plot = Distribution(title="distributions", error_plot=None)
        # distr_plot.add_distribution(distr_obj)
        #
        # samples = estimate_obj.get_level_samples(level_id=0)[..., 0]
        # #print("samples ", samples)
        # distr_plot.add_raw_samples(np.squeeze(samples))  # add histogram
        # distr_plot.show()


if __name__ == "__main__":
    # import cProfile
    # import pstats
    #
    # pr = cProfile.Profile()
    # pr.enable()
    #
    ProcessSimple()
    #
    # pr.disable()
    # ps = pstats.Stats(pr).sort_stats('cumtime')
    # ps.print_stats()

