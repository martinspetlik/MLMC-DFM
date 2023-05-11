import os
import sys
import numpy as np
#import mlmc.tool.simple_distribution
from mlmc.sampler import Sampler
from mlmc.sample_storage_hdf import SampleStorageHDF
from mlmc.sample_storage import Memory
from mlmc.sampling_pool import OneProcessPool
from mlmc.sampling_pool_pbs import SamplingPoolPBS
from homogenization.sim_sample import DFMSim
from mlmc.moments import Legendre
from mlmc.tool.process_base import ProcessBase
from mlmc.quantity.quantity import make_root_quantity
from mlmc.quantity.quantity_estimate import estimate_mean, moments
from mlmc import estimator
import yaml


class ProcessSimple:
    # @TODO: generate more samples, with new seed

    def __init__(self):
        args = ProcessBase.get_arguments(sys.argv[1:])

        self.work_dir = os.path.abspath(args.work_dir)
        # Add samples to existing ones
        self.clean = args.clean
        # Remove HDF5 file, start from scratch
        self.debug = args.debug
        # 'Debug' mode is on - keep sample directories
        self.use_pbs = False
        # Use PBS sampling pool
        self.n_levels = 2
        self.n_moments = 25
        # Number of MLMC levels
        # step_range = [0.055, 0.0035]
        #step_range = [0.809, 0.35]
        step_range = [10.0, 4.325]
        step_range = [25.0, 10.0]
        #step_range = [10, 1] # Used for tested data of CNNs
        #step_range = [100, 10]
        #step_range = [100, 1]
        # step   - elements
        # 0.1    - 262
        # 0.08   - 478
        # 0.06   - 816
        # 0.055  - 996
        # 0.006 -  74188
        # 0.0055 - 87794
        # 0.005  - 106056
        # 0.004  - 165404
        # 0.0035 - 217208

        # step_range [simulation step at the coarsest level, simulation step at the finest level]

        # Determine level parameters at each level (In this case, simulation step at each level) are set automatically
        self.level_parameters = estimator.determine_level_parameters(self.n_levels, step_range)
        #self.level_parameters = [self.level_parameters[4]]
        #self.n_levels = 1
        print("self.level_parameters ", self.level_parameters)
        #exit()

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

    # def process(self):
    #     sample_storage = SampleStorageHDF(file_path=os.path.join(self.work_dir, "mlmc_{}.hdf5".format(self.n_levels)))
    #     sample_storage.chunk_size = 1024
    #     result_format = sample_storage.load_result_format()
    #     root_quantity = make_root_quantity(sample_storage, result_format)
    #
    #     # conductivity = root_quantity['conductivity']
    #     # time = conductivity[1]  # times: [1]
    #     # location = time['0']  # locations: ['0']
    #     # values = location[0, 0]  # result shape: (1, 1)
    #
    #     means = estimate_mean(root_quantity)
    #     # @TODO: How to estimate true_domain?
    #     true_domain = list(QuantityEstimate.estimate_domain(sample_storage, quantile=0.01))
    #
    #     n_moments = 25
    #     # moments_fn = Legendre(n_moments, true_domain)
    #     moments_fn = Monomial(n_moments, true_domain)
    #
    #     moments_quantity = moments(root_quantity, moments_fn=moments_fn, mom_at_bottom=True)
    #     moments_mean = estimate_mean(moments_quantity)
    #
    #     conductivity_mean = moments_mean['conductivity']
    #     time_mean = conductivity_mean[1]  # times: [1]
    #     location_mean = time_mean['0']  # locations: ['0']
    #     values_mean = location_mean[0, 0]  # result shape: (1, 1)
    #     value_mean = values_mean[0]
    #     assert value_mean() == 1
    #
    #     true_domain = [-10, 10]  # keep all values on the original domain
    #     central_moments_fn = Monomial(n_moments, true_domain, ref_domain=true_domain, mean=means())
    #     central_moments_quantity = moments(root_quantity, moments_fn=central_moments_fn, mom_at_bottom=True)
    #     central_moments_mean = estimate_mean(central_moments_quantity)
    #
    #     print("central moments mean ", central_moments_mean())
    #     print("moments mean ", moments_mean())

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

        # Create sampler (mlmc.Sampler instance) - crucial class which actually schedule samples
        sampler = self.setup_config(clean=self.clean)

        # Schedule samples
        # self.generate_jobs(sampler, n_samples=[5], renew=renew)

        # self.generate_jobs(sampler, n_samples=None, renew=renew, target_var=1e-3)
        # self.generate_jobs(sampler, n_samples=[500, 500], renew=renew, target_var=1e-5)
        if recollect:
            raise NotImplementedError("Not supported in released version")
        else:
            self.generate_jobs(sampler, n_samples=[1, 1], renew=renew)
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

        with open(os.path.join(self.work_dir, "sim_config.yaml"), "r") as f:
            sim_config_dict = yaml.load(f, Loader=yaml.FullLoader)
            print("sim_config_dict ", sim_config_dict)

        sim_config_dict['fields_params'] = dict(model='exp', corr_length=0.1, sigma=1, mode_no=10000)
        sim_config_dict['field_template'] = "!FieldFE {mesh_data_file: \"$INPUT_DIR$/%s\", field_name: %s}"
        sim_config_dict['env'] = dict(flow123d=self.flow123d, gmsh=self.gmsh, gmsh_version=1)  # The Environment.
        sim_config_dict['work_dir'] = self.work_dir
        sim_config_dict['yaml_file'] = os.path.join(self.work_dir, '01_conductivity.yaml')
        sim_config_dict['yaml_file_homogenization'] = os.path.join(self.work_dir, 'flow_templ.yaml')
        sim_config_dict['yaml_file_homogenization_vtk'] = os.path.join(self.work_dir, 'flow_templ_vtk.yaml')

        # simulation_config = {
        #     'work_dir': self.work_dir,
        #     'env': dict(flow123d=self.flow123d, gmsh=self.gmsh, gmsh_version=1),  # The Environment.
        #     'yaml_file': os.path.join(self.work_dir, '01_conductivity.yaml'),
        #     'geo_file': os.path.join(self.work_dir, 'square_1x1.geo'),
        #     'fields_params': dict(model='exp', corr_length=0.1, sigma=1, mode_no=10000),
        #     'field_template': "!FieldFE {mesh_data_file: \"$INPUT_DIR$/%s\", field_name: %s}"
        # }

        # Create simulation factory
        simulation_factory = DFMSim(config=sim_config_dict, clean=clean)

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
            self.flow123d = ["flow123d/flow123d-gnu:3.1.0", "flow123d"]  # "/home/martin/flow123d/bin/fterm --no-term rel run"
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
            python='python3.8',
            env_setting=['cd $MLMC_WORKDIR',
                         'module load python/3.8.0-gcc',
                         'source env/bin/activate',
                         'module use /storage/praha1/home/jan-hybs/modules',
                         'module load flow123d',
                         'module unload python-3.6.2-gcc',
                         'module unload python36-modules-gcc'],
            scratch_dir=None
        )

        sampling_pool.pbs_common_setting(flow_3=True, **pbs_config)

        return sampling_pool

    def generate_jobs(self, sampler, n_samples=None, renew=False, target_var=None):
        """
        Generate level samples
        :param n_samples: None or list, number of samples for each level
        :param renew: rerun failed samples with same random seed (= same sample id)
        :return: None
        """
        if renew:
            sampler.ask_sampling_pool_for_samples()
            sampler.renew_failed_samples()
            sampler.ask_sampling_pool_for_samples(sleep=self.sample_sleep, timeout=self.sample_timeout)
        else:
            if n_samples is not None:
                sampler.set_initial_n_samples(n_samples)
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

                # Loop until number of estimated samples is greater than the number of scheduled samples
                while not sampler.process_adding_samples(n_estimated, self.sample_sleep, self.adding_samples_coef,
                                                         timeout=self.sample_timeout):
                    # New estimation according to already finished samples
                    variances, n_ops = estimate_obj.estimate_diff_vars_regression(sampler._n_scheduled_samples)
                    n_estimated = estimator.estimate_n_samples_for_target_variance(target_var, variances, n_ops,
                                                                                   n_levels=sampler.n_levels)

    def set_moments(self, quantity, sample_storage, n_moments=5):
        true_domain = estimator.Estimate.estimate_domain(quantity, sample_storage, quantile=0.01)
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


if __name__ == "__main__":
    pr = ProcessSimple()

