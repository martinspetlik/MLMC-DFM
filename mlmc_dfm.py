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
from homogenization.sim_sample import DFMSim
from mlmc.moments import Legendre
from mlmc.tool.process_base import ProcessBase
from mlmc.quantity.quantity import make_root_quantity
from mlmc.quantity.quantity_estimate import estimate_mean, moments
from mlmc import estimator
import yaml
import ruamel.yaml


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
        self.n_levels = 2
        self.n_moments = 5
        # Number of MLMC levels

        #step_range = [0.055, 0.0035]
        #step_range = [0.809, 0.35] # Used for tested data of CNNs
        #step_range = [0.809, 0.35]
        step_range = [10.0, 4.326]
        #step_range = [10, 1.871]
        #step_range = [4.325, 1.871]
        #step_range = [100, 10]
        # step_range = [0.1, 0.055]
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
            #self.generate_jobs(sampler, n_samples=[5], renew=renew, target_var=1e-5)
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
        sim_config_dict['scratch_dir'] = self.scratch_dir
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
                if self.use_pbs or self.generate_samples_per_level:
                    for level_id in reversed(range(self.n_levels)):
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

                        level_instance_obj = sampler._level_sim_objects[level_id]
                        if level_id > 0:
                            cond_tn_pop_file = os.path.join(level_instance_obj.config_dict["coarse"]["common_files_dir"],
                                                            DFMSim.COND_TN_POP_FILE)
                            pred_cond_tn_pop_file = os.path.join(level_instance_obj.config_dict["coarse"]["common_files_dir"],
                                                            DFMSim.PRED_COND_TN_POP_FILE)

                            if cond_tn_pop_file is not None:
                                sample_cond_tns = []
                                sample_pred_cond_tns = []
                                for s in range(int(sampler.n_finished_samples[level_id])):
                                    sample_dir_name = "L{:02d}_S{:07d}".format(level_id, s)
                                    sample_dir = os.path.join(os.path.join(self.work_dir, "output"), sample_dir_name)

                                    if not os.path.exists(sample_dir):
                                        continue

                                    sample_cond_tns_file = os.path.join(sample_dir, DFMSim.COND_TN_FILE)
                                    sample_pred_cond_tns_file = os.path.join(sample_dir, DFMSim.PRED_COND_TN_FILE)

                                    if os.path.exists(sample_cond_tns_file):
                                        with open(sample_cond_tns_file, "r") as f:
                                            cond_tns = ruamel.yaml.load(f)
                                        sample_cond_tns.extend(list(cond_tns.values()))

                                    if os.path.exists(sample_pred_cond_tns_file):
                                        with open(sample_pred_cond_tns_file, "r") as f:
                                            pred_cond_tns = ruamel.yaml.load(f)
                                        sample_pred_cond_tns.extend(list(pred_cond_tns.values()))

                                np.save(cond_tn_pop_file, sample_cond_tns)
                                np.save(pred_cond_tn_pop_file, sample_pred_cond_tns)

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

    def process(self):
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

        target_var = 1e-5

        root_quantity = make_root_quantity(storage=sample_storage,
                                           q_specs=sample_storage.load_result_format())

        cond_tn_quantity = root_quantity["cond_tn"]
        time_mean = cond_tn_quantity[1]  # times: [1]
        location_mean = time_mean['0']  # locations: ['0']
        k_xx = location_mean[0]
        q_value = k_xx #cond_tn_quantity#k_xx
        #q_value = np.array([location_mean[0], location_mean[1], location_mean[2]])

        # @TODO: How to estimate true_domain?
        quantile = 0.001
        true_domain = mlmc.estimator.Estimate.estimate_domain(q_value, sample_storage, quantile=quantile)
        print("true domain ", true_domain)
        moments_fn = Legendre(self.n_moments, true_domain)

        n_ops = np.array(sample_storage.get_n_ops())
        print("n ops ", n_ops)
        # print("n ops ", n_ops[:, 0] / n_ops[:, 1])

        print("sample storage n collected ", sample_storage.get_n_collected())

        estimate_obj = mlmc.estimator.Estimate(quantity=q_value, sample_storage=sample_storage, moments_fn=moments_fn)
        means, vars = estimate_obj.estimate_moments(moments_fn)

        print('means ', means)
        print("vars ", vars)

        moments_fn = self.set_moments(q_value, sample_storage, n_moments=self.n_moments)
        estimate_obj = estimator.Estimate(q_value, sample_storage=sample_storage,
                                          moments_fn=moments_fn)

        # New estimation according to already finished samples
        variances, n_ops = estimate_obj.estimate_diff_vars_regression(sample_storage.get_n_collected())
        n_estimated = estimator.estimate_n_samples_for_target_variance(target_var, variances, n_ops,
                                                                       n_levels=self.n_levels)

        print("variances ", variances)
        print("n ops ", n_ops)
        print("n estimated ", n_estimated)

        #n_ops[1] = 100
        print("total cost ", np.sum(n_ops * n_estimated)) #np.array(sample_storage.get_n_collected())))

        # moments_quantity = moments(root_quantity, moments_fn=moments_fn, mom_at_bottom=True)
        # moments_mean = estimate_mean(moments_quantity)
        # conductivity_mean = moments_mean['cond_tn']
        # time_mean = conductivity_mean[1]  # times: [1]
        # location_mean = time_mean['0']  # locations: ['0']
        # values_mean = location_mean[0]  # result shape: (1,)
        # value_mean = values_mean[0]
        # assert value_mean.mean == 1

        l_0_samples = estimate_obj.get_level_samples(level_id=0)

        if self.n_levels == 1:
            print("l0 var", np.var(l_0_samples))

            print("L0 mean {}".format(np.mean(l_0_samples)))

        else:
            l_1_samples = estimate_obj.get_level_samples(level_id=1)

            l_1_samples = np.squeeze(l_1_samples)
            #print("squeezed l1 samples ", l_1_samples)
            #print("l_1_samples[:, 0] ", l_1_samples[:, 0])
            #print("l_1_samples[:, 1] ", l_1_samples[:, 1])

            # l1_fine_samples = l_1_samples[1:10, 0]
            # l1_coarse_samples = l_1_samples[1:10, 1]

            l1_fine_samples = l_1_samples[:, 0]
            l1_coarse_samples = l_1_samples[:, 1]

            l1_diff = l1_fine_samples - l1_coarse_samples

            #print("l1 diff ", l1_diff)
            print("l1 diff var", np.var(l1_diff))


            print("l0 var", np.var(l_0_samples))


            print("L0 mean {}, L1 coarse mean: {}".format(np.mean(l_0_samples), np.mean(l1_coarse_samples)))


        from mlmc.plot.plots import Distribution
        distr_obj, result, _, _ = estimate_obj.construct_density()
        distr_plot = Distribution(title="distributions", error_plot=None)
        distr_plot.add_distribution(distr_obj)

        samples = estimate_obj.get_level_samples(level_id=0)[..., 0]
        #print("samples ", samples)
        distr_plot.add_raw_samples(np.squeeze(samples))  # add histogram
        distr_plot.show()


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

