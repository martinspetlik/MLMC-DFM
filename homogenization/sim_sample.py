import os
import os.path
import subprocess
import numpy as np
import shutil
import time
import copy
import ruamel.yaml as yaml
from typing import List
import gstools
from mlmc.level_simulation import LevelSimulation
import gmsh_io
from mlmc.sim.simulation import Simulation
from mlmc.quantity.quantity_spec import QuantitySpec
#from mlmc.random import correlated_field as cf
import homogenization.fracture as fracture
from homogenization.both_sample import FlowProblem, BothSample


def create_corr_field(model='gauss', corr_length=0.125, dim=2, log=True, sigma=1, mode_no=1000):
    """
    Create random fields
    :return:
    """
    #@TODO: constant conductivity SRF
    # if model == 'fourier':
    #     return cf.Fields([
    #         cf.Field('conductivity', cf.FourierSpatialCorrelatedField('gauss', dim=dim,
    #                                                                   corr_length=corr_length,
    #                                                                   log=log, sigma=sigma)),
    #     ])
    #
    # elif model == 'svd':
    #     conductivity = dict(
    #         mu=0.0,
    #         sigma=sigma,
    #         corr_exp='exp',
    #         dim=dim,
    #         corr_length=corr_length,
    #         log=log
    #     )
    #     return cf.Fields([cf.Field("conductivity", cf.SpatialCorrelatedField(**conductivity))])
    #
    # elif model == 'exp':
    #     model = gstools.Exponential(dim=dim, len_scale=corr_length)
    # elif model == 'TPLgauss':
    #     model = gstools.TPLGaussian(dim=dim, len_scale=corr_length)
    # elif model == 'TPLexp':
    #     model = gstools.TPLExponential(dim=dim, len_scale=corr_length)
    # elif model == 'TPLStable':
    #     model = gstools.TPLStable(dim=dim, len_scale=corr_length)
    # else:
    #     model = gstools.Gaussian(dim=dim, len_scale=corr_length)
    #
    # return cf.Fields([
    #     cf.Field('conductivity', cf.GSToolsSpatialCorrelatedField(model, log=log, sigma=sigma, mode_no=mode_no)),
    # ])

def substitute_placeholders(file_in, file_out, params):
    """
    Substitute for placeholders of format '<name>' from the dict 'params'.
    :param file_in: Template file.
    :param file_out: Values substituted.
    :param params: { 'name': value, ...}
    """
    used_params = []
    with open(file_in, 'r') as src:
        text = src.read()
    for name, value in params.items():
        placeholder = '<%s>' % name
        n_repl = text.count(placeholder)
        if n_repl > 0:
            used_params.append(name)
            text = text.replace(placeholder, str(value))
    with open(file_out, 'w') as dst:
        dst.write(text)
    return used_params


def force_mkdir(path, force=False):
    """
    Make directory 'path' with all parents,
    remove the leaf dir recursively if it already exists.
    :param path: path to directory
    :param force: if dir already exists then remove it and create new one
    :return: None
    """
    if force:
        if os.path.isdir(path):
            shutil.rmtree(path)
    os.makedirs(path, mode=0o775, exist_ok=True)



def in_file(base):
    return "flow_{}.yaml".format(base)


def mesh_file(base):
    return "mesh_{}.msh".format(base)


def fields_file(base):
    return "fields_{}.msh".format(base)


class DFMSim(Simulation):
    # placeholders in YAML
    total_sim_id = 0
    MESH_FILE_VAR = 'mesh_file'
    # Timestep placeholder given as O(h), h = mesh step
    TIMESTEP_H1_VAR = 'timestep_h1'
    # Timestep placeholder given as O(h^2), h = mesh step
    TIMESTEP_H2_VAR = 'timestep_h2'

    # files
    GEO_FILE = 'mesh.geo'
    MESH_FILE = 'mesh.msh'
    YAML_TEMPLATE_H = "flow_input_h.yaml.tmpl"
    YAML_TEMPLATE = 'flow_input.yaml.tmpl'
    YAML_FILE = 'flow_input.yaml'
    FIELDS_FILE = "flow_fields.msh"
    COND_TN_POP_FILE = 'cond_tn_pop.npy'
    COND_TN_FILE = "cond_tensors.yaml"
    COMMON_FILES = "l_step_{}_common_files"

    """
    Gather data for single flow call (coarse/fine)

    Usage:
    mlmc.sampler.Sampler uses instance of FlowSim, it calls once level_instance() for each level step (The level_instance() method
     is called as many times as the number of levels), it takes place in main process

    mlmc.tool.pbs_job.PbsJob uses static methods in FlowSim, it calls calculate(). That's where the calculation actually runs,
    it takes place in PBS process
       It also extracts results and passes them back to PbsJob, which handles the rest 

    """

    def __init__(self, config=None, clean=None):
        """
        Simple simulation using flow123d
        :param config: configuration of the simulation, processed keys:
            env - Environment object.
            fields - FieldSet object
            yaml_file: Template for main input file. Placeholders:
                <mesh_file> - replaced by generated mesh
                <FIELD> - for FIELD be name of any of `fields`, replaced by the FieldElementwise field with generated
                 field input file and the field name for the component.
            geo_file: Path to the geometry file.
        :param clean: bool, if True remove existing simulation files - mesh files, ...
        """
        # print("config ", config)
        # self.need_workspace = True
        # # This simulation requires workspace
        self.env = config['env']
        # # Environment variables, flow123d, gmsh, ...
        self._fields_params = config['fields_params']
        # self._fields = create_corr_field(**config['fields_params'])
        self._fields_used_params = None
        # # Random fields instance
        self.time_factor = config.get('time_factor', 1.0)
        # # It is used for minimal element from mesh determination (see level_instance method)
        #
        self.base_yaml_file_homogenization = config['yaml_file_homogenization']
        self.base_yaml_file = config['yaml_file']
        # self.base_geo_file = config['geo_file']
        # self.field_template = config.get('field_template',
        #                                  "!FieldFE {mesh_data_file: $INPUT_DIR$/%s, field_name: %s}")
        self.work_dir = config['work_dir']
        self.clean = clean
        self.config_dict = config
        super(Simulation, self).__init__()

    def level_instance(self, fine_level_params: List[float], coarse_level_params: List[float]) -> LevelSimulation:
        """
        Called from mlmc.Sampler, it creates single instance of LevelSimulation (mlmc.)
        :param fine_level_params: in this version, it is just fine simulation step
        :param coarse_level_params: in this version, it is just coarse simulation step
        :return: mlmc.LevelSimulation object, this object is serialized in SamplingPoolPbs and deserialized in PbsJob,
         so it allows pass simulation data from main process to PBS process
        """
        fine_step = fine_level_params[0]
        coarse_step = coarse_level_params[0]

        # TODO: determine minimal element from mesh
        self.time_step_h1 = self.time_factor * fine_step
        self.time_step_h2 = self.time_factor * fine_step * fine_step

        # Set fine simulation common files directory
        # Files in the directory are used by each simulation at that level
        common_files_dir = os.path.join(self.work_dir, DFMSim.COMMON_FILES.format(fine_step))
        force_mkdir(common_files_dir, force=self.clean)

        self.mesh_file = os.path.join(common_files_dir, DFMSim.MESH_FILE)

        if self.clean:
            # Prepare mesh
            #geo_file = os.path.join(common_files_dir, DFMSim.GEO_FILE)
            #shutil.copyfile(self.base_geo_file, geo_file)
            #self._make_mesh(geo_file, self.mesh_file, fine_step)  # Common computational mesh for all samples.

            # Prepare main input YAML
            yaml_template_h = os.path.join(common_files_dir, DFMSim.YAML_TEMPLATE_H)
            shutil.copyfile(self.base_yaml_file_homogenization, yaml_template_h)

            yaml_template = os.path.join(common_files_dir, DFMSim.YAML_TEMPLATE)
            shutil.copyfile(self.base_yaml_file, yaml_template)

            yaml_file = os.path.join(common_files_dir, DFMSim.YAML_FILE)
            self._substitute_yaml(yaml_template, yaml_file)
        #
        # # Mesh is extracted because we need number of mesh points to determine task_size parameter (see return value)
        # fine_mesh_data = self.extract_mesh(self.mesh_file)

        #@TODO: determine task size
        # Set coarse simulation common files directory
        # Files in the directory are used by each simulation at that level
        coarse_sim_common_files_dir = None
        if coarse_step != 0:
            coarse_sim_common_files_dir = os.path.join(self.work_dir, DFMSim.COMMON_FILES.format(coarse_step))

        # Simulation config
        # Configuration is used in mlmc.tool.pbs_job.PbsJob instance which is run from PBS process
        # It is part of LevelSimulation which is serialized and then deserialized in mlmc.tool.pbs_job.PbsJob
        config = dict()
        config["fine"] = {}
        config["coarse"] = {}
        config["fine"]["step"] = fine_step
        config["coarse"]["step"] = coarse_step
        config["sim_config"] = self.config_dict
        config["fine"]["common_files_dir"] = common_files_dir
        config["coarse"]["common_files_dir"] = coarse_sim_common_files_dir

        #config["fields_used_params"] = self._fields_used_params  # Params for Fields instance, which is created in PbsJob

        print("self.env ", type(self.env))
        config["gmsh"] = self.env['gmsh']
        config["flow123d"] = self.env['flow123d']
        config['fields_params'] = self._fields_params

        # Auxiliary parameter which I use to determine task_size (should be from 0 to 1, if task_size is above 1 then pbs job is scheduled)
        job_weight = 17000000  # 4000000 - 20 min, 2000000 - cca 10 min

        return LevelSimulation(config_dict=config,
                               task_size=1/fine_step,  #len(fine_mesh_data['points']) / job_weight,
                               calculate=DFMSim.calculate,
                               # method which carries out the calculation, will be called from PBS processs
                               need_sample_workspace=True # If True, a sample directory is created
                               )

    @staticmethod
    def homogenization(config):
        #print("config ", config)
        sample_dir = os.getcwd()
        #print("homogenization method")
        #print("os.getcws() ", os.getcwd())
        os.mkdir("homogenization")
        os.chdir("homogenization")

        h_dir = os.getcwd()

        sim_config = config["sim_config"]
        #print("sim config ", sim_config)

        domain_box = sim_config["geometry"]["domain_box"]
        subdomain_box = sim_config["geometry"]["subdomain_box"]
        subdomain_overlap = np.array([0, 0])  # np.array([50, 50])

        # print("sample dict ", sim_config)
        # print(sim_config["seed"])

        # work_dir = "seed_{}".format(sample_dict["seed"])

        # work_dir = os.path.join(
        #     "/home/martin/Documents/Endorse/ms-homogenization/seed_26339/aperture_10_4/test_density",
        #     "n_10_s_100_100_step_5_2")

        #work_dir = "/home/martin/Documents/Endorse/ms-homogenization/test_summary"

        work_dir = sim_config["work_dir"]

        #print("work dir ", work_dir)
        lx, ly = domain_box

        bottom_left_corner = [-lx / 2, -ly / 2]
        bottom_right_corner = [+lx / 2, -ly / 2]
        top_right_corner = [+lx / 2, +ly / 2]
        top_left_corner = [-lx / 2, +ly / 2]
        #               bottom-left corner  bottom-right corner  top-right corner    top-left corner
        complete_polygon = [bottom_left_corner, bottom_right_corner, top_right_corner, top_left_corner]

        #plt.scatter(*zip(*complete_polygon))
        n_subdomains = sim_config["geometry"].get("n_subdomains", 4)
        domain_box = sim_config["geometry"]["domain_box"]
        subdomain_box = sim_config["geometry"]["subdomain_box"]
        lx, ly = domain_box

        n_subdomains = int(np.floor(np.sqrt(n_subdomains)))

        cond_tensors = {}
        percentage_sym_tn_diff = []
        time_measurements = []

        k = 0
        #
        # if n_subdomains == 1:
        #     DFMSim.run_single_subdomain()

        for i in range(n_subdomains):
            #print("subdomain box ", subdomain_box)
            if "outer_polygon" not in sim_config["geometry"]:
                center_x = subdomain_box[0] / 2 + (lx - subdomain_box[0]) / (n_subdomains - 1) * i - lx / 2
            for j in range(n_subdomains):
                start_time = time.time()
                k += 1

                subdir_name = "i_{}_j_{}_k_{}".format(i, j, k)
                #print("subdir_name ", subdir_name)
                os.mkdir(subdir_name)
                os.chdir(subdir_name)
                # if k != 24:pop
                #     continue
                # if k > 1:
                #     continue
                # if k not in [1]:
                #     continue
                # if k < 88:
                #     continue
                if not "outer_polygon" in sim_config["geometry"]:
                    center_y = subdomain_box[1] / 2 + (lx - subdomain_box[1]) / (n_subdomains - 1) * j - lx / 2

                    bl_corner = [center_x - subdomain_box[0] / 2, center_y - subdomain_box[1] / 2]
                    br_corner = [center_x + subdomain_box[0] / 2, center_y - subdomain_box[1] / 2]
                    tl_corner = [center_x - subdomain_box[0] / 2, center_y + subdomain_box[1] / 2]
                    tr_corner = [center_x + subdomain_box[0] / 2, center_y + subdomain_box[1] / 2]

                    #print("center x: {}, y: {}".format(center_x, center_y))

                    outer_polygon = [copy.deepcopy(bl_corner), copy.deepcopy(br_corner), copy.deepcopy(tr_corner),
                                     copy.deepcopy(tl_corner)]

                    sim_config["geometry"]["outer_polygon"] = outer_polygon
                #print("work_dir ", work_dir)

                sim_config["work_dir"] = work_dir
                #config["homogenization"] = True
                fractures = DFMSim.generate_fractures(config)

                # fine problem
                fine_flow = FlowProblem.make_fine((config["fine"]["step"], config["sim_config"]["geometry"]["fr_max_size"]), fractures, config)
                fine_flow.fr_range = [config["fine"]["step"], config["coarse"]["step"]]

                if n_subdomains == 1:
                    mesh_file = "/home/martin/Desktop/mesh_fine.msh"
                    shutil.copy(mesh_file, os.getcwd())
                    elids_same_value = {18: 17, 20: 19, 22: 21, 24: 23, 26: 25, 28: 27, 30: 29, 32: 31, 34: 33}

                    fine_flow.make_mesh(mesh_file)
                    fine_flow.make_fields(elids_same_value=elids_same_value)

                    #fine_flow.make_mesh()
                    #fine_flow.make_fields()
                    done = []
                    #exit()
                    # fine_flow.run() # @TODO replace fine_flow.run by DFMSim._run_sample()
                    #print("run samples ")
                    status, p_loads, outer_reg_names = DFMSim._run_homogenization_sample(fine_flow, config)

                    done.append(fine_flow)
                    cond_tn, diff = fine_flow.effective_tensor_from_bulk(p_loads, outer_reg_names, fine_flow.basename, elids_same_value)
                    #print("cond_tn ", cond_tn)
                    #exit()
                    DFMSim.make_summary(done)
                    percentage_sym_tn_diff.append(diff)

                else:
                    fine_flow.make_mesh()
                    fine_flow.make_fields()
                    done = []
                    # exit()
                    # fine_flow.run() # @TODO replace fine_flow.run by DFMSim._run_sample()
                    #print("run samples ")
                    status, p_loads, outer_reg_names = DFMSim._run_homogenization_sample(fine_flow, config)

                    done.append(fine_flow)
                    cond_tn, diff = fine_flow.effective_tensor_from_bulk(p_loads, outer_reg_names, fine_flow.basename)
                    DFMSim.make_summary(done)
                    percentage_sym_tn_diff.append(diff)

                #print("cond_tn ", cond_tn)
                # @TODO: save cond tn and center to npz file

                if n_subdomains > 1:
                    cond_tensors[(center_x, center_y)] = cond_tn

                    cond_tn_pop_file = os.path.join(config["coarse"]["common_files_dir"], DFMSim.COND_TN_POP_FILE)
                    np.save(cond_tn_pop_file, cond_tn[0])

                dir_name = os.path.join(work_dir, subdir_name)
                config["dir_name"] = dir_name

                #print("dir name ", dir_name)

                try:
                    shutil.move("fine", dir_name)
                    if os.path.exists(os.path.join(dir_name, "fields_fine.msh")):
                        os.remove(os.path.join(dir_name, "fields_fine.msh"))
                    shutil.move("fields_fine.msh", dir_name)
                    shutil.move("summary.yaml", dir_name)
                    shutil.move("mesh_fine.msh", dir_name)
                    shutil.move("mesh_fine.brep", dir_name)
                    shutil.move("mesh_fine.tmp.geo", dir_name)
                    shutil.move("mesh_fine.tmp.msh", dir_name)
                    shutil.rmtree("fine")

                except:
                    pass

                os.chdir(h_dir)

                time_measurements.append(time.time() - start_time)

        # print("np.mean(percentage_sym_tn_diff) ", np.mean(percentage_sym_tn_diff))
        # print("time_measurements ", time_measurements)

        os.chdir(sample_dir)

        return cond_tensors


    @staticmethod
    def calculate(config, seed):
        """
        Method that actually run the calculation, it's called from mlmc.tool.pbs_job.PbsJob.calculate_samples()
        Calculate fine and coarse sample and also extract their results
        :param config: dictionary containing simulation configuration, LevelSimulation.config_dict (set in level_instance)
        :param seed: random seed, int
        :return: List[fine result, coarse result], both flatten arrays (see mlmc.sim.synth_simulation.calculate())
        """
        #@TODO: check sample dir creation
        fractures = DFMSim.generate_fractures(config)

        coarse_step = config["coarse"]["step"]
        fine_step = config["fine"]["step"]

        #print("fine_step", config["fine"]["step"])
        #print("coarse_step", config["coarse"]["step"])

        times = {}
        fine_res = 0

        # try:
        #shutil.rmtree("fine")
        if os.path.exists("fields_fine.msh"):
            os.remove("fields_fine.msh")
        if os.path.exists("fields_coarse.msh"):
            os.remove("fields_coarse.msh")
        if os.path.exists("summary.yaml"):
            os.remove("summary.yaml")
        if os.path.exists("mesh_fine.msh"):
            os.remove("mesh_fine.msh")
        if os.path.exists("mesh_fine.brep"):
            os.remove("mesh_fine.brep")
        if os.path.exists("mesh_fine.tmp.geo"):
            os.remove("mesh_fine.tmp.geo")
        if os.path.exists("mesh_fine.tmp.msh"):
            os.remove("mesh_fine.tmp.msh")
        if os.path.exists("mesh_coarse.msh"):
            os.remove("mesh_coarse.msh")
        if os.path.exists("mesh_coarse.brep"):
            os.remove("mesh_coarse.brep")
        if os.path.exists("mesh_coarse.tmp.geo"):
            os.remove("mesh_coarse.tmp.geo")
        if os.path.exists("mesh_coarse.tmp.msh"):
            os.remove("mesh_coarse.tmp.msh")
        if os.path.exists("water_balance.txt"):
            os.remove("water_balance.txt")
        if os.path.exists("water_balance.yaml"):
            os.remove("water_balance.yaml")
        if os.path.exists("flow_fields.msh"):
            os.remove("flow_fields.msh")
        # if os.path.exists("fields_fine.msh")
        #shutil.rmtree("fine")
        # except:
        #     pass

        # if coarse_step == 0:
        #     print("config fine", config["fine"])
        #     exit()

        ####################
        ### fine problem ###
        ####################
        fine_flow = FlowProblem.make_fine((config["fine"]["step"], config["sim_config"]["geometry"]["fr_max_size"]), fractures, config)
        fine_flow.fr_range = [config["fine"]["step"], fine_flow.fr_range[1]]
        #fine_flow.fr_range = [25, fine_flow.fr_range[1]]

        #print("fine_flow.fr_range ", fine_flow.fr_range)

        make_mesh_start = time.time()

        if config["sim_config"]["geometry"]["n_subdomains"] == 1:
            mesh_file = "/home/martin/Desktop/mesh_fine.msh"
            shutil.copy(mesh_file, os.getcwd())
            elids_same_value = {18: 17, 20: 19, 22: 21, 24: 23, 26: 25, 28: 27, 30: 29, 32: 31, 34: 33}
            fine_flow.make_mesh(mesh_file)
            times['make_mesh'] = time.time() - make_mesh_start
            make_fields_start = time.time()
            fine_flow.make_fields(elids_same_value=elids_same_value)
            times['make_fields'] = time.time() - make_fields_start
        else:
            fine_flow.make_mesh()
            times['make_mesh'] = time.time() - make_mesh_start
            make_fields_start = time.time()
            fine_flow.make_fields()
            times['make_fields'] = time.time() - make_fields_start

        #fine_flow.run() # @TODO replace fine_flow.run by DFMSim._run_sample()
        #print("run samples ")
        fine_res, status = DFMSim._run_sample(fine_flow, config)
        #done = []
        #done.append(fine_flow)
        #fine_flow.effective_tensor_from_bulk(p_loads, outer_reg_names, fine_flow.basename)
        #print("done ", done)
        #DFMSim.make_summary(done)
        #print("fine res ", fine_res)

        coarse_res = 0

        #try:
            #shutil.rmtree("fine")
        # if os.path.exists(os.path.join(dir_name, "fields_fine.msh")):
        #     os.remove(os.path.join(dir_name, "fields_fine.msh"))
        # os.remove("fields_fine.msh")
        # os.remove("summary.yaml")
        # os.remove("mesh_fine.msh")
        # os.remove("mesh_fine.brep")
        # os.remove("mesh_fine.tmp.geo")
        # os.remove("mesh_fine.tmp.msh")
        #shutil.rmtree("fine")
        # except:
        #     pass

        if coarse_step > 0:
            # @TODO: create homogenization mesh and run homogenization samples
            cond_tensors = DFMSim.homogenization(copy.deepcopy(config))
            DFMSim._save_tensors(cond_tensors)

            cond_tn_pop = os.path.join(DFMSim.COMMON_FILES.format(fine_step), DFMSim.COND_TN_POP_FILE)
            # print("cond tn pop ", cond_tn_pop)
            if os.path.exists(cond_tn_pop):
                config["fine"]["cond_tn_pop_file"] = cond_tn_pop

            config["cond_tns_yaml_file"] = os.path.abspath(DFMSim.COND_TN_FILE)

            ######################
            ### coarse problem ###
            ######################
            #coarse_ref = FlowProblem.make_microscale((config["fine"]["step"], config["coarse"]["step"]), fractures, fine_flow, config)
            # coarse_ref.pressure_loads = p_loads
            # coarse_ref.reg_to_group = fine_flow.reg_to_group
            # coarse_ref.regions = fine_flow.regions
            print("coarse problem ")

            coarse_flow = FlowProblem.make_coarse((config["coarse"]["step"], config["sim_config"]["geometry"]["fr_max_size"]), fractures, config)
            coarse_flow.fr_range = [config["coarse"]["step"], coarse_flow.fr_range[1]]
            coarse_flow.make_mesh()
            coarse_flow.make_fields()
            done = []
            # fine_flow.run() # @TODO replace fine_flow.run by DFMSim._run_sample()
            #print("run samples ")
            coarse_res, status = DFMSim._run_sample(coarse_flow, config)

            # done.append(coarse_flow)
            # coarse_flow.effective_tensor_from_bulk(p_loads, outer_reg_names, coarse_flow.basename)
            # DFMSim.make_summary(done)

        return fine_res, coarse_res, times

    @staticmethod
    def _save_tensors(cond_tensors):
        with open(DFMSim.COND_TN_FILE, "w") as f:
            yaml.dump(cond_tensors, f)

        # import time
        # time.sleep(2)
        # with open(DFMSim.COND_TN_FILE, "r") as f:
        #     cond_tns = yaml.load(f)
        #
        # print("cond_tns ", cond_tns)
        # exit()

    # @staticmethod
    # def make_fields(fields, fine_mesh_data, coarse_mesh_data):
    #     """
    #     Create random fields that are used by both coarse and fine simulation
    #     :param fields: correlated_field.Fields instance
    #     :param fine_mesh_data: Dict contains data extracted from fine mesh file (points, point_region_ids, region_map)
    #     :param coarse_mesh_data: Dict contains data extracted from coarse mesh file (points, point_region_ids, region_map)
    #     :return: correlated_field.Fields
    #     """
    #     # One level MC has no coarse_mesh_data
    #     if coarse_mesh_data is None:
    #         fields.set_points(fine_mesh_data['points'], fine_mesh_data['point_region_ids'],
    #                           fine_mesh_data['region_map'])
    #     else:
    #         coarse_centers = coarse_mesh_data['points']
    #         both_centers = np.concatenate((fine_mesh_data['points'], coarse_centers), axis=0)
    #         both_regions_ids = np.concatenate(
    #             (fine_mesh_data['point_region_ids'], coarse_mesh_data['point_region_ids']))
    #         assert fine_mesh_data['region_map'] == coarse_mesh_data['region_map']
    #         fields.set_points(both_centers, both_regions_ids, fine_mesh_data['region_map'])
    #
    #     return fields


    @staticmethod
    def _run_homogenization_sample(flow_problem, config):
        """
               Create random fields file, call Flow123d and extract results
               :param fields_file: Path to file with random fields
               :param ele_ids: Element IDs in computational mesh
               :param fine_input_sample: fields: {'field_name' : values_array, ..}
               :param flow123d: Flow123d command
               :param common_files_dir: Directory with simulations common files (flow_input.yaml, )
               :return: simulation result, ndarray
               """

        # if homogenization:
        outer_reg_names = []
        for reg in flow_problem.side_regions:
            outer_reg_names.append(reg.name)
            outer_reg_names.append(reg.sub_reg.name)

        base = flow_problem.basename
        outer_regions_list = outer_reg_names
        # flow_args = config["flow123d"]
        n_steps = config["sim_config"]["n_pressure_loads"]
        t = np.pi * np.arange(0, n_steps) / n_steps
        p_loads = np.array([np.cos(t), np.sin(t)]).T

        in_f = in_file(flow_problem.basename)
        out_dir = flow_problem.basename
        # n_loads = len(self.p_loads)
        # flow_in = "flow_{}.yaml".format(self.base)
        params = dict(
            mesh_file=mesh_file(flow_problem.basename),
            fields_file=fields_file(flow_problem.basename),
            outer_regions=str(outer_regions_list),
            n_steps=len(p_loads)
        )

        # print("in_f ", in_f)
        # print("outerregions ", outer_regions_list)
        # print("p_loads ", p_loads)
        #
        # print("os.getcwd() ", os.getcwd())

        out_dir = os.getcwd()

        common_files_dir = config["fine"]["common_files_dir"]
        substitute_placeholders(os.path.join(common_files_dir, DFMSim.YAML_TEMPLATE_H), in_f, params)
        flow_args = ["singularity", "exec", "/storage/liberec3-tul/home/martin_spetlik/flow_3_1_0.sif", "flow123d"]

        flow_args.extend(['--output_dir', out_dir, os.path.join(out_dir, in_f)])

        # print("flow args ", flow_args)
        # print("out dir ", out_dir)

        if os.path.exists(os.path.join(out_dir, DFMSim.FIELDS_FILE)):
            return True
        with open(base + "_stdout", "w") as stdout:
            with open(base + "_stderr", "w") as stderr:
                completed = subprocess.run(flow_args, stdout=stdout, stderr=stderr)
            print("Exit status: ", completed.returncode)
            status = completed.returncode == 0
        conv_check = DFMSim.check_conv_reasons(os.path.join(out_dir, "flow123.0.log"))
        print("converged: ", conv_check)

        return status, p_loads, outer_reg_names  # and conv_check
        # return  status, p_loads, outer_reg_names  # and conv_check

    @staticmethod
    def _run_sample(flow_problem, config):
        """
        Create random fields file, call Flow123d and extract results
        :param fields_file: Path to file with random fields
        :param ele_ids: Element IDs in computational mesh
        :param fine_input_sample: fields: {'field_name' : values_array, ..}
        :param flow123d: Flow123d command
        :param common_files_dir: Directory with simulations common files (flow_input.yaml, )
        :return: simulation result, ndarray
        """
        #if homogenization:
        outer_reg_names = []
        for reg in flow_problem.side_regions:
            outer_reg_names.append(reg.name)
            outer_reg_names.append(reg.sub_reg.name)

        base = flow_problem.basename
        outer_regions_list = outer_reg_names
        in_f = in_file(flow_problem.basename)
        out_dir = flow_problem.basename
        # n_loads = len(self.p_loads)
        # flow_in = "flow_{}.yaml".format(self.base)
        params = dict(
            mesh_file=mesh_file(flow_problem.basename),
            fields_file=fields_file(flow_problem.basename),
            #outer_regions=str(outer_regions_list),
        )

        out_dir = os.getcwd()

        common_files_dir = config["fine"]["common_files_dir"]
        substitute_placeholders(os.path.join(common_files_dir, DFMSim.YAML_TEMPLATE), in_f, params)
        flow_args = ["singularity", "exec", "/storage/liberec3-tul/home/martin_spetlik/flow_3_1_0.sif", "flow123d"]

        flow_args.extend(['--output_dir', out_dir, os.path.join(out_dir, in_f)])

        if os.path.exists(os.path.join(out_dir, DFMSim.FIELDS_FILE)):
            return True
        with open(base + "_stdout", "w") as stdout:
            with open(base + "_stderr", "w") as stderr:
                completed = subprocess.run(flow_args, stdout=stdout, stderr=stderr)
            print("Exit status: ", completed.returncode)
            status = completed.returncode == 0
        conv_check = DFMSim.check_conv_reasons(os.path.join(out_dir, "flow123.0.log"))
        print("converged: ", conv_check)

        return DFMSim._extract_result(os.getcwd()), status # and conv_check
        #return  status, p_loads, outer_reg_names  # and conv_check

    @staticmethod
    def check_conv_reasons(log_fname):
        with open(log_fname, "r") as f:
            for line in f:
                tokens = line.split(" ")
                try:
                    i = tokens.index('convergence')
                    if tokens[i + 1] == 'reason':
                        value = tokens[i + 2].rstrip(",")
                        conv_reason = int(value)
                        if conv_reason < 0:
                            print("Failed to converge: ", conv_reason)
                            return False
                except ValueError:
                    continue
        return True


        # mesh = gmsh_io.GmshIO(mesh_file)
        #
        # with open(fields_file, 'w') as fields_mshfile:
        #     mesh.write_ascii_data(fields_mshfile, ele_ids, fine_input_sample)
        #
        # st = time.time()
        # subprocess.call(
        #     ["docker", "run", "-v", "{}:{}".format(os.getcwd(), os.getcwd()), "-v", "{}:{}".format(common_files_dir, common_files_dir), *flow123d, "--yaml_balance", '-i',
        #
        #      os.getcwd(), '-s', "{}/flow_input.yaml".format(common_files_dir),
        #      "-o", os.getcwd(), ">{}/flow.out".format(os.getcwd())])
        # end = time.time() - st
        #
        # return DFMSim._extract_result(os.getcwd()), end

    #@staticmethod
    # def generate_random_sample(fields, coarse_step, n_fine_elements):
    #     """
    #     Generate random field, both fine and coarse part.
    #     Store them separeted.
    #     :return: Dict, Dict
    #     """
    #     fields_sample = fields.sample()
    #     fine_input_sample = {name: values[:n_fine_elements, None] for name, values in fields_sample.items()}
    #     coarse_input_sample = {}
    #     if coarse_step != 0:
    #         coarse_input_sample = {name: values[n_fine_elements:, None] for name, values in
    #                                fields_sample.items()}
    #
    #     return fine_input_sample, coarse_input_sample
    #
    # def _make_mesh(self, geo_file, mesh_file, fine_step):
    #     """
    #     Make the mesh, mesh_file: <geo_base>_step.msh.
    #     Make substituted yaml: <yaml_base>_step.yaml,
    #     using common fields_step.msh file for generated fields.
    #     :return:
    #     """
    #     if self.env['gmsh_version'] == 2:
    #         subprocess.call(
    #             [self.env['gmsh'], "-2", '-format', 'msh2', '-clscale', str(fine_step), '-o', mesh_file, geo_file])
    #     else:
    #         subprocess.call([self.env['gmsh'], "-2", '-clscale', str(fine_step), '-o', mesh_file, geo_file])
    #
    # @staticmethod
    # def extract_mesh(mesh_file):
    #     """
    #     Extract mesh from file
    #     :param mesh_file: Mesh file path
    #     :return: Dict
    #     """
    #     mesh = gmsh_io.GmshIO(mesh_file)
    #     is_bc_region = {}
    #     region_map = {}
    #     for name, (id, _) in mesh.physical.items():
    #         unquoted_name = name.strip("\"'")
    #         is_bc_region[id] = (unquoted_name[0] == '.')
    #         region_map[unquoted_name] = id
    #
    #     bulk_elements = []
    #     for id, el in mesh.elements.items():
    #         _, tags, i_nodes = el
    #         region_id = tags[0]
    #         if not is_bc_region[region_id]:
    #             bulk_elements.append(id)
    #
    #     n_bulk = len(bulk_elements)
    #     centers = np.empty((n_bulk, 3))
    #     ele_ids = np.zeros(n_bulk, dtype=int)
    #     point_region_ids = np.zeros(n_bulk, dtype=int)
    #
    #     for i, id_bulk in enumerate(bulk_elements):
    #         _, tags, i_nodes = mesh.elements[id_bulk]
    #         region_id = tags[0]
    #         centers[i] = np.average(np.array([mesh.nodes[i_node] for i_node in i_nodes]), axis=0)
    #         point_region_ids[i] = region_id
    #         ele_ids[i] = id_bulk
    #
    #     min_pt = np.min(centers, axis=0)
    #     max_pt = np.max(centers, axis=0)
    #     diff = max_pt - min_pt
    #     min_axis = np.argmin(diff)
    #     non_zero_axes = [0, 1, 2]
    #     # TODO: be able to use this mesh_dimension in fields
    #     if diff[min_axis] < 1e-10:
    #         non_zero_axes.pop(min_axis)
    #     points = centers[:, non_zero_axes]
    #
    #     return {'points': points, 'point_region_ids': point_region_ids, 'ele_ids': ele_ids, 'region_map': region_map}
    #
    def _substitute_yaml(self, yaml_tmpl, yaml_out):
        """
        Create substituted YAML file from the tamplate.
        :return:
        """
        param_dict = {}
        # field_tmpl = self.field_template
        # for field_name in self._fields.names:
        #     param_dict[field_name] = field_tmpl % (self.FIELDS_FILE, field_name)
        param_dict[self.MESH_FILE_VAR] = self.mesh_file
        param_dict[self.TIMESTEP_H1_VAR] = self.time_step_h1
        param_dict[self.TIMESTEP_H2_VAR] = self.time_step_h2
        used_params = substitute_placeholders(yaml_tmpl, yaml_out, param_dict)

        self._fields_used_params = used_params

    @staticmethod
    def _extract_result(sample_dir):
        """
        Extract the observed value from the Flow123d output.
        :param sample_dir: str, path to sample directory
        :return: None, inf or water balance result (float) and overall sample time
        """
        # extract the flux
        balance_file = os.path.join(sample_dir, "water_balance.yaml")

        with open(balance_file, "r") as f:
            balance = yaml.load(f)

        flux_regions = ['.side_2']
        total_flux = 0.0
        found = False
        for flux_item in balance['data']:
            if flux_item['time'] > 0:
                break

            if flux_item['region'] in flux_regions:
                flux = float(flux_item['data'][0])
                flux_in = float(flux_item['data'][1])
                if flux_in > 1e-10:
                    raise Exception("Possitive inflow at outlet region.")
                total_flux += flux  # flux field
                found = True

        # Get flow123d computing time
        # run_time = FlowSim.get_run_time(sample_dir)

        if not found:
            raise Exception
        return np.array([-total_flux])

    # @staticmethod
    # def _extract_result(sample_dir):
    #     """
    #     Extract the observed value from the Flow123d output.
    #     :param sample_dir: str, path to sample directory
    #     :return: None, inf or water balance result (float) and overall sample time
    #     """
    #     # extract the flux
    #     balance_file = os.path.join(sample_dir, "water_balance.yaml")
    #
    #     with open(balance_file, "r") as f:
    #         balance = yaml.load(f)
    #
    #     flux_regions = ['.side_0', '.side_1', '.side_2', '.side_3']
    #     total_flux = 0.0
    #     found = False
    #     for flux_item in balance['data']:
    #         if flux_item['time'] > 0:
    #             break
    #
    #         if flux_item['region'] in flux_regions:
    #             flux = float(flux_item['data'][0])
    #             #flux_in = float(flux_item['data'][1])
    #             # if flux_in > 1e-10:
    #             #     raise Exception("Positive inflow at outlet region.")
    #             total_flux += flux  # flux field
    #             found = True
    #
    #     # Get flow123d computing time
    #     # run_time = FlowSim.get_run_time(sample_dir)
    #
    #     if not found:
    #         raise Exception
    #     return np.array([-total_flux])

    @staticmethod
    def result_format() -> List[QuantitySpec]:
        """
        Define simulation result format
        :return: List[QuantitySpec, ...]
        """
        spec1 = QuantitySpec(name="conductivity", unit="m", shape=(1, 1), times=[1], locations=['0'])
        # spec2 = QuantitySpec(name="width", unit="mm", shape=(2, 1), times=[1, 2, 3], locations=['30', '40'])
        return [spec1]

    @staticmethod
    def excluded_area(r_min, r_max, kappa, coef=1):
        norm_coef = kappa / (r_min ** (-kappa) - r_max ** (-kappa))
        return coef * (norm_coef * (r_max ** (1 - kappa) - r_min ** (1 - kappa)) / (1 - kappa)) ** 2

    @staticmethod
    def calculate_mean_excluded_volume(r_min, r_max, kappa, geom=False):
        #print("r min ", r_min)
        #print("r max ", r_max)
        #print("kappa ", kappa)
        if geom:
            # return 0.5 * (kappa / (r_min**(-kappa) - r_max**(-kappa)))**2 * 2*(((r_max**(2-kappa) - r_min**(2-kappa))) * ((r_max**(1-kappa) - r_min**(1-kappa))))/(kappa**2 - 3*kappa + 2)
            return ((r_max ** (1.5 * kappa - 0.5) - r_min ** (1.5 * kappa - 0.5)) / (
                    (-1.5 * kappa - 0.5) * (r_min ** (-1 * kappa) - r_max ** (-1 * kappa)))) ** 2
        else:
            return 0.5 * (kappa / (r_min ** (-kappa) - r_max ** (-kappa))) ** 2 * 2 * (
                        ((r_max ** (2 - kappa) - r_min ** (2 - kappa))) * (
                (r_max ** (1 - kappa) - r_min ** (1 - kappa)))) / (kappa ** 2 - 3 * kappa + 2)

    @staticmethod
    def calculate_mean_fracture_size(r_min, r_max, kappa, power=1):
        f0 = (r_min ** (-kappa) - r_max ** (-kappa)) / kappa
        return (1 / f0) * (r_max ** (-kappa + power) - r_min ** (-kappa + power)) / (-kappa + power)

    @staticmethod
    def generate_fractures(config):
        #print("config ", config)
        sim_config = config["sim_config"]
        geom = sim_config["geometry"]
        lx, ly = geom["fractures_box"]
        fr_size_range = geom["pow_law_size_range"]
        pow_law_exp_3d = geom["pow_law_size_exp"]
        pow_law_sample_range = geom["pow_law_sample_range"]
        rho = False
        rho_2D = False
        if "rho" in geom:
            rho = geom["rho"]  # P_30 * V_ex
        if "rho_2D" in geom:
            rho_2D = geom["rho_2D"]  # P_30 * V_ex
        n_frac_limit = geom["n_frac_limit"]
        #print("n frac limit ", n_frac_limit)
        #print("fr size range ", fr_size_range)
        p_32 = geom["p_32"]

        #print("lx: {}, ly: {} ".format(lx, ly))

        # generate fracture set
        fracture_box = [lx, ly, 0]
        area = lx * ly

        #print("pow_law_sample_range ", pow_law_sample_range)

        if rho_2D is not False:
            sim_config["fracture_model"]["max_fr"] = pow_law_sample_range[1]
            A_ex = BothSample.excluded_area(pow_law_sample_range[0], pow_law_sample_range[1],
                                            kappa=pow_law_exp_3d - 1, coef=np.pi / 2)
            #print("A_ex ", A_ex)
            # rho_2D = N_f/A * A_ex, N_f/A = intensity
            #print("rho_2D ", rho_2D)

            intensity = rho_2D / A_ex
            #print("intensity ", intensity)

            pop = fracture.Population(area, fracture.LineShape)
            pop.add_family("all",
                           fracture.FisherOrientation(0, 90, np.inf),
                           fracture.VonMisesOrientation(0, 0),
                           fracture.PowerLawSize(pow_law_exp_3d - 1, pow_law_sample_range, intensity)
                           )

            pop.set_sample_range(pow_law_sample_range)

        elif rho is not False:
            V_ex = BothSample.calculate_mean_excluded_volume(r_min=fr_size_range[0],
                                                             r_max=fr_size_range[1],
                                                             kappa=pow_law_exp_3d - 1)

            v_ex = BothSample.calculate_mean_fracture_size(r_min=fr_size_range[0],
                                                           r_max=fr_size_range[1],
                                                           kappa=pow_law_exp_3d - 1, power=1)

            R2 = BothSample.calculate_mean_fracture_size(r_min=fr_size_range[0],
                                                         r_max=fr_size_range[1],
                                                         kappa=pow_law_exp_3d - 1, power=2)

            #print("V_ex ", V_ex)
            #print("rho ", rho)
            p_30 = rho / V_ex
            #print("p_30 ", p_30)

            #print("v_ex ", v_ex)
            #print("R2 ", R2)

            p_30 = rho / (v_ex * R2)
            #print("final P_30 ", p_30)

            pop = fracture.Population(area, fracture.LineShape)
            pop.add_family("all",
                           fracture.FisherOrientation(0, 90, np.inf),
                           fracture.VonMisesOrientation(0, 0),
                           fracture.PowerLawSize(pow_law_exp_3d - 1, fr_size_range, p_30)
                           )

            pop.set_sample_range(pow_law_sample_range)

        else:
            pop = fracture.Population(area, fracture.LineShape)
            pop.add_family("all",
                           fracture.FisherOrientation(0, 90, np.inf),
                           fracture.VonMisesOrientation(0, 0),
                           fracture.PowerLawSize.from_mean_area(pow_law_exp_3d - 1, fr_size_range, p_32, pow_law_exp_3d)
                           )
            if n_frac_limit is not False:
                # pop.set_sample_range([None, np.min([lx, ly, self.config_dict["geometry"]["fr_max_size"]])],
                #                      sample_size=n_frac_limit)
                # pop.set_sample_range([None, max(lx, ly)],
                #                      sample_size=n_frac_limit)

                pop.set_sample_range(fr_size_range,
                                     sample_size=n_frac_limit)
            elif pow_law_sample_range:
                pop.set_sample_range(pow_law_sample_range)

        #print("total mean size: ", pop.mean_size())
        #print("size range:", pop.families[0].size.sample_range)


        pos_gen = fracture.UniformBoxPosition(fracture_box)
        fractures = pop.sample(pos_distr=pos_gen, keep_nonempty=False)

        #print("fractures len ", len(fractures))
        #print("fractures ", fractures)
        #print("fr_size_range[0] ", fr_size_range[0])

        fr_set = fracture.Fractures(fractures, fr_size_range[0] / 2)

        return fr_set

    def make_summary(done_list):
        results = {problem.basename: problem.summary() for problem in done_list}

        #print("results ", results)

        with open("summary.yaml", "w") as f:
            yaml.dump(results, f)



