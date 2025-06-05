import os
import os.path
import subprocess
import numpy as np
import shutil
import time
import copy
import torch
import itertools
import joblib
#import ruamel.yaml as yaml
import yaml
from typing import List
import gstools
from mlmc.level_simulation import LevelSimulation
import gmsh_io
from mlmc.sim.simulation import Simulation
from mlmc.quantity.quantity_spec import QuantitySpec
#from mlmc.random import correlated_field as cf
import homogenization.fracture as fracture
#from homogenization.both_sample import FlowProblem, BothSample
#from metamodel.cnn.datasets import create_dataset
from metamodel.cnn.postprocess.optuna_results import load_study, load_models, get_saved_model_path, get_inverse_transform, get_transform
from metamodel.cnn3D.datasets.dfm3d_dataset import DFM3DDataset
#from npy_append_array import NpyAppendArray
import numpy.random as rnd
from pathlib import Path
import pyvista as pv
from homogenization.gstools_bulk_3D import GSToolsBulk3D
from bgem import stochastic
from bgem import stochastic
from bgem.gmsh import gmsh, options
from mesh_class import Mesh
from bgem.core import call_flow, dotdict, workdir as workdir_mng
from bgem.upscale import fem_plot, fem, voigt_to_tn, tn_to_voigt, FracturedMedia, voxelize
from bgem.upscale.homogenization import equivalent_posdef_tensor
import decovalex_dfnmap as dmap
from typing import *
import zarr
from bgem.stochastic import FractureSet, EllipseShape, PolygonShape
#from bgem.upscale.voxelize import fr_conductivity
from bgem.upscale import *
import scipy.interpolate as sc_interpolate
from bgem.gmsh.gmsh import ObjectSet

#os.environ["CUDA_VISIBLE_DEVICES"]=""


print("torch.get_num_threads() ", torch.get_num_threads())
torch.set_num_threads(torch.get_num_threads())  # Max out CPU cores
torch.backends.mkldnn.enabled = True  # Use MKL-DNN for Conv3D
torch.backends.openmp.enabled = True  # Enable OpenMP

# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# torch.set_num_threads(1)


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


class DFMSim3D(Simulation):
    # placeholders in YAML
    model = None
    checkpoint = None
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
    YAML_TEMPLATE_H_VTK = "flow_input_h_vtk.yaml.tmpl"
    YAML_TEMPLATE = 'flow_input.yaml.tmpl'
    YAML_FILE = 'flow_input.yaml'
    FIELDS_FILE = "flow_fields.msh"
    COND_TN_POP_FILE = 'cond_tn_pop.npy'
    PRED_COND_TN_POP_FILE = 'pred_cond_tn_pop.npy'
    COND_TN_FILE = "cond_tensors.yaml"
    PRED_COND_TN_FILE = "pred_cond_tensors.yaml"
    COMMON_FILES = "l_step_{}_common_files"
    SAMPLES_COND_TNS_DIR = "samples_cond_tns"
    ZARR_FILE = "samples_data.zarr"

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
        self.base_yaml_file_homogenization_vtk = config['yaml_file_homogenization_vtk']
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
        common_files_dir = os.path.join(self.work_dir, DFMSim3D.COMMON_FILES.format(fine_step))
        force_mkdir(common_files_dir, force=self.clean)

        self.mesh_file = os.path.join(common_files_dir, DFMSim3D.MESH_FILE)

        if self.clean:
            # Prepare mesh
            #geo_file = os.path.join(common_files_dir, DFMSim3D.GEO_FILE)
            #shutil.copyfile(self.base_geo_file, geo_file)
            #self._make_mesh(geo_file, self.mesh_file, fine_step)  # Common computational mesh for all samples.

            # Prepare main input YAML
            yaml_template_h = os.path.join(common_files_dir, DFMSim3D.YAML_TEMPLATE_H)
            shutil.copyfile(self.base_yaml_file_homogenization, yaml_template_h)

            yaml_template_h_vtk = os.path.join(common_files_dir, DFMSim3D.YAML_TEMPLATE_H_VTK)
            shutil.copyfile(self.base_yaml_file_homogenization_vtk, yaml_template_h_vtk)

            yaml_template = os.path.join(common_files_dir, DFMSim3D.YAML_TEMPLATE)
            shutil.copyfile(self.base_yaml_file, yaml_template)

            yaml_file = os.path.join(common_files_dir, DFMSim3D.YAML_FILE)
            self._substitute_yaml(yaml_template, yaml_file)

            yaml_file = os.path.join(common_files_dir, DFMSim3D.YAML_FILE)
            self._substitute_yaml(yaml_template, yaml_file)
        #
        # # Mesh is extracted because we need number of mesh points to determine task_size parameter (see return value)
        # fine_mesh_data = self.extract_mesh(self.mesh_file)

        #@TODO: determine task size
        # Set coarse simulation common files directory
        # Files in the directory are used by each simulation at that level
        coarse_sim_common_files_dir = None
        if coarse_step != 0:
            coarse_sim_common_files_dir = os.path.join(self.work_dir, DFMSim3D.COMMON_FILES.format(coarse_step))
            samples_cond_tns = os.path.join(coarse_sim_common_files_dir, DFMSim3D.SAMPLES_COND_TNS_DIR)
            force_mkdir(samples_cond_tns, force=self.clean)

            #force_mkdir(coarse_sim_common_files_dir, force=self.clean)

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

        if coarse_step != 0:
            config["coarse"]["sample_cond_tns"] = samples_cond_tns
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
                               calculate=DFMSim3D.calculate,
                               # method which carries out the calculation, will be called from PBS processs
                               need_sample_workspace=True # If True, a sample directory is created
                               )
    @staticmethod
    def eliminate_far_points(outer_polygon, center_cond_field, fine_step=0):
        centers, cond_field = center_cond_field
        centers = np.array(centers)
        cond_field = np.array(cond_field)
        # print("outer polygon ", outer_polygon)
        # print("centers ", centers)
        # print("fine step ", fine_step)
        #exit()

        # indices = np.where((outer_polygon[0][0] - fine_step <= centers[:,0]) & (centers[:,0] <= outer_polygon[1][0] + fine_step)
        #                      & (outer_polygon[0][1] - fine_step <= centers[:,1]) & (centers[:,1] <= outer_polygon[2][1] + fine_step))[0]

        # print("indices ", indices)
        # print("len indices ", len(indices))

        #rest = centers[indices]
        #print("Rest ", rest)
        #print("cond field shape ", cond_field.shape)
        # #cond_field = ncond_field)
        # print("cond_field[indices] ", cond_field[indices])
        # exit()

        #return centers[indices], cond_field[indices]
        return centers, cond_field

    # @staticmethod
    # def get_outer_cube(center_x, center_y, center_z, box_size_x, box_size_y, box_size_z):
    #     bl_corner = [center_x - box_size_x / 2, center_y - box_size_y / 2]
    #     br_corner = [center_x + box_size_x / 2, center_y - box_size_y / 2]
    #     tl_corner = [center_x - box_size_x / 2, center_y + box_size_y / 2]
    #     tr_corner = [center_x + box_size_x / 2, center_y + box_size_y / 2]
    #
    #     # print("center x: {}, y: {}".format(center_x, center_y))
    #
    #     return [bl_corner, br_corner, tr_corner, tl_corner]


    @staticmethod
    def split_domain(config, dataset_path, n_split_subdomains, fractures, sample_dir, seed=None, fields_file="fields_fine_to_rast.msh", mesh_file="mesh_fine_to_rast.msh"):
        domain_box = config["sim_config"]["geometry"]["domain_box"]
        x_subdomains = int(np.sqrt(n_split_subdomains))
        subdomain_box = [domain_box[0]/x_subdomains, domain_box[1]/x_subdomains]

        if x_subdomains == 1:
            sample_0_path = os.path.join(dataset_path, "sample_0")
            os.mkdir(sample_0_path)
            shutil.copy(os.path.join(sample_dir, fields_file), os.path.join(sample_0_path, "fields_fine.msh"))
            shutil.copy(os.path.join(sample_dir, mesh_file), os.path.join(sample_0_path, "mesh_fine.msh"))
            return sample_0_path
        else:
            lx, ly = domain_box
            k = 0
            sample_center = {}
            for i in range(x_subdomains):
                center_x = subdomain_box[0] / 2 + (lx - subdomain_box[0]) / (n_split_subdomains - 1) * i - lx / 2
                for j in range(x_subdomains):
                    k += 1
                    # subdir_name = "i_{}_j_{}_k_{}".format(i, j, k)
                    # os.mkdir(subdir_name)
                    # os.chdir(subdir_name)
                    center_y = subdomain_box[1] / 2 + (lx - subdomain_box[1]) / (n_split_subdomains - 1) * j - lx / 2
                    box_size_x = subdomain_box[0]
                    box_size_y = subdomain_box[1]
                    #outer_polygon = DFMSim3D.get_outer_cube(center_x, center_y, box_size_x, box_size_y)
                    #config["sim_config"]["geometry"]["outer_polygon"] = outer_polygon

                    while True:
                        if fractures is None:
                            fractures = DFMSim3D.generate_fractures(config)

                        fine_flow = FlowProblem.make_fine((config["fine"]["step"],
                                                           config["sim_config"]["geometry"]["fr_max_size"]),
                                                          fractures,
                                                          config, seed=config["sim_config"]["seed"] + seed)
                        fine_flow.fr_range = [config["fine"]["step"], config["coarse"]["step"]]

                        cond_fields = config["center_cond_field"]
                        if "center_larger_cond_field" in config:
                            cond_fields = config["center_larger_cond_field"]

                        center_cond_field = DFMSim3D.eliminate_far_points(outer_polygon,
                                                                        cond_fields,
                                                                        fine_step=config["fine"]["step"])
                        try:
                            fine_flow.make_mesh(center_box=([center_x, center_y], subdomain_box))
                        except:
                            pass

                        if not os.path.exists("mesh_fine.msh"):
                            box_size_x += box_size_x * 0.05
                            box_size_y += box_size_y * 0.05
                            outer_polygon = DFMSim3D.get_outer_cube(center_x, center_y, box_size_x, box_size_y)
                            config["sim_config"]["geometry"]["outer_polygon"] = outer_polygon
                            print("new outer polygon make_mesh failed", outer_polygon)
                            continue

                        #print("center cond field ", center_cond_field)
                        #fine_flow.interpolate_fields(center_cond_field, mode="linear")

                        fine_flow.interpolate_fields(center_cond_field, mode="linear")

                        # fine_flow.make_fields()
                        break

                    sample_name = "sample_{}".format(k - 1)
                    dset_sample_dir = os.path.join(dataset_path, sample_name)
                    os.mkdir(dset_sample_dir)
                    shutil.copy("fields_fine.msh", dset_sample_dir)
                    shutil.copy("mesh_fine.msh", dset_sample_dir)
                    # shutil.copy("summary.yaml", dset_dir)
                    sample_center[sample_name] = (center_x, center_y)

    # @staticmethod
    # def rasterize_at_once(sample_dir, dataset_path, config, n_subdomains, fractures, hom_dir, seed=None, fields_file="fields_fine_to_rast.msh", mesh_file="mesh_fine_to_rast.msh" ):
    #     n_non_overlapping_subdomains = n_subdomains - ((n_subdomains-1)/2)
    #
    #     # import psutil
    #     # float_size_bytes = 8  # Assuming double precision (8 bytes)
    #     # available_memory = psutil.virtual_memory().available
    #     # print(f"Available memory: {available_memory} bytes")
    #     # max_floats = (available_memory // float_size_bytes) * 0.8
    #     max_floats = 10000000000 # 10 Gb
    #
    #     #print("n subdomains ", n_subdomains)
    #
    #     n_pixels = 256
    #     total_n_pixels_x = n_pixels**2 * n_non_overlapping_subdomains **2
    #     #total_n_pixels_x = 4**2
    #     print("total n pixels x ", total_n_pixels_x)
    #     split_exp = 0
    #     if total_n_pixels_x >= max_floats:
    #         split_exp = 1
    #         tot_pixels = total_n_pixels_x
    #
    #         tot_pixels /= 4**split_exp
    #         while tot_pixels >= max_floats:
    #             split_exp += 1
    #             tot_pixels /= 4 ** split_exp
    #
    #     if split_exp > 1:
    #         raise Exception("Total number of pixels: {} does not fit into memory. Not supported yet".format(total_n_pixels_x))
    #
    #     sample_0_path = DFMSim3D.split_domain(config, dataset_path, n_split_subdomains=4**split_exp, fractures=fractures, sample_dir=sample_dir, seed=seed, fields_file=fields_file, mesh_file=mesh_file)
    #
    #     print("dataset path ", dataset_path)
    #
    #     ####################
    #     ## Create dataset ##
    #     ####################
    #     process = subprocess.run(["bash", config["sim_config"]["create_dataset_script"], dataset_path, "{}".format(int(np.sqrt(total_n_pixels_x)))],
    #                              capture_output=True, text=True)
    #     pred_cond_tensors = {}
    #
    #     if process.returncode == 0:
    #         bulk_path = os.path.join(sample_0_path, "bulk.npz")
    #         fractures_path = os.path.join(sample_0_path, "fractures.npz")
    #         cross_section_path = os.path.join(sample_0_path, "cross_sections.npz")
    #         bulk = np.load(bulk_path)["data"]
    #         fractures = np.load(fractures_path)["data"]
    #         cross_section = np.load(cross_section_path)["data"]
    #         domain_box = config["sim_config"]["geometry"]["orig_domain_box"]
    #         subdomain_box = config["sim_config"]["geometry"]["subdomain_box"]
    #
    #         #print("bulk.shape ", bulk.shape)
    #         #print("fractures.shape ", fractures.shape)
    #
    #         final_dataset_path = os.path.join(hom_dir, "final_dataset")
    #         os.mkdir(final_dataset_path)
    #         subdomain_size = 256
    #         sample_center = {}
    #         lx, ly = domain_box
    #         lx += subdomain_box[0]
    #         ly += subdomain_box[1]
    #         k = 0
    #
    #         #print("subdomain box ", subdomain_box)
    #
    #         # print("bulk[0, ...] ", bulk[0, ...])
    #         print("rasterize n subdomains ", n_subdomains)
    #         for i in range(n_subdomains):
    #             #print("lx ", lx)
    #             center_y = -(subdomain_box[0] / 2 + (lx - subdomain_box[0]) / (n_subdomains - 1) * i - lx / 2)
    #             for j in range(n_subdomains):
    #                 sample_name = "sample_{}".format(k)
    #                 print("sample name ", sample_name)
    #                 f_dset_sample_dir = os.path.join(final_dataset_path, sample_name)
    #                 os.mkdir(f_dset_sample_dir)
    #
    #                 # print("subdomain box ", subdomain_box)
    #                 # exit()
    #
    #                 center_x = (subdomain_box[1] / 2 + (lx - subdomain_box[1]) / (n_subdomains - 1) * j - lx / 2)
    #
    #                 #print("center(i: {}, j:{}) x: {}, y: {}".format(i, j, center_x, center_y))
    #                 # print("x from: {}, to: {}".format(i*int((subdomain_size)/2), i*int((subdomain_size)/2) + subdomain_size))
    #                 # print("y from: {} to: {}".format(j*int((subdomain_size)/2), j*int((subdomain_size)/2) + subdomain_size))
    #                 # print("j*int((subdomain_size)/2) + subdomain_size ", j*int((subdomain_size)/2) + subdomain_size)
    #
    #                 bulk_subdomain = bulk[:, i*int((subdomain_size)/2): i*int((subdomain_size)/2) + subdomain_size,
    #                                  j*int((subdomain_size)/2): j*int((subdomain_size)/2) + subdomain_size]
    #                 fractures_subdomain = fractures[:, i * int((subdomain_size) / 2): i * int((subdomain_size) / 2) + subdomain_size,
    #                                  j * int((subdomain_size) / 2): j * int((subdomain_size) / 2) + subdomain_size]
    #                 cross_section_subdomain = cross_section[:,
    #                                       i * int((subdomain_size) / 2): i * int((subdomain_size) / 2) + subdomain_size,
    #                                       j * int((subdomain_size) / 2): j * int((subdomain_size) / 2) + subdomain_size]
    #
    #                 #print("bulk_subdomain[0,...] ", bulk_subdomain[0,...])
    #
    #                 np.savez_compressed(os.path.join(f_dset_sample_dir, "bulk"), data=bulk_subdomain)
    #                 np.savez_compressed(os.path.join(f_dset_sample_dir, "fractures"), data=fractures_subdomain)
    #                 np.savez_compressed(os.path.join(f_dset_sample_dir, "cross_sections"), data=cross_section_subdomain)
    #
    #                 #print("sample name ", sample_name)
    #                 sample_center[sample_name] = (center_x, center_y)
    #                 k += 1
    #
    #         if DFMSim3D.model is None:
    #             nn_path = config["sim_config"]["nn_path"]
    #             study = load_study(nn_path)
    #             model_path = get_saved_model_path(nn_path, study.best_trial)
    #             model_kwargs = study.best_trial.user_attrs["model_kwargs"]
    #             DFMSim3D.model = study.best_trial.user_attrs["model_class"](**model_kwargs)
    #             if not torch.cuda.is_available():
    #                 DFMSim3D.checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    #             else:
    #                 DFMSim3D.checkpoint = torch.load(model_path)
    #             DFMSim3D.inverse_transform = get_inverse_transform(study, results_dir=nn_path)
    #             DFMSim3D.transform = get_transform(study, results_dir=nn_path)
    #
    #         ##
    #         # Create dataset
    #         ##
    #         dataset_for_prediction = DFMDataset(data_dir=final_dataset_path,
    #                                             # output_file_name=output_file_name,
    #                                             init_transform=DFMSim3D.transform[0],
    #                                             input_transform=DFMSim3D.transform[1],
    #                                             output_transform=DFMSim3D.transform[2],
    #                                             two_dim=True,
    #                                             cross_section=True,
    #                                             # input_channels=config[
    #                                             #     "input_channels"] if "input_channels" in config else None,
    #                                             # output_channels=config[
    #                                             #     "output_channels"] if "output_channels" in config else None,
    #                                             # fractures_sep=config[
    #                                             #     "fractures_sep"] if "fractures_sep" in config else False,
    #                                             # vel_avg=config["vel_avg"] if "vel_avg" in config else False
    #                                             )
    #
    #         dset_prediction_loader = torch.utils.data.DataLoader(dataset_for_prediction, batch_size=1, shuffle=False)
    #         with torch.no_grad():
    #             DFMSim3D.model.load_state_dict(DFMSim3D.checkpoint['best_model_state_dict'])
    #             DFMSim3D.model.eval()
    #
    #             for i, sample in enumerate(dset_prediction_loader):
    #                 inputs, targets = sample
    #                 print("inputs ", inputs.shape)
    #                 inputs = inputs.float()
    #                 sample_n = dataset_for_prediction._bulk_file_paths[i].split('/')[-2]
    #                 center = sample_center[sample_n]
    #                 # if args.cuda and torch.cuda.is_available():
    #                 #    inputs = inputs.cuda()
    #                 predictions = DFMSim3D.model(inputs)
    #                 predictions = np.squeeze(predictions)
    #
    #                 print("predictions ", predictions)
    #
    #                 # if np.any(predictions < 0):
    #                 #     print("inputs ", inputs)
    #                 #     print("negative predictions ", predictions)
    #
    #                 inv_predictions = torch.squeeze(
    #                     DFMSim3D.inverse_transform(torch.reshape(predictions, (*predictions.shape, 1, 1))))
    #
    #
    #                 #print("inv predictions shape ", inv_predictions.shape)
    #
    #
    #                 if dataset_for_prediction.init_transform is not None:
    #                     inv_predictions *= dataset_for_prediction._bulk_features_avg
    #
    #                 pred_cond_tn = np.array([[inv_predictions[0], inv_predictions[1]],
    #                                          [inv_predictions[1], inv_predictions[2]]])
    #
    #                 if pred_cond_tn is not None:
    #                     pred_cond_tn_flatten = pred_cond_tn.flatten()
    #
    #                     if not np.any(np.isnan(pred_cond_tn_flatten)):
    #                         pred_cond_tensors[center] = [pred_cond_tn_flatten[0],
    #                                                      (pred_cond_tn_flatten[1] + pred_cond_tn_flatten[2]) / 2,
    #                                                      pred_cond_tn_flatten[3]]
    #
    #                         #print("pred cond tn: {}".format(pred_cond_tensors[center]))
    #                         # if pred_cond_tn_flatten[0] < 0:
    #                         #     print("inputs ", inputs)
    #
    #     else:
    #         raise Exception(process.stderr)
    #
    #     return pred_cond_tensors

    @staticmethod
    def create_zarr_file(dir_path, n_samples, config_dict, centers=False):
        input_shape_n_voxels = config_dict["sim_config"]["geometry"]["n_voxels"]
        input_shape_n_channels = 6  # 6 channels for cond_tn
        n_cond_tn_channels = 6  # 1 channel for cross_section
        # n_cross_section_channels = 1
        output_shape = (6,)
        centers_shape = (3,)
        chunk_size = 1

        zarr_file_path = os.path.join(dir_path, DFMSim3D.ZARR_FILE)

        if not os.path.exists(zarr_file_path):
            zarr_file = zarr.open(zarr_file_path, mode='w')
            # # Create the 'inputs' dataset with the specified shape
            inputs = zarr_file.create_dataset('inputs',
                                              shape=(n_samples,) + (input_shape_n_channels, *input_shape_n_voxels),
                                              dtype='float32',
                                              chunks=(chunk_size, n_cond_tn_channels, *input_shape_n_voxels),
                                              fill_value=0)
            # inputs[:, :, :, :, :] = np.zeros((n_samples, n_cond_tn_channels, *input_shape_n_voxels,
            #                                                         ))  # Populate the first 6 channels
            # inputs[:, :, :, :, n_cond_tn_channels] = np.random.rand(n_samples,
            #                                                         *input_shape_n_voxels)  # Populate the last channel

            # Create the 'outputs' dataset with the specified shape
            outputs = zarr_file.create_dataset('outputs', shape=(n_samples,) + output_shape, dtype='float32',
                                               chunks=(chunk_size, n_cond_tn_channels), fill_value=0)

            bulk_avg = zarr_file.create_dataset('bulk_avg', shape=(n_samples,) + output_shape, dtype='float32',
                                                chunks=(chunk_size, n_cond_tn_channels), fill_value=0)

            # Assign metadata to indicate channel names
            inputs.attrs['channel_names'] = ['cond_tn_0', 'cond_tn_1', 'cond_tn_2', 'cond_tn_3', 'cond_tn_4',
                                             'cond_tn_5']
            outputs.attrs['channel_names'] = ['cond_tn_0', 'cond_tn_1', 'cond_tn_2', 'cond_tn_3', 'cond_tn_4',
                                              'cond_tn_5']
            bulk_avg.attrs['channel_names'] = ['cond_tn_0', 'cond_tn_1', 'cond_tn_2', 'cond_tn_3', 'cond_tn_4',
                                               'cond_tn_5']

            if centers:
                centers_dset = zarr_file.create_dataset('centers', shape=(n_samples,) + centers_shape, dtype='float32',
                                                   chunks=(chunk_size), fill_value=0)
                centers_dset.attrs['channel_names'] = ['center_x', 'center_y', 'center_z']

        return zarr_file_path

    @staticmethod
    def predict_on_hom_sample(config, bulk_cond_values, bulk_cond_points, dfn,
                                        fr_cond,
                                        fem_grid_rast, centers):
        index = 0
        batch_size = 1

        n_steps = config["sim_config"]["geometry"]["n_voxels"]

        zarr_file_path = DFMSim3D.create_zarr_file(os.getcwd(), n_samples=1, config_dict=config, centers=True)
        zarr_file = zarr.open(zarr_file_path, mode='r+')

        bulk_cond_fem_rast = DFMSim3D._bulk_cond_to_rast_grid(bulk_cond_values, bulk_cond_points,
                                                              fem_grid_rast.grid)

        bulk_cond_fem_rast_voigt = tn_to_voigt(bulk_cond_fem_rast)
        bulk_cond_fem_rast_voigt = bulk_cond_fem_rast_voigt.reshape(*n_steps,
                                                                    bulk_cond_fem_rast_voigt.shape[-1]).T

        if len(dfn) == 0:
            rasterized_input = bulk_cond_fem_rast
        else:
            rasterized_input = DFMSim3D.rasterize(fem_grid_rast, dfn, bulk_cond=bulk_cond_fem_rast, fr_cond=fr_cond)

        rasterized_input_voigt = tn_to_voigt(rasterized_input)

        # print("before reshape rasterized_input_voigt ", rasterized_input_voigt)
        # print("before reshape rasterized_input_voigt shape", rasterized_input_voigt.shape)

        rasterized_input_voigt = rasterized_input_voigt.reshape(*n_steps, rasterized_input_voigt.shape[-1]).T

        # print("rasterized_input_voigt[:, 0, 0, 0] ", rasterized_input_voigt[:, 0, 0, 0])
        #
        # print("rasterized_input_voigt ", rasterized_input_voigt)
        # print("rasterized_input_voigt shape ", rasterized_input_voigt.shape)

        zarr_file["inputs"][index, :] = rasterized_input_voigt
        zarr_file["centers"][index, :] = centers

        #print("zarr_file[inputs][index, :].shape ", zarr_file["inputs"][index, :].shape)

        hom_block_bulk = bulk_cond_fem_rast_voigt

        # print("np.mean(hom_block_bulk, axis=(1, 2, 3)) ", np.mean(hom_block_bulk, axis=(1, 2, 3)))

        zarr_file["bulk_avg"][index, :] = np.mean(hom_block_bulk, axis=(1, 2, 3))

        #index += 1

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        import torch.autograd.profiler as profiler

        if DFMSim3D.model is None:
            nn_path = config["sim_config"]["nn_path_for_block_hom"]
            study = load_study(nn_path)
            model_path = get_saved_model_path(nn_path, study.best_trial)
            model_kwargs = study.best_trial.user_attrs["model_kwargs"]
            DFMSim3D.model = study.best_trial.user_attrs["model_class"](**model_kwargs)
            if not torch.cuda.is_available():
                DFMSim3D.checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            else:
                DFMSim3D.checkpoint = torch.load(model_path)
            DFMSim3D.inverse_transform = get_inverse_transform(study, results_dir=nn_path)
            DFMSim3D.transform = get_transform(study, results_dir=nn_path)

        ##
        # Create dataset
        ##
        dataset_for_prediction = DFM3DDataset(zarr_path=zarr_file_path,
                                              init_transform=DFMSim3D.transform[0],
                                              input_transform=DFMSim3D.transform[1],
                                              output_transform=DFMSim3D.transform[2],
                                              return_centers_bulk_avg=True)

        dset_prediction_loader = torch.utils.data.DataLoader(dataset_for_prediction, batch_size=batch_size,
                                                             shuffle=False)

        DFMSim3D.model.load_state_dict(DFMSim3D.checkpoint['best_model_state_dict'])

        # DFMSim3D.model.eval()

        DFMSim3D.model = DFMSim3D.model.to(memory_format=torch.channels_last_3d)
        DFMSim3D.model.to(device).eval()

        # model.half()

        # example_input = torch.randn(1, 6, 64, 64, 64)
        # scripted_model = torch.jit.trace(DFMSim3D.model, example_input)
        # scripted_model = torch.compile(DFMSim3D.model, backend="aot_eager")
        # output = scripted_model(data)

        # print(" torch.cuda.is_available() ", torch.cuda.is_available())

        # DFMSim3D.model = torch.compile(DFMSim3D.model)

        with torch.inference_mode():
            for i, sample in enumerate(dset_prediction_loader):
                # print("i ", i)
                inputs, targets, centers, bulk_features_avg = sample
                # print("inputs ", inputs.shape)
                # print("centers ", centers)
                # print("bulk features avg ", bulk_features_avg)

                inputs = inputs.to(memory_format=torch.channels_last_3d)  # Optimize for 3D convolution
                inputs = inputs.float().to(device)

                # inputs = inputs.contiguous()
                # sample_n = dataset_for_prediction._bulk_file_paths[i].split('/')[-2]
                # center = sample_center[sample_n]
                # print("torch.cuda.is_available() ", torch.cuda.is_available())
                # if torch.cuda.is_available():
                #     # print("cuda available")
                #     inputs = inputs.cuda()
                #     DFMSim3D.model = DFMSim3D.model.cuda()

                # with profiler.profile() as prof:
                #     #with profiler.record_function("conv3d"):
                #     predictions = DFMSim3D.model(inputs)
                # print(prof.key_averages().table(sort_by="cpu_time_total"))
                # exit()
                # with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                predictions = DFMSim3D.model(inputs)
                # predictions = model(inputs)
                print("predictions ", predictions)

                predictions = np.squeeze(predictions)

                # print("dset_prediction_loader ", dset_prediction_loader._bulk_features_avg)

                # print("zarr predictions ", predictions)
                # print("torch.reshape(predictions, (*predictions.shape, 1, 1))) ", torch.reshape(predictions, (*predictions.shape, 1, 1)).shape)

                # if np.any(predictions < 0):
                #     print("inputs ", inputs)
                #     print("negative predictions ", predictions)

                inv_predictions = torch.squeeze(
                    DFMSim3D.inverse_transform(torch.reshape(predictions, (*predictions.shape, 1, 1))))
                # print("inv predictions shape ", inv_predictions.shape)
                # print("bulk_features_avg.shape ", bulk_features_avg.shape)
                # print("inv predictions ", inv_predictions)

                print("inv predictions ", inv_predictions)

                if dataset_for_prediction.init_transform is not None:
                    # print("dataset_for_prediction._bulk_features_avg ", dataset_for_prediction._bulk_features_avg)
                    if len(inv_predictions.shape) > 1:
                        bulk_features_avg = bulk_features_avg.view(-1, 1)

                    print("bulk_features_avg ", bulk_features_avg)
                    # print("inv predictions ", inv_predictions)

                    inv_predictions *= bulk_features_avg

                print("inv predictions scaled ", inv_predictions)

                # return
                # return pred_cond_tensors

                # print("inv prediction shape ", inv_predictions.shape)

                # pred_cond_tn = np.array([[inv_predictions[0], inv_predictions[1]],
                #                          [inv_predictions[1], inv_predictions[2]]])

                # if pred_cond_tn is not None:

                # print("list(centers.numpy()) ", centers.tolist())
                # print("type list(centers.numpy()) ", type(centers.tolist()))

                # print("centers ", centers)
                # print("centers.numpy ", centers.numpy())
                # print("centers.numpy.shape ", centers.numpy().shape)
                # print("inv_predictions.numpy() ", inv_predictions.numpy().shape)
                # print("len(inv_predictions.numpy().shape) ", len(inv_predictions.numpy().shape))

                inv_predictions_numpy = inv_predictions.numpy()
                return inv_predictions_numpy

    @staticmethod
    def extract_subdomain(grid_values, coordinates, center, domain_size):
        """
        Extracts a subdomain of grid values around a given center point.

        Parameters:
        - grid_values: np.ndarray of shape (N, 3, 3), the values on a regular grid.
        - coordinates: np.ndarray of shape (N, 3), the x, y, z coordinates.
        - center: tuple (cx, cy, cz), center of the subdomain.
        - domain_size: tuple (dx, dy, dz), size of the subdomain.

        Returns:
        - subdomain_values: np.ndarray, the extracted subdomain values.
        - subdomain_coords: np.ndarray, the corresponding coordinates.
        """
        # Compute the bounding box limits
        lower_bounds = np.array(center) - np.array(domain_size) / 2
        upper_bounds = np.array(center) + np.array(domain_size) / 2
        # Apply boolean mask to filter coordinates inside the subdomain
        mask = np.all((coordinates >= lower_bounds) & (coordinates <= upper_bounds), axis=1)

        # Extract subdomain values and coordinates
        subdomain_values = grid_values[mask]
        subdomain_coords = coordinates[mask]

        return subdomain_values, subdomain_coords

    @staticmethod
    def extract_dfn(dfn, dfn_list, domain_center, domain_size):
        fracture_centers = dfn.center
        # Compute bounding box limits
        lower_bounds = np.array(domain_center) - np.array(domain_size) / 2
        upper_bounds = np.array(domain_center) + np.array(domain_size) / 2
        # Boolean mask: Select fractures whose centers are within the bounding box
        mask = np.all((fracture_centers >= lower_bounds) & (fracture_centers <= upper_bounds), axis=1)
        subdomain_dfn_list = np.array(dfn_list)[mask]
        subdomain_dfn = stochastic.FractureSet.from_list(subdomain_dfn_list)
        return subdomain_dfn

    @staticmethod
    def homogenization(config, dfn_to_homogenize, dfn_to_homogenize_list, bulk_cond_values, bulk_cond_points, seed=None, hom_dir_name="homogenization"):
        sample_dir = os.getcwd()

        print("homogenization method")
        print("os.getcwd() ", os.getcwd())
        # print("config[scratch_dir] ", config["sim_config"]["scratch_dir"])
        print("os environ get SCRATCHDIR ", os.environ.get("SCRATCHDIR"))

        #hom_dir_abs_path = os.path.join(sample_dir, hom_dir_name)
        #hom_dir_name = "homogenization"
        if os.environ.get("SCRATCHDIR") is not None:
            hom_dir_abs_path = os.path.join(os.environ.get("SCRATCHDIR"), hom_dir_name)
        else:
            hom_dir_abs_path = os.path.join(sample_dir, hom_dir_name)

        print("hom_dir_abs_path ", hom_dir_abs_path)

        # dfn_to_homogenization = []
        # #fr_rad_values = []
        # for fr in dfn:
        #     if fr.r <= config["coarse"]["step"]:
        #         #print("fr.radius ", fr.radius)
        #         #fr_rad_values.append(fr.radius[0]* fr.radius[1]**2 + fr.radius[0]**2*fr.radius[1])
        #         dfn_to_homogenization.append(fr)
        # if len(dfn_to_homogenization) > 0:
        #     dfn = stochastic.FractureSet.from_list(dfn_to_homogenization)
        # else:
        #     dfn = []
        # #print("len dfn_to_homogenization ", len(dfn_to_homogenization))
        #
        # #print("mean fr rad values ", np.mean(fr_rad_values))

        #fr_media = FracturedMedia.fracture_cond_params(dfn_to_homogenize, 1e-4, 0.00001)

        # for fr in dfn:
        #     print("hom fr.r ", fr.r)

        #hom_dir_name = "homogenization"
        # if "scratch_dir" in config["sim_config"] and config["sim_config"]["scratch_dir"] is not None:
        #     #hom_dir_abs_path = os.path.join(config["sim_config"]["scratch_dir"], hom_dir_name)
        #     hom_dir_abs_path = os.path.join(os.environ.get("SCRATCHDIR"), hom_dir_name)
        # else:
        #hom_dir_abs_path = os.path.join(sample_dir, hom_dir_name)

        #print("hom_dir_abs_path ", hom_dir_abs_path)

        # os.chdir(config["scratch_dir"])
        if os.path.exists(hom_dir_abs_path):
            shutil.rmtree(hom_dir_abs_path)

        os.mkdir(hom_dir_abs_path)
        os.chdir(hom_dir_abs_path)

        h_dir = os.getcwd()
        print("h_dir os.getcwd() ", os.getcwd())
        # exit()

        os.mkdir("dataset")
        dataset_path = os.path.join(h_dir, "dataset")
        sim_config = config["sim_config"]

        # print("sim config ", sim_config)
        # print("rasterize at once ", config["sim_config"]["rasterize_at_once"])

        domain_box = sim_config["geometry"]["domain_box"]
        subdomain_box = sim_config["geometry"]["subdomain_box"]
        subdomain_overlap = np.array([0, 0])  # np.array([50, 50])

        # print("sample dict ", sim_config)
        # print(sim_config["seed"])

        # work_dir = "seed_{}".format(sample_dict["seed"])

        # work_dir = os.path.join(
        #     "/home/martin/Documents/Endorse/ms-homogenization/seed_26339/aperture_10_4/test_density",
        #     "n_10_s_100_100_step_5_2")

        # work_dir = "/home/martin/Documents/Endorse/ms-homogenization/test_summary"

        work_dir = sim_config["work_dir"]

        # print("work dir ", work_dir)
        lx, ly, lz = domain_box
        print("lx: {}, lz: {}, ly: {} ".format(lx, lz, ly))

        # bottom_left_corner = [-lx / 2, -ly / 2, -lz / 2]
        # bottom_right_corner = [+lx / 2, -ly / 2]
        # top_right_corner = [+lx / 2, +ly / 2]
        # top_left_corner = [-lx / 2, +ly / 2]
        # #               bottom-left corner  bottom-right corner  top-right corner    top-left corner
        # complete_polygon = [bottom_left_corner, bottom_right_corner, top_right_corner, top_left_corner]

        # plt.scatter(*zip(*complete_polygon))
        n_subdomains_per_axes = sim_config["geometry"].get("n_subdomains_per_axes", 4)
        domain_box = sim_config["geometry"]["domain_box"]
        subdomain_box = sim_config["geometry"]["subdomain_box"]
        #lx, ly = domain_box

        print("n_subdomains_per_axes ", n_subdomains_per_axes)
        # exit()
        #n_subdomains = int(np.floor(np.sqrt(n_subdomains)))

        cond_tensors = {}
        pred_cond_tensors = {}
        percentage_sym_tn_diff = []
        time_measurements = []
        sample_center = {}

        # bulk_cond_values = []
        # bulk_cond_points = []

        fine_flow = None

        # if "nn_path" in sim_config["nn_path"]:
        #     zarr_file_path = DFMSim3D.create_zarr_file(os.getcwd(), n_samples=int(n_subdomains_per_axes ** 3), config_dict=config)
        #
        #     n_steps = config["sim_config"]["geometry"]["n_voxels"]

        k = -1
        # if "rasterize_at_once" in config["sim_config"] and config["sim_config"]["rasterize_at_once"]:
        #     fields_file = config["sim_config"]["fields_file_to_rast"] if "fields_file_to_rast" in config["sim_config"] else "fields_fine_to_rast.msh"
        #     mesh_file = config["sim_config"]["mesh_file_to_rast"] if "mesh_file_to_rast" in config["sim_config"] else "mesh_fine_to_rast.msh"
        #
        #     pred_cond_tensors = DFMSim3D.rasterize_at_once(sample_dir, dataset_path, config, n_subdomains, fractures,
        #                                                  h_dir, seed=seed, fields_file=fields_file, mesh_file=mesh_file)
        #     # print("pred cond tensors ", pred_cond_tensors)
        # else:
        #create_hom_samples_start_time = time.time()
        for i in range(n_subdomains_per_axes):
            center_x = subdomain_box[0] / 2 + (lx - subdomain_box[0]) / (n_subdomains_per_axes - 1) * i - lx / 2

            for j in range(n_subdomains_per_axes):
                #start_time = time.time()

                center_y = subdomain_box[1] / 2 + (ly - subdomain_box[1]) / (n_subdomains_per_axes - 1) * j - ly / 2

                for l in range(n_subdomains_per_axes):
                    k += 1
                    subdir_name = "i_{}_j_{}_l_{}_k_{}".format(i, j, l, k)
                    os.mkdir(subdir_name)
                    os.chdir(subdir_name)

                    hom_sample_dir = Path(os.getcwd()).absolute()

                    center_z = subdomain_box[1] / 2 + (lz - subdomain_box[1]) / (n_subdomains_per_axes - 1) * l - lz / 2

                    # center_z_2 = subdomain_box[1] / 2 + (lz - subdomain_box[1]) / (
                    #             n_subdomains_per_axes - 1) * (l+1) - lz / 2

                    #center_x = 0.0
                    #center_y = -7.5
                    #center_z = -7.5
                    #k = 10
                    #
                    print("center x:{} y:{}, z:{}, k: {}".format(center_x, center_y, center_z, k))

                    # box_size_x = subdomain_box[0]
                    # box_size_y = subdomain_box[1]
                    # box_size_z = subdomain_box[2]

                    #outer_cube = DFMSim3D.get_outer_cube(center_x, center_y, center_z, box_size_x, box_size_y, box_size_z)
                    #sim_config["geometry"]["outer_cube"] = outer_cube
                    #print("work_dir ", work_dir)

                    #print("outer polygon ", outer_polygon)
                    #print("center x:{} y:{}".format(center_x, center_y))

                    #sim_config["work_dir"] = work_dir
                    #config["homogenization"] = True

                    subdomain_box_run_samples = copy.deepcopy(subdomain_box)

                    # print("subdomain box run samples ", subdomain_box_run_samples)
                    # print("np.array(subdomain_box_run_samples)*1.1 ", np.array(subdomain_box_run_samples)*1.1)

                    cond_field_step = np.abs(bulk_cond_points[0][-1]) - np.abs(bulk_cond_points[1][-1])
                    subdomain_to_extract = np.array(subdomain_box_run_samples) * 1.1 + (2 * cond_field_step)
                    subdomain_bulk_cond_values, subdomain_bulk_cond_points = DFMSim3D.extract_subdomain(bulk_cond_values, bulk_cond_points, (center_x, center_y, center_z), subdomain_to_extract)

                    try:
                        subdomain_dfn_to_homogenize = DFMSim3D.extract_dfn(dfn_to_homogenize, dfn_to_homogenize_list, (center_x, center_y, center_z), np.array(subdomain_box_run_samples) * 1.5)
                    except:
                        subdomain_dfn_to_homogenize = dfn_to_homogenize
                    subdomain_fr_media = FracturedMedia.fracture_cond_params(subdomain_dfn_to_homogenize, 1e-4, 0.00001)

                    # subdomain_bulk_cond_values, subdomain_bulk_cond_points = bulk_cond_values, bulk_cond_points
                    # subdomain_dfn_to_homogenize = dfn_to_homogenize
                    # subdomain_fr_media = FracturedMedia.fracture_cond_params(subdomain_dfn_to_homogenize, 1e-4, 0.00001)

                    # print("subdomain_bulk_cond_values.shape ", subdomain_bulk_cond_values.shape)
                    # print("subdomain_bulk_cond_points.shape ", subdomain_bulk_cond_points.shape)

                    orig_center_x = center_x
                    orig_center_y = center_y
                    orig_center_z = center_z

                    num_iterations = 0
                    while True:
                        try:
                            # subdomain_box_run_samples[0] += subdomain_box_run_samples[0] * 0.1
                            # subdomain_box_run_samples[1] += subdomain_box_run_samples[1] * 0.1
                            # subdomain_box_run_samples[2] += subdomain_box_run_samples[2] * 0.1
                            print("center x:{} y:{}, z:{}, k: {}".format(center_x, center_y, center_z, k))
                            #print("subdomain_box_run_samples ", subdomain_box_run_samples)

                            bc_pressure_gradient = [1, 0, 0]
                            cond_file, fr_cond, fr_region_map = DFMSim3D._run_sample(bc_pressure_gradient, subdomain_fr_media, config, hom_sample_dir,
                                                                      subdomain_bulk_cond_values, subdomain_bulk_cond_points, subdomain_box_run_samples, mesh_step=config["fine"]["step"], center=[center_x, center_y, center_z])
                            flux_response_0 = DFMSim3D.get_flux_response()  # bc_pressure_gradient, fr_media, fem_grid, config, sample_dir,
                            # sim_config)

                            bc_pressure_gradient = [0, 1, 0]
                            DFMSim3D._run_sample(bc_pressure_gradient, subdomain_fr_media, config, hom_sample_dir, subdomain_bulk_cond_values,
                                                 subdomain_bulk_cond_points,
                                                 subdomain_box_run_samples, mesh_step=config["fine"]["step"], cond_file=cond_file, center=[center_x, center_y, center_z])
                            flux_response_1 = DFMSim3D.get_flux_response()  # bc_pressure_gradient, fr_media, fem_grid, config, sample_dir,
                            # sim_config)

                            bc_pressure_gradient = [0, 0, 1]
                            DFMSim3D._run_sample(bc_pressure_gradient, subdomain_fr_media, config, hom_sample_dir, subdomain_bulk_cond_values,
                                                 subdomain_bulk_cond_points,
                                                 subdomain_box_run_samples,  mesh_step=config["fine"]["step"], cond_file=cond_file, center=[center_x, center_y, center_z])

                            flux_response_2 = DFMSim3D.get_flux_response()  # bc_pressure_gradient, fr_media, fem_grid, config, sample_dir,
                            # sim_config)

                            bc_pressure_gradients = np.stack(([1, 0, 0], [0, 1, 0], [0, 0, 1]), axis=0)
                            flux_responses = np.squeeze(
                                np.stack((flux_response_0, flux_response_1, flux_response_2), axis=0))

                            # exit()
                            break

                        except Exception as msg:
                            print(msg)
                            num_iterations += 1
                            # subdomain_box_run_samples[0] += subdomain_box_run_samples[0] * 0.05
                            # subdomain_box_run_samples[1] += subdomain_box_run_samples[1] * 0.05
                            # subdomain_box_run_samples[2] += subdomain_box_run_samples[2] * 0.05
                            #
                            current_dir = os.getcwd()
                            # center_x += center_x * 0.05
                            # center_y += center_y * 0.05
                            # center_z += center_z * 0.05

                            delta = 1.0

                            domain_center = np.array([0, 0, 0])
                            p = np.array([center_x, center_y, center_z])
                            direction = p - domain_center
                            unit_dir = direction / np.linalg.norm(direction)

                            # Move slightly toward the center
                            p_nudged = p - delta * unit_dir

                            # Check if nudged point is still within domain
                            # within_domain = np.all(np.abs(p_nudged) <= domain_half)
                            center_x, center_y, center_z = p_nudged[0], p_nudged[1], p_nudged[2]

                            if num_iterations > 10:
                                break

                            # Loop through the files in the directory and delete them
                            # for filename in os.listdir(current_dir):
                            #     file_path = os.path.join(current_dir, filename)
                            #
                            #     # Check if it's a file and delete it
                            #     if os.path.isfile(file_path):
                            #         os.remove(file_path)
                            #         print(f"Deleted: {file_path}")

                    new_center_x = center_x
                    new_center_y = center_y
                    new_center_z = center_z

                    center_x = orig_center_x
                    center_y = orig_center_y
                    center_z = orig_center_z

                    if num_iterations > 10:
                        print("subdir_name {} num iterations 10".format(subdir_name))
                        # DFMSim3D._remove_files()
                        os.chdir(h_dir)
                        continue

                    equivalent_cond_tn_voigt = equivalent_posdef_tensor(np.array(bc_pressure_gradients),
                                                                        flux_responses)

                    print("equivalent_cond_tn_voigt ", equivalent_cond_tn_voigt)

                    equivalent_cond_tn = voigt_to_tn(np.array([equivalent_cond_tn_voigt]))  # np.zeros((3, 3))
                    #print("equivalent cond tn ", equivalent_cond_tn)
                    # evals, evecs = np.linalg.eigh(equivalent_cond_tn)
                    # print("evals equivalent cond tn ", evals)
                    # assert np.all(evals) > 0

                    cond_tensors[(new_center_x, new_center_y, new_center_z)] = equivalent_cond_tn

                    if "nn_path_for_block_hom" in config["sim_config"]:
                        #try:
                        fem_grid_rast = fem.fem_grid(subdomain_box[0], config["sim_config"]["geometry"]["n_voxels"], fem.Fe.Q(dim=3),
                                                     origin=[new_center_x - subdomain_box[0] / 2,
                                                             new_center_y - subdomain_box[0] / 2,
                                                             new_center_z - subdomain_box[0] / 2])

                        equivalent_cond_tn_predictions = DFMSim3D.predict_on_hom_sample(config,
                                                                                        subdomain_bulk_cond_values,
                                                                                        subdomain_bulk_cond_points,
                                                                                        subdomain_dfn_to_homogenize,
                                                                                        fr_cond,
                                                                                        fem_grid_rast, (
                                                                                        new_center_x, new_center_y,
                                                                                        new_center_z))

                        print("equivalent_cond_tn_predictions ", equivalent_cond_tn_predictions)

                        equivalent_cond_tn_predictions = voigt_to_tn(np.array([equivalent_cond_tn_predictions]))

                        pred_cond_tensors[(new_center_x, new_center_y, new_center_z)] = equivalent_cond_tn_predictions
                        #except:
                        #    DFMSim3D._remove_files()
                        #    os.chdir(h_dir)
                        #    continue


                    # print("equivalent_cond_tn ", tn_to_voigt(equivalent_cond_tn))
                    # print("equivalent_cond_tn_predictions ", equivalent_cond_tn_predictions)
                    # exit()

                    #print("fem_grid_rast.grid ", fem_grid_rast.grid)
                    #print("fem_grid_rast.grid center ", fem_grid_rast.grid.grid_center())

                    # bulk_cond_fem_rast = DFMSim3D._bulk_cond_to_rast_grid(bulk_cond_values, bulk_cond_points,
                    #                                                       fem_grid_rast.grid)

                    #rasterized_input = DFMSim3D.rasterize(fem_grid_rast, dfn, bulk_cond=bulk_cond_fem_rast,
                    #                                      fr_cond=fr_cond)

                    #rasterized_input_voigt = tn_to_voigt(rasterized_input)
                    #rasterized_input_voigt = rasterized_input_voigt.reshape(*config["sim_config"]["geometry"]["n_voxels"],
                    #                                                        rasterized_input_voigt.shape[-1]).T

                    # fine_res = np.squeeze(equivalent_cond_tn_voigt)
                    #
                    # DFMSim3D.rasterize_save_to_zarr(zarr_file_path, config, k, fine_res, bulk_cond_values,
                    #                                 bulk_cond_points, dfn, fr_cond,
                    #                                 fem_grid_rast, n_steps=config["sim_config"]["geometry"]["n_voxels"])

                    #np.save("equivalent_cond_tn", equivalent_cond_tn)
                    #np.savez_compressed("rasterized_input_voigt ", rasterized_input_voigt)

                    # if os.path.exists(zarr_file_path):
                    #     zarr_file = zarr.open(zarr_file_path, mode='r+')
                    #
                    #     zarr_file["inputs"][k, ...] = rasterized_input_voigt
                    #     zarr_file["outputs"][k, :] = fine_res

                    DFMSim3D._remove_files()

                    os.chdir(h_dir)

        try:
            shutil.move(h_dir, sample_dir)
        except:
            pass

        os.chdir(sample_dir)

        return cond_tensors, pred_cond_tensors

    @staticmethod
    def _remove_files():
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

    @staticmethod
    def calculate_P32(P30, size_range=(1, 564), exp=2.5, p32_exp=2.5, shape_area=1):
        """
        Calculate P32 given a P30 value using the formula derived from the integral relations.

        Parameters:
        - P30: Value of P30 (fracture intensity for P30)
        - size_range: Tuple defining the size range (a, b)
        - exp: The exponent used in P30 calculation
        - p32_exp: The exponent used in P32 calculation
        - shape_area: The shape area

        Returns:
        - P32: Corresponding value for P32
        """

        a, b = size_range
        # Calculate integral area and intensity for P32
        integral_area = (b ** (2 - p32_exp) - a ** (2 - p32_exp)) / (2 - p32_exp)
        integral_intensity = (b ** (-exp) - a ** (-exp)) / -exp

        # Reversing the relationship to get P32 from P30
        P32 = P30 * integral_area * shape_area / integral_intensity

        return P32


    @staticmethod
    def fracture_random_set(seed, size_range, work_dir, max_frac=1e21):
        #script_dir = Path(__file__).absolute().parent
        rmin, rmax = size_range
        box_dimensions = (rmax, rmax, rmax)
        fr_cfg_path = os.path.join(work_dir, "fractures_conf.yaml")

        # with open() as f:
        #    pop_cfg = yaml.load(f, Loader=yaml.SafeLoader)
        fr_pop = stochastic.Population.from_cfg(fr_cfg_path, box_dimensions, shape=stochastic.EllipseShape())
        if fr_pop.mean_size() > max_frac:
            #common_range, intensities = fr_pop.common_range_for_sample_size(sample_size=max_frac)
            common_range = fr_pop.common_range_for_sample_size(sample_size=max_frac)
            fr_pop = fr_pop.set_sample_range(common_range)
        print(f"fr set range: {[rmin, rmax]}, fr_lim: {max_frac}, mean population size: {fr_pop.mean_size()}")

        pos_gen = stochastic.UniformBoxPosition(fr_pop.domain)
        np.random.seed(seed)
        fractures = fr_pop.sample(pos_distr=pos_gen, keep_nonempty=True)

        # new_p32 = []
        # for idx, new_intensity in enumerate(intensities):
        #     # print("fr_pop.families[idx] ", fr_pop.families[idx].size.power)
        #     # exit()
        #     new_p32.append(DFMSim3D.calculate_P32(new_intensity, size_range, exp=fr_pop.families[idx].size.power,
        #                                           p32_exp=fr_pop.families[idx].size.power))

        # print("intensities ", intensities)
        # print("new p32 ", new_p32)
        # print("np.sum new p32 ", np.sum(new_p32))
        #
        # print("SUM P_30 ", np.sum(intensities))
        #
        # exit()


        # for fr in fractures:
        #    fr.region = gmsh.Region.get("fr", 2)
        return fractures


    # @staticmethod
    # def homo_decovalex(fr_media: FracturedMedia, grid: fem.Grid):
    #     """
    #     Homogenize fr_media to the conductivity tensor field on grid.
    #     :return: conductivity_field, np.array, shape (n_elements, n_voight)
    #     """
    #     ellipses = [dmap.Ellipse(fr.normal, fr.center, fr.scale) for fr in fr_media.dfn]
    #     d_grid = dmap.Grid.make_grid(grid.origin, grid.step, grid.dimensions)
    #     fractures = dmap.map_dfn(d_grid, ellipses)
    #     fr_transmissivity = fr_media.fr_conductivity * fr_media.fr_cross_section
    #     k_iso_zyx = dmap.permIso(d_grid, fractures, fr_transmissivity, fr_media.conductivity)
    #     k_iso_xyz = grid.cell_field_C_like(k_iso_zyx)
    #     k_voigt = k_iso_xyz[:, None] * np.array([1, 1, 1, 0, 0, 0])[None, :]
    #     return k_voigt

    @staticmethod
    def create_fractures_rectangles(gmsh_geom, fractures: FrozenSet, base_shape: gmsh.ObjectSet,
                                    shift=np.array([0, 0, 0])):
        """

        :param gmsh_geom:
        :param fractures:
        :param base_shape:
        :param shift:
        :return:
        """
        # From given fracture date list 'fractures'.
        # transform the base_shape to fracture objects
        # fragment fractures by their intersections
        # return dict: fracture.region -> GMSHobject with corresponding fracture fragments
        if len(fractures) == 0:
            return [], []

        shapes = []
        region_map = {}
        for i, fr in enumerate(fractures):
            shape = base_shape.copy()

            #print("fr: ", i, "tag: ", shape.dim_tags, "fr.r: ", fr.r, "fr.rx: ", fr.rx, "fr.ry: ", fr.ry)
            region_name = f"fam_{fr.family}_{i:03d}"
            shape = shape.scale([fr.rx, fr.ry, 1]) \
                .rotate(axis=[0, 0, 1], angle=fr.shape_angle) \
                .rotate(axis=fr.rotation_axis, angle=fr.rotation_angle) \
                .translate(fr.center + shift) \
                .set_region(region_name)

            region_map[region_name] = i
            shapes.append(shape)

        fracture_fragments = gmsh_geom.fragment(*shapes)
        return fracture_fragments, region_map

    @staticmethod
    def box_with_sides(factory, dimensions, center):
        """
        # code from: https://github.com/GeoMop/endorse/blob/74af18831c61be3397068e015786c95ced3a3f87/src/endorse/mesh/mesh_tools.py#L10
        Make a box and dictionary of its sides named: 'side_[xyz][01]'
        :return: box, sides_dict
        """
        box = factory.box(dimensions, center).set_region("box")
        side_z = factory.rectangle([dimensions[0], dimensions[1]])
        side_y = factory.rectangle([dimensions[0], dimensions[2]])
        side_x = factory.rectangle([dimensions[2], dimensions[1]])
        sides = dict(
            side_z0=side_z.copy().translate([0, 0, -dimensions[2] / 2]),
            side_z1=side_z.copy().translate([0, 0, +dimensions[2] / 2]),
            side_y0=side_y.copy().translate([0, 0, -dimensions[1] / 2]).rotate([-1, 0, 0], np.pi / 2),
            side_y1=side_y.copy().translate([0, 0, +dimensions[1] / 2]).rotate([-1, 0, 0], np.pi / 2),
            side_x0=side_x.copy().translate([0, 0, -dimensions[0] / 2]).rotate([0, 1, 0], np.pi / 2),
            side_x1=side_x.copy().translate([0, 0, +dimensions[0] / 2]).rotate([0, 1, 0], np.pi / 2)
        )
        for name, side in sides.items():
            side.modify_regions(name)
        return box, sides

    @staticmethod
    def fragment(factory, object_dict, boundary_object_dict):
        """
        # FROM: https://github.com/GeoMop/endorse/blob/74af18831c61be3397068e015786c95ced3a3f87/apps/Chodby_trans/mesh/create_mesh.py#L122
        :param factory: bgem gmsh factory
        :param object_dict: "volumetric" objects (of arbitrary dimensions)
        :param boundary_object_dict: "boundary" objects to determine boundary regions later
        :return:
        """
        # Merge the dictionaries (order preserved since Python 3.7+)
        merged_inputs = {**object_dict, **boundary_object_dict}

        # determine ObjectSets which are not None
        enabled_keys = [key for key, value in merged_inputs.items() if value is not None]
        # create list of ObjectSets for fragmentation
        input_list = [merged_inputs[key] for key in enabled_keys]

        output_list = factory.fragment(*input_list)

        # Map outputs to "_fr" keys
        fragmented_all = {f"{key}_fr": val for key, val in zip(enabled_keys, output_list)}

        # Split back into two output dictionaries
        fragmented_1 = {f"{key}_fr": fragmented_all[f"{key}_fr"]
                        for key in object_dict if f"{key}_fr" in fragmented_all}

        fragmented_2 = {f"{key}_fr": fragmented_all[f"{key}_fr"]
                        for key in boundary_object_dict if f"{key}_fr" in fragmented_all}

        return fragmented_1, fragmented_2

    @staticmethod
    def ref_solution_mesh_outflow_problem(work_dir, domain_dimensions, fractures, fr_step, bulk_step, center):
        # inspired by https://github.com/GeoMop/endorse/blob/74af18831c61be3397068e015786c95ced3a3f87/apps/Chodby_trans/mesh/create_mesh.py#L227
        print("MESH OUTFLOW PROBLEM")
        # Prepare objects
        vol_dict = {
            "box": None,
            "fractures_group": None,
        }
        bnd_dict = {}

        factory = gmsh.GeometryOCC("homo_cube", verbose=False)
        gopt = options.Geometry()
        gopt.Tolerance = 0.0001
        gopt.ToleranceBoolean = 0.001

        #box = factory.box(domain_dimensions, center)
        #print("orig box ", box)

        box, box_sides_dict = DFMSim3D.box_with_sides(factory, domain_dimensions, center)
        #print("box ", box)
        #print("box_sides_dict ", box_sides_dict)

        box_sides = {}
        for side_name, side in box_sides_dict.items():
            box_sides[side_name] = box_sides_dict[side_name].copy()

        #bnd_dict["box_side_x0"] = factory.group(box_sides_dict["side_x0"]).copy()
        #bnd_dict["box_side_x1"] = factory.group(box_sides_dict["side_x1"]).copy()

        #box_sides_dict_copy = copy.deepcopy(box_sides_dict)
        bnd_dict = {**bnd_dict, **box_sides_dict}

        vol_dict["box"] = box.copy()
        vol_dict["box"].set_region("box")

        ##########
        ## Add fractures
        ######
        fractures, fr_region_map = DFMSim3D.create_fractures_rectangles(factory, fractures, factory.rectangle())
        #fractures_group = factory.group(*fractures).intersect(box)

        vol_dict["fractures_group"] = factory.group(*fractures).intersect(vol_dict["box"])
        #vol_dict["fractures_group"].set_region("fractures").mesh_step(fr_step)

        #box_fr, fractures_fr = factory.fragment(box, fractures_group)
        fr_dict, fr_bnd_dict = DFMSim3D.fragment(factory, vol_dict, bnd_dict)

        #[print(k, v) for k, v in fr_dict.items()]
        #[print(k, v) for k, v in fr_bnd_dict.items()]

        geometry_set = list(fr_dict.values())

        #print("box_sides_dict 2", box_sides_dict)

        for side_name, side in box_sides.items():
            fr_bnd_dict[side_name + "_fr"] \
                .set_region('.' + side_name) \
                .mesh_step(bulk_step)
            geometry_set.append(fr_bnd_dict[side_name + "_fr"])

            b_fractures_fr = fr_dict.get("fractures_group_fr").get_boundary().split_by_dimension()[1]
            b_fractures_side_intersect = b_fractures_fr.select_by_intersect(side.copy())
            if b_fractures_side_intersect is not None:
                b_fractures_out_side = b_fractures_side_intersect.set_region(".fractures_{}".format(side_name)) \
                    .mesh_step(bulk_step)
                geometry_set.append(b_fractures_out_side)

        # ###
        # ## Intersection - fractures with left side
        # ###
        # b_fractures_fr = fr_dict.get("fractures_group_fr").get_boundary().split_by_dimension()[1]
        # b_fractures_sidex0_intersect = b_fractures_fr.select_by_intersect(side_x0.copy())
        # if b_fractures_sidex0_intersect is not None:
        #     b_fractures_out_side_x0 = b_fractures_sidex0_intersect.set_region(".fractures_side_x0") \
        #     .mesh_step(bulk_step)
        #     geometry_set.append(b_fractures_out_side_x0)
        #
        # ###
        # ## Intersection - fractures with left side
        # ###
        # b_fractures_fr = fr_dict.get("fractures_group_fr").get_boundary().split_by_dimension()[1]
        # b_fractures_sidex0_intersect = b_fractures_fr.select_by_intersect(side_y0.copy())
        # if b_fractures_sidex0_intersect is not None:
        #     b_fractures_out_side_x0 = b_fractures_sidex0_intersect.set_region(".fractures_side_y0") \
        #         .mesh_step(bulk_step)
        #     geometry_set.append(b_fractures_out_side_x0)
        #
        # ###
        # ## Intersection - fractures with left side
        # ###
        # b_fractures_fr = fr_dict.get("fractures_group_fr").get_boundary().split_by_dimension()[1]
        # b_fractures_sidex0_intersect = b_fractures_fr.select_by_intersect(side_z0.copy())
        # if b_fractures_sidex0_intersect is not None:
        #     b_fractures_out_side_x0 = b_fractures_sidex0_intersect.set_region(".fractures_side_z0") \
        #         .mesh_step(bulk_step)
        #     geometry_set.append(b_fractures_out_side_x0)
        #
        # ###
        # ## Intersection - fractures with right side
        # ###
        # b_fractures_fr = fr_dict.get("fractures_group_fr").get_boundary().split_by_dimension()[1]
        # b_fractures_sidex1_intersect = b_fractures_fr.select_by_intersect(side_x1.copy())
        # if b_fractures_sidex1_intersect is not None:
        #     b_fractures_out_side_x1 = b_fractures_sidex1_intersect.set_region(".fractures_side_x1") \
        #         .mesh_step(bulk_step)
        #     geometry_set.append(b_fractures_out_side_x1)
        #
        # ###
        # ## Intersection - fractures with right side
        # ###
        # b_fractures_fr = fr_dict.get("fractures_group_fr").get_boundary().split_by_dimension()[1]
        # b_fractures_sidex1_intersect = b_fractures_fr.select_by_intersect(side_y1.copy())
        # if b_fractures_sidex1_intersect is not None:
        #     b_fractures_out_side_x1 = b_fractures_sidex1_intersect.set_region(".fractures_side_y1") \
        #         .mesh_step(bulk_step)
        #     geometry_set.append(b_fractures_out_side_x1)
        #
        # ###
        # ## Intersection - fractures with right side
        # ###
        # b_fractures_fr = fr_dict.get("fractures_group_fr").get_boundary().split_by_dimension()[1]
        # b_fractures_sidex1_intersect = b_fractures_fr.select_by_intersect(side_z1.copy())
        # if b_fractures_sidex1_intersect is not None:
        #     b_fractures_out_side_x1 = b_fractures_sidex1_intersect.set_region(".fractures_side_z1") \
        #         .mesh_step(bulk_step)
        #     geometry_set.append(b_fractures_out_side_x1)

        geometry_final = factory.group(*geometry_set)

        #fractures_fr.mesh_step(fr_step)  # .set_region("fractures")
        #objects = [box_fr, fractures_fr]
        factory.write_brep(str(factory.model_name))
        # factory.mesh_options.CharacteristicLengthMin = bulk_step #cfg.get("min_mesh_step", cfg.boreholes_mesh_step)
        factory.mesh_options.CharacteristicLengthMax = bulk_step
        # factory.mesh_options.Algorithm = options.Algorithm3d.MMG3D

        # mesh.Algorithm = options.Algorithm2d.MeshAdapt # produce some degenerated 2d elements on fracture boundaries ??
        # mesh.Algorithm = options.Algorithm2d.Delaunay
        # mesh.Algorithm = options.Algorithm2d.FrontalDelaunay

        factory.mesh_options.Algorithm = options.Algorithm3d.Delaunay
        # mesh.ToleranceInitialDelaunay = 0.01
        # mesh.ToleranceEdgeLength = fracture_mesh_step / 5
        # mesh.CharacteristicLengthFromPoints = True
        # factory.mesh_options.CharacteristicLengthFromCurvature = False
        # factory.mesh_options.CharacteristicLengthExtendFromBoundary = 2  # co se stane if 1
        # mesh.CharacteristicLengthMin = min_el_size
        # mesh.CharacteristicLengthMax = max_el_size

        # factory.keep_only(*objects)
        # factory.remove_duplicate_entities()
        factory.make_mesh([geometry_final], dim=3)
        # factory.write_mesh(me gmsh.MeshFormat.msh2) # unfortunately GMSH only write in version 2 format for the extension 'msh2'
        f_name = factory.model_name + ".msh2"

        mesh_file = work_dir / (factory.model_name + ".msh2")

        # print("mesh file name ", mesh_file)
        factory.write_mesh(str(mesh_file), format=gmsh.MeshFormat.msh2)
        return mesh_file, fr_region_map

    @staticmethod
    def ref_solution_mesh(work_dir, domain_dimensions, fractures, fr_step, bulk_step, center):
        factory = gmsh.GeometryOCC("homo_cube", verbose=False)
        gopt = options.Geometry()
        gopt.Tolerance = 0.0001
        gopt.ToleranceBoolean = 0.001

        box = factory.box(domain_dimensions, center)
        print("orig box ", box)

        fractures, fr_region_map = DFMSim3D.create_fractures_rectangles(factory, fractures, factory.rectangle())

        fractures_group = factory.group(*fractures).intersect(box)
        box_fr, fractures_fr = factory.fragment(box, fractures_group)

        fractures_fr.mesh_step(fr_step)  # .set_region("fractures")
        objects = [box_fr, fractures_fr]
        factory.write_brep(str(factory.model_name))
        #factory.mesh_options.CharacteristicLengthMin = bulk_step #cfg.get("min_mesh_step", cfg.boreholes_mesh_step)
        factory.mesh_options.CharacteristicLengthMax = bulk_step
        # factory.mesh_options.Algorithm = options.Algorithm3d.MMG3D

        # mesh.Algorithm = options.Algorithm2d.MeshAdapt # produce some degenerated 2d elements on fracture boundaries ??
        # mesh.Algorithm = options.Algorithm2d.Delaunay
        # mesh.Algorithm = options.Algorithm2d.FrontalDelaunay

        factory.mesh_options.Algorithm = options.Algorithm3d.Delaunay
        # mesh.ToleranceInitialDelaunay = 0.01
        # mesh.ToleranceEdgeLength = fracture_mesh_step / 5
        # mesh.CharacteristicLengthFromPoints = True
        # factory.mesh_options.CharacteristicLengthFromCurvature = False
        # factory.mesh_options.CharacteristicLengthExtendFromBoundary = 2  # co se stane if 1
        # mesh.CharacteristicLengthMin = min_el_size
        # mesh.CharacteristicLengthMax = max_el_size

        # factory.keep_only(*objects)
        # factory.remove_duplicate_entities()
        factory.make_mesh(objects, dim=3)
        # factory.write_mesh(me gmsh.MeshFormat.msh2) # unfortunately GMSH only write in version 2 format for the extension 'msh2'
        f_name = factory.model_name + ".msh2"

        mesh_file = work_dir / (factory.model_name + ".msh2")

        #print("mesh file name ", mesh_file)
        factory.write_mesh(str(mesh_file), format=gmsh.MeshFormat.msh2)
        return mesh_file, fr_region_map

    @staticmethod
    def fr_cross_section(fractures, cross_to_r):
        return [cross_to_r * fr.r for fr in fractures]

    @staticmethod
    def fr_field(mesh, dfn, reg_id_to_fr, fr_values, bulk_value, rnd_cond=False, field_dim=3):
        """
        Provide implicit fields on fractures as input.
        :param mesh:
        :param fractures:
        :param fr_values:
        :param bulk_value:
        :return:
        """
        fr_map = mesh.fr_map(dfn,
                             reg_id_to_fr)  # np.array of fracture indices of elements, n_frac for nonfracture elements

        if field_dim == 3:
            bulk_cond_tn = np.eye(3)*bulk_value
            bulk_cond_tn = np.expand_dims(bulk_cond_tn, axis=0)

            if len(fr_values) == 0:
                fr_values_ = bulk_cond_tn

            elif rnd_cond:
                pass
            else:
                fr_values_ = np.concatenate((
                    fr_values,
                    bulk_cond_tn), axis=0)
        elif field_dim == 1:
            fr_values_ = np.concatenate((
                np.array(fr_values),
                np.atleast_1d(bulk_value)))

        field_vals = fr_values_[fr_map]
        return field_vals, fr_map

    # @staticmethod
    # def reference_solution(fr_media: FracturedMedia, dimensions, bc_gradients, mesh_step, sample_dir, work_dir):
    #     dfn = fr_media.dfn
    #     bulk_conductivity = fr_media.conductivity
    #
    #     # Input crssection and conductivity
    #     print("mesh step ", mesh_step)
    #     mesh_file, fr_region_map = DFMSim3D.ref_solution_mesh(sample_dir, dimensions, dfn, fr_step=mesh_step, bulk_step=mesh_step)
    #     print("mesh file ", mesh_file)
    #     full_mesh = Mesh.load_mesh(mesh_file, heal_tol=0.001)  # gamma
    #
    #     fr_cond, fr_map = DFMSim3D.fr_field(full_mesh, dfn, fr_region_map, fr_media.fr_conductivity, bulk_conductivity,
    #                       rnd_cond=False, field_dim=1)
    #
    #     fr_cross_section, fr_map = DFMSim3D.fr_field(full_mesh, dfn, fr_region_map, fr_media.fr_cross_section, 1.0, field_dim=1)
    #
    #     print("fr cond ", fr_cond)
    #     #print("fr el ids ", fr_el_ids)
    #
    #     fields = dict(
    #         conductivity=fr_cond,
    #         cross_section=fr_cross_section)
    #
    #     cond_file = full_mesh.write_fields(str(sample_dir / "input_fields.msh2"), fields)
    #     print("cond_file ", cond_file)
    #     cond_file = Path(cond_file)
    #     cond_file = cond_file.rename(cond_file.with_suffix(".msh"))
    #
    #     print("final cond file ", cond_file)
    #
    #     flow_cfg = dotdict(
    #         flow_executable=[
    #             "docker",
    #             "run",
    #             "-v",
    #             "{}:{}".format(os.getcwd(), os.getcwd()),
    #             "flow123d/ci-gnu:4.0.0a01_94c428",
    #             "flow123d",
    #             #        - flow123d/endorse_ci:a785dd
    #             #        - flow123d/ci-gnu:4.0.0a_d61969
    #             # "dbg",
    #             # "run",
    #             "--output_dir",
    #             os.getcwd()
    #         ],
    #         mesh_file=cond_file,
    #         pressure_grad=bc_gradients,
    #         # pressure_grad_0=bc_gradients[0],
    #         # pressure_grad_1=bc_gradients[1],
    #         # pressure_grad_2=bc_gradients[2]
    #     )
    #     f_template = "flow_upscale_templ.yaml"
    #     shutil.copy(os.path.join(work_dir, f_template), sample_dir)
    #     with workdir_mng(sample_dir):
    #         flow_out = call_flow(flow_cfg, f_template, flow_cfg)
    #
    #     # Project to target grid
    #     #print(flow_out)
    #     # vel_p0 = velocity_p0(target_grid, flow_out)
    #     # projection of fields
    #     return flow_out

    _rel_corner = np.array([[0, 0, 0], [1, 0, 0],
                             [1, 1, 0], [0, 1, 0],
                             [0, 0, 1], [1, 0, 1],
                             [1, 1, 1], [0, 1, 1]])


    # Define transformation matrices and index mappings for 2D and 3D refinements
    _transformation_matrices = {
        3: np.array([
            [1, 0, 0],  # Vertex 0
            [0, 1, 0],  # Vertex 1
            [0, 0, 1],  # Vertex 2
            [0.5, 0.5, 0],  # Midpoint between vertices 0 and 1
            [0, 0.5, 0.5],  # Midpoint between vertices 1 and 2
            [0.5, 0, 0.5],  # Midpoint between vertices 0 and 2
        ]),
        4: np.array([
            [1, 0, 0, 0],  # Vertex 0
            [0, 1, 0, 0],  # Vertex 1
            [0, 0, 1, 0],  # Vertex 2
            [0, 0, 0, 1],  # Vertex 3
            [0.5, 0.5, 0, 0],  # Midpoint between vertices 0 and 1
            [0.5, 0, 0.5, 0],  # Midpoint between vertices 0 and 2
            [0.5, 0, 0, 0.5],  # Midpoint between vertices 0 and 3
            [0, 0.5, 0.5, 0],  # Midpoint between vertices 1 and 2
            [0, 0.5, 0, 0.5],  # Midpoint between vertices 1 and 3
            [0, 0, 0.5, 0.5],  # Midpoint between vertices 2 and 3
        ])
    }

    _index_maps = {
        3: np.array([
            [0, 3, 5],  # Triangle 1
            [3, 1, 4],  # Triangle 2
            [3, 4, 5],  # Triangle 3
            [5, 4, 2]  # Triangle 4
        ]),
        4: np.array([
            [0, 4, 5, 6],  # Tetrahedron 1
            [1, 4, 7, 8],  # Tetrahedron 2
            [2, 5, 7, 9],  # Tetrahedron 3
            [3, 6, 8, 9],  # Tetrahedron 4
            [4, 5, 6, 7],  # Center tetrahedron 1
            [4, 7, 8, 6],  # Center tetrahedron 2
            [5, 7, 9, 6],  # Center tetrahedron 3
            [6, 8, 9, 7],  # Center tetrahedron 4
        ])
    }

    @staticmethod
    def refine_element(element, level):
        """
        Recursively refines an element (triangle or tetrahedron) in space using matrix multiplication.

        :param element: A numpy array of shape (1, N, M), where N is the number of vertices (3 or 4).
        :param level: Integer, the level of refinement.
        :return: A numpy array containing the vertices of all refined elements.
        """
        if level == 0:
            return element
        n_tria, num_vertices, dim = element.shape
        assert n_tria == 1
        assert num_vertices == dim + 1
        transformation_matrix = DFMSim3D._transformation_matrices[num_vertices]
        index_map = DFMSim3D._index_maps[num_vertices]
        # Generate all nodes by applying the transformation matrix to the original vertices
        nodes = np.dot(transformation_matrix, element[0])
        # Construct new elements using advanced indexing
        new_elements = nodes[index_map]
        # Recursively refine each smaller element
        result = np.vstack([
            DFMSim3D.refine_element(new_elem[None, :, :], level - 1) for new_elem in new_elements
        ])
        return result

    @staticmethod
    def refine_barycenters(element, level):
        """
        Produce refinement of given element (triangle or tetrahedra), shape (N, n_vertices, 3)
        and return barycenters of refined subelements.
        """
        return np.mean(DFMSim3D.refine_element(element, level), axis=1)

    @staticmethod
    def project_adaptive_source_quad(flow_out, grid: fem.Grid):
        grid_cell_volume = np.prod(grid.step) / 27

        ref_el_2d = np.array([(0, 0), (1, 0), (0, 1)])
        ref_el_3d = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])

        pvd_content = pv.get_reader(flow_out.hydro.spatial_file.path)
        pvd_content.set_active_time_point(0)
        dataset = pvd_content.read()[0]  # Take first block of the Multiblock dataset

        velocities = dataset.cell_data['velocity_p0']
        cross_section = dataset.cell_data['cross_section']

        p_dataset = dataset.cell_data_to_point_data()
        p_dataset.point_data['velocity_magnitude'] = np.linalg.norm(p_dataset.point_data['velocity_p0'], axis=1)
        plane = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1))
        cut_dataset = p_dataset.clip_surface(plane)

        plotter = pv.Plotter()
        plotter.add_mesh(p_dataset, color='white', opacity=0.3, label='Original Dataset')
        plotter.add_mesh(cut_dataset, scalars='velocity_magnitude', cmap='viridis', label='Velocity Magnitude')

        # Add legend and show the plot
        plotter.add_scalar_bar(title='Velocity Magnitude')
        plotter.add_legend()
        plotter.show()

        # num_cells = dataset.n_cells
        # shifts = np.zeros((num_cells, 3))
        # transform_matrices = np.zeros((num_cells, 3, 3))
        # volumes = np.zeros(num_cells)

        weights_sum = np.zeros((grid.n_elements,))
        grid_velocities = np.zeros((grid.n_elements, 3))
        levels = np.zeros(dataset.n_cells, dtype=np.int32)
        # Loop through each cell
        for i in range(dataset.n_cells):
            cell = dataset.extract_cells(i)
            points = cell.points

            if len(points) < 3:
                continue  # Skip cells with less than 3 vertices

            # Shift: the first vertex of the cell
            shift = points[0]
            # shifts[i] = shift

            transform_matrix = points[1:] - shift
            if len(points) == 4:  # Tetrahedron
                # For a tetrahedron, we use all three vectors formed from the first vertex
                # transform_matrices[i] = transform_matrix[:3].T
                # Volume calculation for a tetrahedron:
                volume = np.abs(np.linalg.det(transform_matrix[:3])) / 6
                ref_el = ref_el_3d
            elif len(points) == 3:  # Triangle
                # For a triangle, we use only two vectors
                # transform_matrices[i, :2] = transform_matrix.T
                # Area calculation for a triangle:
                volume = 0.5 * np.linalg.norm(np.cross(transform_matrix[0], transform_matrix[1])) * cross_section[i]
                ref_el = ref_el_2d
            level = max(int(np.log2(volume / grid_cell_volume) / 3.0), 0)
            levels[i] = level
            ref_barycenters = DFMSim3D.refine_barycenters(ref_el[None, :, :], level)
            barycenters = shift[None, :] + ref_barycenters @ transform_matrix
            grid_indices = grid.project_points(barycenters)
            weights_sum[grid_indices] += volume
            grid_velocities[grid_indices] += volume * velocities[i]
        print(np.bincount(levels))
        grid_velocities = grid_velocities / weights_sum[:, None]
        return grid_velocities

    @staticmethod
    def element_volume(mesh, nodes):
        nodes = np.array([mesh.nodes[nid] for nid in nodes])

        if len(nodes) == 1:
            return 0
        elif len(nodes) == 2:
            # return np.linalg.norm((nodes[1] - nodes[0])/11.11)
            return np.linalg.norm(nodes[1] - nodes[0])
        elif len(nodes) == 3:
            # print("nodes ", nodes)
            # print("volume ", np.linalg.norm(np.cross(nodes[1] - nodes[0], nodes[2] - nodes[0])))
            # return np.linalg.norm(np.cross((nodes[1] - nodes[0])/11.11, (nodes[2] - nodes[0])/11.11))
            return np.linalg.norm(np.cross(nodes[1] - nodes[0], nodes[2] - nodes[0]))
        elif len(nodes) == 4:

            #print("nodes ", nodes)

            AB = nodes[1] - nodes[0]
            AC = nodes[2] - nodes[0]
            AD = nodes[3] - nodes[0]

            # Compute the cross product of vectors AC and AD
            cross_product = np.cross(AC, AD)

            # Compute the dot product of AB with the cross product
            dot_product = np.dot(AB, cross_product)

            # The volume is 1/6 of the absolute value of the dot product
            volume = abs(dot_product) / 6.0

            #print("volume ", volume)

            return volume

        else:
            assert False

    @staticmethod
    def get_outflow(sample_dir):
        # extract the flux
        balance_file = os.path.join(sample_dir, "water_balance.yaml")

        #("balance file ", balance_file)

        #balance_file = "/home/martin/Documents/MLMC-DFM/test/01_cond_field/test.yaml"
        # try:
        #
        # except yaml.YAMLError as e:
        #     print(f"YAML parsing error: {e}")
        # except Exception as e:
        #     print(f"Unexpected error: {e}")

        with open(balance_file, "r") as f:
            balance = yaml.safe_load(f)

        flux_regions = ['.side_x1', '.fr_side_x1']
        total_flux = 0.0
        found = False

        for flux_item in balance['data']:
            if flux_item['time'] > 0:
                break

            if flux_item['region'] in flux_regions:
                flux = float(flux_item['data'][0])
                flux_in = float(flux_item['data'][1])
                flux_out = float(flux_item['data'][2])
                # if flux_in > 1e-10:
                #    raise Exception("Possitive inflow at outlet region.")
                total_flux += flux_out  # flux field
                found = True

        # Get flow123d computing time
        # run_time = FlowSim.get_run_time(sample_dir)
        print("found ", found)

        if not found:
            raise Exception
        return np.array([-total_flux])


    @staticmethod
    def get_flux_response():#bc_pressure_gradients, fr_media, fem_grid, config, sample_dir, sim_config):
        # print("pressure shape ", pressure.shape)
        # flow_out = DFMSim3D.reference_solution(fr_media, fem_grid.grid.dimensions, bc_pressure_gradients,
        #                                        mesh_step=config["fine"]["step"],
        #                                        sample_dir=sample_dir,
        #                                        work_dir=sim_config["work_dir"])
        # #project_fn = DFMSim3D.project_adaptive_source_quad

        out_mesh = gmsh_io.GmshIO()

        with open("flow_fields.msh", "r") as f:
            out_mesh.read(f)
        time_idx = 0
        time, field_cs = out_mesh.element_data['cross_section'][time_idx]

        ele_reg_vol = {eid: (tags[0] - 10000, DFMSim3D.element_volume(out_mesh, nodes))
                       for eid, (tele, tags, nodes) in out_mesh.elements.items()}

        #print("ele reg vol ", ele_reg_vol)

        velocity_field = out_mesh.element_data['velocity_p0']

        # group_idx = {group_id: i_group for i_group, group_id in enumerate(set(bulk_regions.values()))}
        # n_groups = len(group_idx)
        n_groups = 1

        n_directions = 1 #len(bc_pressure_gradients)  # len(loads)
        flux_response = np.zeros((n_groups, n_directions, 3))
        #print("flux response shape ", flux_response.shape)
        total_volume = np.zeros((n_groups, n_directions))
        print("Averaging velocities ...")
        for i_time, (time, velocity) in velocity_field.items():
            i_time = int(i_time)
            #print("i_time ", i_time)

            for eid, ele_vel in velocity.items():
                # print("i_time ", i_time)
                # print("ele_vel ", ele_vel)
                # print("eid : {}, ele_vel: {}".format(eid, ele_vel))
                reg_id, vol = ele_reg_vol[eid]
                cs = field_cs[eid][0]
                # print("vol: {}, cs: {}, ele vel: {}".format(vol, cs, np.array(ele_vel[0:2])))
                volume = cs * vol
                i_group = 0  # group_idx[bulk_regions[reg_id]]
                # print("np.array(ele_vel[0:2]) ", (volume * np.array(ele_vel[0:3])))
                # print("flux_response[i_group, i_time, :] ", flux_response[i_group, i_time, :])
                flux_response[i_group, i_time, :] += -(volume * np.array(ele_vel[0:3]))
                # neg_pressure = np.matmul(np.linalg.inv(cond_tn), velocities[e_id])
                total_volume[i_group, i_time] += volume

        flux_response /= total_volume  # [:, :, None]
        flux_response = np.squeeze(flux_response, axis=1).transpose(1, 0)

        return flux_response

    @staticmethod
    def plot_isec_fields2(isec: Intersection, in_field, out_field, outpath: Path):
        """
        Assume common grid
        :param intersections:
        :return:
        """
        grid = isec.grid
        cell_fields = {
            'cell_field': isec.cell_field(),
            'in_field': in_field,
            'out_field': out_field}

        pv_grid = fem_plot.grid_fields_vtk(grid, cell_fields, vtk_path=outpath)

        # plotter = fem_plot.create_plotter()  # off_screen=True, window_size=(1024, 768))
        # plotter.add_mesh(pv_grid, scalars='cell_field')
        # plotter.show()
    @staticmethod
    def is_point_inside(x, y, shape):
        return np.square(x) + np.square(y) <= shape._scale_sqr

    @staticmethod
    def intersect_cells_batch(loc_corners: np.array, shape) -> np.ndarray:
        """
        loc_corners - shape (n_cells, 3, 8) where n_cells is the number of cells
        Returns: np.ndarray with shape (n_cells,), True where the cell intersects the fracture
        """
        # Compute centers for all cells at once (shape: n_cells x 3)
        centers = loc_corners.mean(axis=2)  # Resulting shape: (n_cells, 3)

        # centers = np.add.reduce(loc_corners, axis=2) / loc_corners.shape[2]

        # Check if centers are inside the fracture shape (only for x and y)
        # point_inside = np.array([shape.is_point_inside(*center[:2]) for center in centers])

        # print("np.sum(point_inside) ", np.sum(point_inside))
        point_inside = np.array(DFMSim3D.is_point_inside(centers[:, 0], centers[:, 1], shape))
        # print("new np.sum(point_inside) ", np.sum(point_inside))

        # Check the Z-coordinate condition: any value >= 0 and any value < 0 for each cell
        z_conditions = (np.min(loc_corners[:, 2, :], axis=1) >= 0.) | (np.max(loc_corners[:, 2, :], axis=1) < 0.)

        # The final condition: Both center inside check and valid Z-condition
        return point_inside & ~z_conditions

    @staticmethod
    def intersection_cell_corners_vec(dfn: FractureSet, grid: Grid) -> 'Intersection':
        domain = FracturedDomain(dfn, np.ones(len(dfn)), grid)

        i_cell = []
        i_fr = []

        grid_cumul_prod = np.array([1, grid.shape[0], grid.shape[0] * grid.shape[1]])
        for i in range(len(dfn)):
            i_box_min, i_box_max = grid.coord_aabb(dfn.AABB[i])
            #print("i box min: {}, i box max: {}".format(i_box_min, i_box_max))
            axis_ranges = [range(max(0, a), min(b, n)) for a, b, n in zip(i_box_min, i_box_max, grid.shape)]
            all_kji = np.array(list(itertools.product(*reversed(axis_ranges))))  # Convert to array at once
            if all_kji.shape[0] > 0:
                all_ijk = np.flip(all_kji, axis=1)
                corners = grid.origin + (all_ijk[:, None, :] + DFMSim3D._rel_corner) * grid.step  # Shape: (N_cells, 8, 3)
                loc_corners = np.einsum('ij,jkl->ikl', dfn.inv_transform_mat[i], (corners - dfn.center[i]).T).transpose(
                    2, 0, 1)

                intersections = DFMSim3D.intersect_cells_batch(loc_corners, dfn.base_shape)

                # Compute cell indices for valid intersections
                valid_cells = np.where(intersections)[0]  # Get the indices where intersection is True
                valid_ijk = all_ijk[np.where(intersections)[0]]
                cell_indices = valid_ijk.dot(grid_cumul_prod)

                # Store results for valid intersections
                i_cell.extend(cell_indices)
                i_fr.extend([i] * valid_cells.shape[0])

        return Intersection.const_isec(domain, i_cell, i_fr, 1.0)

    @staticmethod
    def rasterize(fem_grid, dfn, bulk_cond, fr_cond):
        #steps = 3 * [41]
        target_grid = fem_grid.grid #Grid(3 * [15], steps, origin=3 * [-7.5])  # test grid with center in (0,0,0)

        # import cProfile
        # import pstats
        # pr = cProfile.Profile()
        # pr.enable()

        #isec_corners_orig = intersection_cell_corners(dfn, target_grid)

        #print("target_grid ", target_grid)


        isec_corners = DFMSim3D.intersection_cell_corners_vec(dfn, target_grid)

        #print("isec corners ", isec_corners.grid)
        #print("isec_corners.count_fr_cells ", isec_corners.count_fr_cells())
        #print("isec_corners_vec.fr_cells ", isec_corners_vec.count_fr_cells())

        # pr.disable()
        # ps = pstats.Stats(pr).sort_stats('cumtime')
        # ps.print_stats(15)

        #print("isec corners ", isec_corners)
        #print("isec corners vec ", isec_corners_vec)

        # isec_probe = probe_fr_intersection(fr_set, target_grid)
        #cross_section, fr_cond = fr_conductivity(dfn)
        rasterized = isec_corners.interpolate(bulk_cond, fr_cond, source_grid=fem_grid.grid)

        #print("rasterized ", rasterized.shape)

        #np.save("rasterized", rasterized)

        # axes = isec_corners.grid.axes_linspace()
        #
        # data = np.log10(rasterized[:, 0, 0]) #np.random.rand(64, 64, 64)  # Replace with your actual voxel data
        #
        # # Grid parameters
        # nx, ny, nz = 64, 64, 64 #data.shape
        # length = 15.0
        # spacing = length / nx  # 15 / 64
        # origin = (-length / 2, -length / 2, -length / 2)
        #
        # # Create the structured grid
        # grid = pv.ImageData()#pv.UniformGrid()
        #
        # # Set dimensions (one more than number of cells in each direction)
        # grid.dimensions = np.array([65, 65, 65])
        #
        # # Set spacing and origin
        # grid.spacing = (spacing, spacing, spacing)
        # grid.origin = origin
        #
        # # Add the scalar data
        # grid.cell_data["values"] = data#.flatten(order="F")  # Fortran order
        #
        # # Plot it
        # #grid.plot(show_edges=True, opacity=0.5)
        #
        # # Use threshold to hide zero/low values if needed
        # plotter = pv.Plotter()
        # actor = plotter.add_volume(
        #     grid,
        #     scalars="values",
        #     cmap="viridis",
        #     opacity="sigmoid",  # can also try "linear" or a float (e.g., 0.2)
        #     show_scalar_bar=True
        # )
        # plotter.show()
        #DFMSim3D.plot_isec_fields2(isec_corners, bulk_cond, rasterized, "raster_field.vtk")

        for i_ax in range(3):
            assert np.all(bulk_cond[:, i_ax, i_ax] <= rasterized[:, i_ax, i_ax])
        for i_ax in range(3):
            assert np.all(rasterized[:, i_ax, i_ax].max() <= fr_cond[:, i_ax, i_ax].max())

        return rasterized

    @staticmethod
    def make_fr_cond_pd(fr_cond):
        for i in range(fr_cond.shape[0]):
            evals, evecs = np.linalg.eigh(fr_cond[i])
            if np.any(evals <= 0):
                epsilon = 1e-10  # Small regularization term
                fr_cond[i] = fr_cond[i] + epsilon * np.eye(3)

        return fr_cond

    @staticmethod
    def fr_conductivity(dfn, cross_section_factor=1e-4, perm_factor=1.0):
        """
        :param dfn:
        :param cross_section_factor: scalar = cross_section / fracture mean radius
        :return:
        """
        rho = 1000
        g = 9.81
        viscosity = 8.9e-4
        perm_to_cond = rho * g / viscosity
        cross_section = cross_section_factor * np.sqrt(np.prod(dfn.radius, axis=1))
        perm = perm_factor * cross_section * cross_section / 12

        conductivity = perm_to_cond * perm
        cond_tn = conductivity[:, None, None] * (np.eye(3))  # - dfn.normal[:, :, None] * dfn.normal[:, None, :])

        # for r, per in zip(dfn.radius, cond_tn):
        #     print("radius: {}, cond: {}".format(r, per))

        return cross_section, cond_tn

    @staticmethod
    def create_mesh_fields(fr_media, bulk_cond_values, bulk_cond_points, dimensions, mesh_step, sample_dir, work_dir, center=[0, 0, 0], outflow_problem=False, file_prefix="", fr_region_map=None):
        dfn = fr_media.dfn
        bulk_conductivity = fr_media.conductivity

        interp = sc_interpolate.LinearNDInterpolator(bulk_cond_points, bulk_cond_values, fill_value=0)
        #interp = sc_interpolate.RegularGridInterpolator(bulk_cond_points, bulk_cond_values)
        #interp = sc_interpolate.NearestNDInterpolator(bulk_cond_points, bulk_cond_values)

        ###########################
        ## Fracture conductivity ##
        ###########################
        fr_cond, fr_cross_section = [], []
        if len(dfn) > 0:
            fr_cross_section, fr_cond = DFMSim3D.fr_conductivity(dfn, cross_section_factor=1e-4)

        fr_cond = DFMSim3D.make_fr_cond_pd(fr_cond)

        n = 0
        for i in range(fr_cond.shape[0]):
            evals, evecs = np.linalg.eigh(fr_cond[i])
            #print("evals ", evals)
            if np.any(evals <= 0):
                print("evals equivalent cond tn ", evals)
                print("cond_tensors[i].shape", fr_cond[i])
                n += 1

        print("NUM fr cond non positive evals ", n)

        if not os.path.exists(os.path.join(sample_dir, "homo_cube_healed.msh2")) or fr_region_map is None:
            if outflow_problem:
                mesh_file, fr_region_map = DFMSim3D.ref_solution_mesh_outflow_problem(sample_dir, dimensions, dfn, fr_step=mesh_step,
                                                                      bulk_step=mesh_step, center=center)
            else:
                mesh_file, fr_region_map = DFMSim3D.ref_solution_mesh(sample_dir, dimensions, dfn, fr_step=mesh_step,
                                                                      bulk_step=mesh_step, center=center)
        else:
            mesh_file = sample_dir / ("homo_cube_healed.msh2")


        full_mesh = Mesh.load_mesh(mesh_file, heal_tol=0.001)

        #if len(dfn) > 0:
        fr_cond_tn, fr_map = DFMSim3D.fr_field(full_mesh, dfn, fr_region_map, fr_cond,
                                                   bulk_conductivity, rnd_cond=False, field_dim=3)

        cross_sections, fr_map_cs = DFMSim3D.fr_field(full_mesh, dfn, fr_region_map, fr_cross_section, 1.0, field_dim=1)
        cross_sections = np.array(cross_sections)
        #print("cross sections ", cross_sections)

        #######################################
        ## Interpolate SRF to mesh elements  ##
        #######################################
        bulk_elements_barycenters = full_mesh.el_barycenters(elements=full_mesh._bulk_elements)
        full_mesh_bulk_cond_values = interp(bulk_elements_barycenters)
        zero_rows = np.where(np.all(full_mesh_bulk_cond_values == 0, axis=1))[0]
        if len(zero_rows) > 0:
            print("ZERO ROWS")
            from scipy.interpolate import NearestNDInterpolator
            nn_interp = NearestNDInterpolator(bulk_cond_points, bulk_cond_values)
            full_mesh_bulk_cond_values[zero_rows] = nn_interp(bulk_elements_barycenters[zero_rows])

        ##################
        ## Write fields ##
        ##################
        #print("full_mesh_bulk_cond_values.shape ", full_mesh_bulk_cond_values.shape)
        #print("mean full_mesh_bulk_cond_values ", np.mean(full_mesh_bulk_cond_values, axis=(0,1,2)))
        fr_cond_tn[-full_mesh_bulk_cond_values.shape[0]:, ...] = full_mesh_bulk_cond_values
        conductivity = fr_cond_tn.reshape(fr_cond_tn.shape[0], 9)

        #######################
        #######################
        # unique conductivity tensors and their sizes
        fr_cond_unique = fr_cond
        fracture_size_unique = dfn.radius[:, 0]  # shape (N_unique,)

        # voxelized conductivity tensors (repeated values)
        voxel_fr_cond = fr_cond_tn[:-full_mesh_bulk_cond_values.shape[0], ...]  # shape (N_voxels, 3, 3)

        # Step 1: Flatten for comparison
        unique_flat = fr_cond_unique.reshape(len(fr_cond_unique), -1)
        voxel_flat = voxel_fr_cond.reshape(len(voxel_fr_cond), -1)

        # Step 2: Use broadcasting to find matches
        # (N_voxels, 1, features) == (1, N_unique, features) => (N_voxels, N_unique)
        matches = np.all(np.isclose(voxel_flat[:, None, :], unique_flat[None, :, :], rtol=1e-5), axis=2)

        # Step 3: Get index of matching unique tensor for each voxel
        voxel_to_unique_idx = np.argmax(matches, axis=1)  # shape (N_voxels,)

        # Step 4: Map fracture sizes
        voxel_fracture_size = fracture_size_unique[voxel_to_unique_idx]  # shape (N_voxels,)

        # print("voxel_fracture_sizes ", voxel_fracture_size)
        # print("fr_cond_tn[:-full_mesh_bulk_cond_values.shape[0], ...] ", fr_cond_tn[:-full_mesh_bulk_cond_values.shape[0], ...])

        np.save("{}voxel_fracture_sizes".format(file_prefix), voxel_fracture_size)
        np.save("{}fr_cond_data".format(file_prefix), fr_cond_tn[:-full_mesh_bulk_cond_values.shape[0], ...])
        np.save("{}bulk_cond_data".format(file_prefix), fr_cond_tn[-full_mesh_bulk_cond_values.shape[0]:, ...])

        fields = dict(conductivity=conductivity, cross_section=cross_sections.reshape(-1, 1))
        cond_file = full_mesh.write_fields(str(sample_dir / "input_fields.msh2"), fields)
        cond_file = Path(cond_file)
        cond_file = cond_file.rename(cond_file.with_suffix(".msh"))

        #cond_file_vtk = full_mesh.write_fields(str(sample_dir / "input_fields.vtk"), fields)
        # cond_file = Path(cond_file)
        # cond_file = cond_file.rename(cond_file.with_suffix(".msh"))

        # print(cond_file)
        # print("cond_file.with_suffix(.msh ", cond_file.with_suffix(".msh"))
        # import meshio
        # mesh = meshio.read(cond_file.with_suffix(".msh"))
        # mesh.write(cond_file.with_suffix(".vtk"))  # Use .vtk if you prefer
        # exit()

        return cond_file, fr_cond, fr_region_map, full_mesh.regions

    @staticmethod
    def calculate_hom_sample(config, sample_dir, sample_idx, sample_seed):
        domain_size = 15 #15  # 15  # 100
        fem_grid_cond_domain_size = 16 #16
        fr_domain_size = 100
        fr_range = (config["fine"]["step"], fr_domain_size) #(5, fr_domain_size)
        coarse_step = config["coarse"]["step"]
        coarse_step = 10

        print("coarse step ", coarse_step)

        # domain_size = 20  # 15  # 100
        # fem_grid_cond_domain_size = 21
        # fr_domain_size = 100
        # fr_range = (config["fine"]["step"], fr_domain_size)  # (5, fr_domain_size)
        # coarse_step = config["coarse"]["step"]
        fine_step = config["fine"]["step"]

        sim_config = config["sim_config"]
        # print("sim config ", sim_config)
        geom = sim_config["geometry"]

        # print("n frac limit ", geom["n_frac_limit"])

        ###################
        # DFN
        ###################
        dfn = []
        if geom["n_frac_limit"] > 0:
            dfn = DFMSim3D.fracture_random_set(sample_seed, fr_range, sim_config["work_dir"], max_frac=geom["n_frac_limit"])
            dfn_to_homogenization = []

            fr_rad_values = []
            total_area = 0
            total_excluded_volume = []
            for fr in dfn:
                # fr.r should be a radius of inscribed circle
                print("fr.r ", fr.r)
                #print("fr.R", fr.R)
                total_area += 4 * fr.r ** 2
                #total_excluded_volume.append(32/3 * np.pi * fr.r ** 3)
                total_excluded_volume.append(4*np.sqrt(2) * fr.r ** 3)
                fr_rad_values.append(fr.radius[0] * (fr.radius[1] ** 2) + (fr.radius[0] ** 2) * fr.radius[1])

            print("Whole sample mean fr rad values ", np.mean(fr_rad_values))
            rho_3D = (np.pi ** 2) / 2 * np.mean(fr_rad_values) * (len(dfn) / fr_range[1] ** 3)
            rho_3D_new = np.mean(total_excluded_volume) * len(dfn) / fr_range[1] ** 3
            print("orig DFN rho_3D ", rho_3D)
            print("orig DFN rho_3D_new ", rho_3D_new)

            for fr in dfn:
                print("fr.r ", fr.r)
                if fine_step <= fr.r <= coarse_step:
                    dfn_to_homogenization.append(fr)
            print("len dfn_to_homogenization ", len(dfn_to_homogenization))

            dfn = stochastic.FractureSet.from_list(dfn_to_homogenization)

        # dfn = dfn_to_homogenization
        ########################
        ########################
        ########################
        total_area = 0
        fr_rad_values = []
        for fr in dfn:
            fr_rad_values.append(fr.radius[0] * (fr.radius[1] ** 2) + (fr.radius[0] ** 2) * fr.radius[1])
            total_area += (fr.radius[0] * fr.radius[1])

        print("total area ", total_area)
        #print("dfn.area ", dfn.area)

        print("Whole sample mean fr rad values ", np.mean(fr_rad_values))
        rho_3D = np.pi ** 2 / 2 * np.mean(fr_rad_values) * (len(dfn) / fr_range[1] ** 3)
        print("DFN to homogenization rho_3D ", rho_3D)

        # exit()
        # steps = (int(fine_step), int(fine_step), int(fine_step))

        # n_steps_coef = 1.5
        # n_steps = (int(domain_size/int(fine_step)*n_steps_coef), int(domain_size/int(fine_step)*n_steps_coef), int(domain_size/int(fine_step)*n_steps_coef))

        # n_steps = (64, 64, 64)
        # n_steps = (25, 25, 25)
        # n_steps = (4, 4, 4)

        n_steps = config["sim_config"]["geometry"]["n_voxels"]
        print("n steps ", n_steps)

        # Cubic law transmissvity
        fr_media = FracturedMedia.fracture_cond_params(dfn, 1e-4, 0.00001)

        # cross_section, cond_tn = fr_conductivity(dfn)
        #
        # print("cross section ", cross_section)
        # print("cond_tn ", cond_tn)

        # fem_grid_cond = fem.fem_grid(domain_size, n_steps, fem.Fe.Q(dim=3),
        #                              origin=-domain_size / 2)  # 27 cells
        n_steps_cond_grid = (fem_grid_cond_domain_size, fem_grid_cond_domain_size, fem_grid_cond_domain_size)

        #n_steps_cond_grid = (2,2,2)

        print("n steps cond grid ", n_steps_cond_grid)

        fem_grid_cond = fem.fem_grid(fem_grid_cond_domain_size, n_steps_cond_grid, fem.Fe.Q(dim=3),
                                     origin=-fem_grid_cond_domain_size / 2)  # 27 cells
        fem_grid_rast = fem.fem_grid(domain_size, n_steps, fem.Fe.Q(dim=3),
                                     origin=-domain_size / 2)  # 27 cells

        #######################
        ## Bulk conductivity ##
        #######################
        bulk_cond_values, bulk_cond_points = DFMSim3D.generate_grid_cond(fem_grid_cond, config, seed=sample_seed)

        print("bulk cond values shape ", bulk_cond_values.shape)
        print("bulk cond points shaoe ", bulk_cond_points.shape)

        # grid_cond = DFMSim3D.homo_decovalex(fr_media, fem_grid.grid)
        # grid_cond = grid_cond.reshape(*n_steps, grid_cond.shape[-1])

        # print("grid_cond shape ", grid_cond.shape)
        # @TODO: rasterize input

        # print(grid_cond.shape)
        # print("np.array(bc_pressure_gradient)[None, :] ", np.array(bc_pressure_gradient)[None, :])

        # grid_cond = np.ones(grid.n_elements)[:, None] * np.array([1, 1, 1, 0, 0, 0])[None, :]

        ######
        ## @TODO: there is no flow123d called inside
        # pressure = fem_grid.solve_sparse(grid_cond, np.array(bc_pressure_gradient)[None, :])
        # assert not np.any(np.isnan(pressure))

        ##################################
        ## Create mesh and input fields ##
        ##################################
        dimensions = (domain_size, domain_size, domain_size)

        # bc_pressure_gradient = [1, 0, 0]
        # flux_response_0 = DFMSim3D.get_flux_response(bc_pressure_gradient, fr_media, fem_grid, config, sample_dir, sim_config)
        # exit()

        bc_pressure_gradient = [1, 0, 0]
        cond_file, fr_cond = DFMSim3D._run_sample(bc_pressure_gradient, fr_media, config, sample_dir, bulk_cond_values,
                                                  bulk_cond_points, dimensions,  mesh_step=config["fine"]["step"])
        flux_response_0 = DFMSim3D.get_flux_response()  # bc_pressure_gradient, fr_media, fem_grid, config, sample_dir,
        # sim_config)

        bc_pressure_gradient = [0, 1, 0]
        DFMSim3D._run_sample(bc_pressure_gradient, fr_media, config, sample_dir, bulk_cond_values, bulk_cond_points,
                             dimensions, cond_file=cond_file,  mesh_step=config["fine"]["step"])
        flux_response_1 = DFMSim3D.get_flux_response()  # bc_pressure_gradient, fr_media, fem_grid, config, sample_dir,
        # sim_config)

        bc_pressure_gradient = [0, 0, 1]
        DFMSim3D._run_sample(bc_pressure_gradient, fr_media, config, sample_dir, bulk_cond_values, bulk_cond_points,
                             dimensions, cond_file=cond_file, mesh_step=config["fine"]["step"])
        flux_response_2 = DFMSim3D.get_flux_response()  # bc_pressure_gradient, fr_media, fem_grid, config, sample_dir,
        # sim_config)

        # print("flux response ", flux_response_0)

        #
        # #bc_pressure_gradients = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        # flux_response_0 = DFMSim3D.get_flux_response(bc_pressure_gradient, fr_media, fem_grid, config, sample_dir, sim_config)
        #
        # # print("flux responses shape ", flux_responses.shape)
        # # print("flux responses ", flux_responses)
        # #
        # bc_pressure_gradient = [0, 1, 0]
        # flux_response_1 = DFMSim3D.get_flux_response(bc_pressure_gradient, fr_media, fem_grid, config, sample_dir,
        #                                            sim_config)
        #
        # bc_pressure_gradient = [0, 0, 1]
        # flux_response_2 = DFMSim3D.get_flux_response(bc_pressure_gradient, fr_media, fem_grid, config, sample_dir,
        #                                            sim_config)

        bc_pressure_gradients = np.stack(([1, 0, 0], [0, 1, 0], [0, 0, 1]), axis=0)
        flux_responses = np.squeeze(np.stack((flux_response_0, flux_response_1, flux_response_2), axis=0))

        # print("bc pressure gradients shape ", bc_pressure_gradients.shape)
        # print("bc pressure gradients ", bc_pressure_gradients)
        #
        # print("flux_responses shape ", flux_responses.shape)
        # print("flux_responses ", flux_responses)

        equivalent_cond_tn_voigt = equivalent_posdef_tensor(np.array(bc_pressure_gradients), flux_responses)
        # print("equivalent_cond_tn_voigt ", equivalent_cond_tn_voigt)

        equivalent_cond_tn = voigt_to_tn(np.array([equivalent_cond_tn_voigt]))  # np.zeros((3, 3))

        # # Map the Voigt vector to the symmetric matrix
        # equivalent_cond_tn[0, 0] = equivalent_cond_tn_voigt[0]  # xx
        # equivalent_cond_tn[1, 1] = equivalent_cond_tn_voigt[1]  # yy
        # equivalent_cond_tn[2, 2] = equivalent_cond_tn_voigt[2]  # zz
        # equivalent_cond_tn[1, 2] = equivalent_cond_tn[2, 1] = equivalent_cond_tn_voigt[3]  # yz or zy
        # equivalent_cond_tn[0, 2] = equivalent_cond_tn[2, 0] = equivalent_cond_tn_voigt[4]  # xz or zx
        # equivalent_cond_tn[0, 1] = equivalent_cond_tn[1, 0] = equivalent_cond_tn_voigt[5]  # xy or yx
        #print("equivalent cond tn ", equivalent_cond_tn)
        evals, evecs = np.linalg.eigh(equivalent_cond_tn)
        #print("evals equivalent cond tn ", evals)
        assert np.all(evals) > 0

        fine_res = np.squeeze(equivalent_cond_tn_voigt)

        print("fine res", fine_res)

        #DFMSim3D._remove_files()

        gen_hom_samples = False
        if "generate_hom_samples" in config["sim_config"] and config["sim_config"]["generate_hom_samples"]:
            gen_hom_samples = True

        cond_tn_pop = os.path.join(config["fine"]["common_files_dir"], DFMSim3D.COND_TN_POP_FILE)
        if os.path.exists(cond_tn_pop):
            config["fine"]["cond_tn_pop_file"] = cond_tn_pop

        if "nn_path" in config["sim_config"]:
            pred_cond_tn_pop = os.path.join(config["fine"]["common_files_dir"], DFMSim3D.PRED_COND_TN_POP_FILE)
            if os.path.exists(pred_cond_tn_pop):
                config["fine"]["pred_cond_tn_pop_file"] = pred_cond_tn_pop

        sample_cond_tns = os.path.join(config["fine"]["common_files_dir"], DFMSim3D.SAMPLES_COND_TNS_DIR)
        if os.path.exists(sample_cond_tns):
            config["fine"]["sample_cond_tns"] = sample_cond_tns

        coarse_res = [0, 0, 0, 0, 0, 0]
        # print("fine res ", fine_res)

        #######################
        ## save to zarr file  #
        #######################
        # Shape of the data
        zarr_file_path = os.path.join(config["fine"]["common_files_dir"], DFMSim3D.ZARR_FILE)
        DFMSim3D.rasterize_save_to_zarr(zarr_file_path, config, sample_idx, bulk_cond_values, bulk_cond_points, dfn, fr_cond,
                               fem_grid_rast, n_steps, fine_res)

        #zarr_file_path = os.path.join(config["fine"]["common_files_dir"], DFMSim3D.ZARR_FILE)
        # print("zarr file path ", zarr_file_path)
        # if os.path.exists(zarr_file_path):
        #     print("zarr file path exists ", zarr_file_path)
        #     zarr_file = zarr.open(zarr_file_path, mode='r+')
        #     # Write data to the specified slice or index
        #
        #     #######
        #     ## bulk cond values to fem grid rast
        #     #######
        #     bulk_cond_fem_rast = DFMSim3D._bulk_cond_to_rast_grid(bulk_cond_values, bulk_cond_points, fem_grid_rast.grid)
        #
        #     rasterized_input = DFMSim3D.rasterize(fem_grid_rast, dfn, bulk_cond=bulk_cond_fem_rast, fr_cond=fr_cond)
        #
        #     rasterized_input_voigt = tn_to_voigt(rasterized_input)
        #     rasterized_input_voigt = rasterized_input_voigt.reshape(*n_steps, rasterized_input_voigt.shape[-1]).T
        #
        #     zarr_file["inputs"][sample_idx, ...] = rasterized_input_voigt
        #     zarr_file["outputs"][sample_idx, :] = fine_res

        return fine_res, coarse_res

    @staticmethod
    def rasterize_save_to_zarr(zarr_file_path, config, sample_idx, bulk_cond_values, bulk_cond_points, dfn, fr_cond, fem_grid_rast, n_steps, fine_res=None):
        if os.path.exists(zarr_file_path):
            zarr_file = zarr.open(zarr_file_path, mode='r+')
            # Write data to the specified slice or index
            #######
            ## bulk cond values to fem grid rast
            #######
            bulk_cond_fem_rast = DFMSim3D._bulk_cond_to_rast_grid(bulk_cond_values, bulk_cond_points,
                                                                  fem_grid_rast.grid)
            if len(dfn) == 0:
                rasterized_input = bulk_cond_fem_rast
            else:
                rasterized_input = DFMSim3D.rasterize(fem_grid_rast, dfn, bulk_cond=bulk_cond_fem_rast, fr_cond=fr_cond)

            rasterized_input_voigt = tn_to_voigt(rasterized_input)
            rasterized_input_voigt = rasterized_input_voigt.reshape(*n_steps, rasterized_input_voigt.shape[-1]).T

            zarr_file["inputs"][sample_idx, ...] = rasterized_input_voigt
            zarr_file["bulk_avg"][sample_idx, :] = np.mean(tn_to_voigt(bulk_cond_fem_rast), axis=0)
            if fine_res is not None:
                zarr_file["outputs"][sample_idx, :] = fine_res

    @staticmethod
    def _bulk_cond_to_rast_grid(bulk_cond_values, bulk_cond_points, grid_rast):
        from scipy.interpolate import griddata
        # Interpolate the scattered data onto the regular grid
        resized_data = griddata(bulk_cond_points, bulk_cond_values, grid_rast.barycenters(), method='nearest')
        return resized_data

    @staticmethod
    def _calculate_subdomains(coarse_step, geometry):
        #print("geometry ", geometry)
        if coarse_step > 0:
            hom_box_size = coarse_step * 1.5
            domain_box_size = geometry["orig_domain_box"][0]
            n_centers = np.round(domain_box_size / hom_box_size)
            n_total_centers = np.round(n_centers * geometry["pixel_stride_div"] + 1)
            hom_box_size = domain_box_size / n_centers
        else:
            return geometry["subdomain_box"], 4, 4

        return [hom_box_size, hom_box_size, hom_box_size], int(n_centers+1), int(n_total_centers)

    @staticmethod
    def get_equivalent_cond_tn(fr_media, config, sample_dir, bulk_cond_values, bulk_cond_points, dimensions, mesh_step, fr_region_map=None):
        bc_pressure_gradient = [1, 0, 0]
        cond_file, fr_cond, fr_region_map = DFMSim3D._run_sample(bc_pressure_gradient, fr_media, config, sample_dir, bulk_cond_values,
                                                  bulk_cond_points, dimensions, mesh_step)
        flux_response_0 = DFMSim3D.get_flux_response()  # bc_pressure_gradient, fr_media, fem_grid, config, sample_dir,
        # sim_config)

        bc_pressure_gradient = [0, 1, 0]
        DFMSim3D._run_sample(bc_pressure_gradient, fr_media, config, sample_dir, bulk_cond_values, bulk_cond_points,
                             dimensions, mesh_step=mesh_step, cond_file=cond_file)
        flux_response_1 = DFMSim3D.get_flux_response()  # bc_pressure_gradient, fr_media, fem_grid, config, sample_dir,
        # sim_config)

        bc_pressure_gradient = [0, 0, 1]
        DFMSim3D._run_sample(bc_pressure_gradient, fr_media, config, sample_dir, bulk_cond_values, bulk_cond_points,
                             dimensions, mesh_step=mesh_step, cond_file=cond_file)
        flux_response_2 = DFMSim3D.get_flux_response()  # bc_pressure_gradient, fr_media, fem_grid, config, sample_dir,
        # sim_config)

        # print("flux response ", flux_response_0)

        bc_pressure_gradients = np.stack(([1, 0, 0], [0, 1, 0], [0, 0, 1]), axis=0)
        flux_responses = np.squeeze(np.stack((flux_response_0, flux_response_1, flux_response_2), axis=0))

        equivalent_cond_tn_voigt = equivalent_posdef_tensor(np.array(bc_pressure_gradients), flux_responses)

        # print("equivalent_cond_tn_voigt ", equivalent_cond_tn_voigt)

        #equivalent_cond_tn = voigt_to_tn(np.array([equivalent_cond_tn_voigt]))  # np.zeros((3, 3))

        return equivalent_cond_tn_voigt, fr_cond, fr_region_map

    @staticmethod
    def rasterize_at_once_zarr(config, dfn_to_homogenization, bulk_cond_values, bulk_cond_points, fem_grid_rast, n_subdomains_per_axes):
        zarr_file_path = DFMSim3D.create_zarr_file(os.getcwd(), n_samples=int(n_subdomains_per_axes ** 3), config_dict=config, centers=True)

        fem_grid_n_steps = fem_grid_rast.grid.shape

        print("fem grid n steps ", fem_grid_n_steps)

        bulk_cond_fem_rast = DFMSim3D._bulk_cond_to_rast_grid(bulk_cond_values, bulk_cond_points, fem_grid_rast.grid)

        fr_cond, fr_cross_section = [], []
        if len(dfn_to_homogenization) > 0:
            fr_cross_section, fr_cond = DFMSim3D.fr_conductivity(dfn_to_homogenization, cross_section_factor=1e-4)

        bulk_cond_fem_rast_voigt = tn_to_voigt(bulk_cond_fem_rast)
        bulk_cond_fem_rast_voigt = bulk_cond_fem_rast_voigt.reshape(*fem_grid_n_steps, bulk_cond_fem_rast_voigt.shape[-1]).T
        #print("bulk cond fem rast ", bulk_cond_fem_rast.shape)

        #print("len(dfn_to_homogenization) ", len(dfn_to_homogenization))

        if len(dfn_to_homogenization) == 0:
            rasterized_input = bulk_cond_fem_rast
        else:
            rasterized_input = DFMSim3D.rasterize(fem_grid_rast, dfn_to_homogenization, bulk_cond=bulk_cond_fem_rast,
                                                  fr_cond=fr_cond)

        rasterized_input_voigt = tn_to_voigt(rasterized_input)

        rasterized_input_voigt = rasterized_input_voigt.reshape(*fem_grid_n_steps, rasterized_input_voigt.shape[-1]).T

        # print("rasterized input voigt ", rasterized_input_voigt.shape)
        # print("rasterized input voigt ", rasterized_input_voigt[0, ...])

        domain_box = config["sim_config"]["geometry"]["orig_domain_box"]
        subdomain_box = config["sim_config"]["geometry"]["subdomain_box"]
        subdomain_pixel_size = 64
        pixel_stride = subdomain_pixel_size // config["sim_config"]["geometry"]["pixel_stride_div"]

        lx, ly, lz = domain_box
        lx += subdomain_box[0]
        ly += subdomain_box[1]
        lz += subdomain_box[2]

        batch_size = 1

        C, H, W, D = rasterized_input_voigt.shape

        # Calculate the number of subdomains in each dimension
        num_subdomains_x = (H - subdomain_pixel_size) // pixel_stride + 1
        num_subdomains_y = (W - subdomain_pixel_size) // pixel_stride + 1
        num_subdomains_z = (D - subdomain_pixel_size) // pixel_stride + 1

        index = 0
        dict_centers = []
        dict_cond_tn_values = []

        stride = config["sim_config"]["geometry"]["pixel_stride_div"]

        zarr_file = zarr.open(zarr_file_path, mode='r+')
        # for i in range(num_subdomains_x):
        #     center_x = subdomain_box[0] / stride + (lx - subdomain_box[0]) / (n_subdomains_per_axes - 1) * i - lx / stride
        #
        #     for j in range(num_subdomains_y):
        #         #start_time = time.time()
        #
        #         center_y = subdomain_box[1] / stride + (ly - subdomain_box[1]) / (n_subdomains_per_axes - 1) * j - ly / stride
        #
        #         for l in range(num_subdomains_z):
        #
        #             #subdir_name = "i_{}_j_{}_l_{}_k_{}".format(i, j, l, k)
        #             #os.mkdir(subdir_name)
        #             #os.chdir(subdir_name)
        #
        #             #hom_sample_dir = Path(os.getcwd()).absolute()
        #
        #             center_z = subdomain_box[1] / stride + (lz - subdomain_box[1]) / (n_subdomains_per_axes - 1) * l - lz / stride
        #
        #             # center_z_2 = subdomain_box[1] / 2 + (lz - subdomain_box[1]) / (
        #             #             n_subdomains_per_axes - 1) * (l+1) - lz / 2
        #
        #             #center_x = 0.0
        #             #center_y = -7.5
        #             #center_z = -7.5
        #             #k = 10
        #             #
        #
        #             zarr_file["centers"][index, :] = (center_x, center_y, center_z)
        #
        #             zarr_file["inputs"][index, :] = rasterized_input_voigt[:,
        #                                             i * pixel_stride: i * pixel_stride + subdomain_pixel_size,
        #                                             j * pixel_stride: j * pixel_stride + subdomain_pixel_size,
        #                                             l * pixel_stride: l * pixel_stride + subdomain_pixel_size]
        #
        #
        #             # if index == 0:
        #             #     print("center x:{} y:{}, z:{}, k: {}".format(center_x, center_y, center_z, index))
        #             #     # print("rasterized_input_voigt[:,] ", rasterized_input_voigt[:,
        #             #     #                             i * pixel_stride: i * pixel_stride + subdomain_pixel_size,
        #             #     #                             j * pixel_stride: j * pixel_stride + subdomain_pixel_size,
        #             #     #                             l * pixel_stride: l * pixel_stride + subdomain_pixel_size])
        #             #
        #             #     print("rasterized_input_voigt[:, 0, 0, 0] ", rasterized_input_voigt[:, 0, 0, 0])
        #
        #
        #
        #             #print("i * pixel_stride: {}, i * pixel_stride + subdomain_pixel_size: {}".format(i * pixel_stride, i * pixel_stride + subdomain_pixel_size))
        #
        #             #print("zarr_file[inputs][index, :].shape ", zarr_file["inputs"][index, :].shape)
        #
        #
        #             hom_block_bulk = bulk_cond_fem_rast_voigt[:,
        #                              i * pixel_stride: i * pixel_stride + subdomain_pixel_size,
        #                              j * pixel_stride: j * pixel_stride + subdomain_pixel_size,
        #                              l * pixel_stride: l * pixel_stride + subdomain_pixel_size]
        #
        #             #print("hom_block_bulk shape ", hom_block_bulk.shape)
        #             #print("np.mean(hom_block_bulk, axis=(1, 2, 3)) ", np.mean(hom_block_bulk, axis=(1, 2, 3)))
        #
        #             zarr_file["bulk_avg"][index, :] = np.mean(hom_block_bulk, axis=(1, 2, 3))
        #             index += 1

        num_subdomains = num_subdomains_x * num_subdomains_y * num_subdomains_z
        # Generate all index combinations
        all_indices = itertools.product(range(num_subdomains_x), range(num_subdomains_y), range(num_subdomains_z))

        zarr_batch_size = 1

        # Process in batches
        for start in range(0, num_subdomains, zarr_batch_size):
            end = min(start + zarr_batch_size, num_subdomains)

            # Preallocate batch arrays
            batch_centers = np.zeros((end - start, 3), dtype=np.float32)
            batch_inputs = np.zeros((end - start, *(rasterized_input_voigt.shape[0], subdomain_pixel_size, subdomain_pixel_size, subdomain_pixel_size)), dtype=np.float32)
            batch_bulk_avg = np.zeros((end - start, rasterized_input_voigt.shape[0]), dtype=np.float32)

            # Use `itertools.islice` to get a batch from the iterator
            batch_indices = list(itertools.islice(all_indices, zarr_batch_size))

            # Fill batch
            for batch_index, (i, j, l) in enumerate(batch_indices):
                center_x = subdomain_box[0] / stride + (lx - subdomain_box[0]) / (
                            n_subdomains_per_axes - 1) * i - lx / stride
                center_y = subdomain_box[1] / stride + (ly - subdomain_box[1]) / (
                            n_subdomains_per_axes - 1) * j - ly / stride
                center_z = subdomain_box[2] / stride + (lz - subdomain_box[2]) / (
                            n_subdomains_per_axes - 1) * l - lz / stride

                #print("center x:{} y:{}, z:{}".format(center_x, center_y, center_z))

                batch_centers[batch_index] = (center_x, center_y, center_z)
                batch_inputs[batch_index] = rasterized_input_voigt[:,
                                            i * pixel_stride: i * pixel_stride + subdomain_pixel_size,
                                            j * pixel_stride: j * pixel_stride + subdomain_pixel_size,
                                            l * pixel_stride: l * pixel_stride + subdomain_pixel_size]

                batch_bulk_avg[batch_index] = np.mean(bulk_cond_fem_rast_voigt[:,
                                 i * pixel_stride: i * pixel_stride + subdomain_pixel_size,
                                 j * pixel_stride: j * pixel_stride + subdomain_pixel_size,
                                 l * pixel_stride: l * pixel_stride + subdomain_pixel_size], axis=(1, 2, 3))

            # Batch write to Zarr
            zarr_file["centers"][start:end] = batch_centers
            zarr_file["inputs"][start:end] = batch_inputs
            zarr_file["bulk_avg"][start:end] = batch_bulk_avg

        # ####################
        # inputs = zarr_file['inputs']
        # outputs = zarr_file['outputs']
        # bulk_avg = zarr_file['bulk_avg']
        # centers = zarr_file['centers']
        #
        # # Get the shape of the datasets
        # n_samples_existing = inputs.shape[0]  # Number of existing samples
        #
        # # Define how many times you want to repeat the data to form a larger Zarr
        # repeat_factor = 5  # For example, repeat 10 times
        #
        # # Calculate the new number of samples for the larger Zarr
        # new_n_samples = n_samples_existing * repeat_factor
        # new_zarr_file_path = os.path.join(os.getcwd(), "Larger_Zarr_File.zarr")
        #
        # # Create a new Zarr file to store the larger dataset
        # if not os.path.exists(new_zarr_file_path):
        #     new_zarr_file = zarr.open(new_zarr_file_path, mode='w')
        #
        #     # Create the 'inputs' dataset with the new shape
        #     new_inputs = new_zarr_file.create_dataset('inputs',
        #                                               shape=(new_n_samples,) + inputs.shape[1:],
        #                                               # New shape with repeated samples
        #                                               dtype='float32',
        #                                               chunks=(inputs.chunks[0],) + inputs.shape[1:],
        #                                               fill_value=0)
        #
        #     # Create the 'outputs' dataset with the new shape
        #     new_outputs = new_zarr_file.create_dataset('outputs',
        #                                                shape=(new_n_samples,) + outputs.shape[1:],
        #                                                # New shape with repeated samples
        #                                                dtype='float32',
        #                                                chunks=(outputs.chunks[0],) + outputs.shape[1:],
        #                                                fill_value=0)
        #
        #     # Create the 'bulk_avg' dataset with the new shape
        #     new_bulk_avg = new_zarr_file.create_dataset('bulk_avg',
        #                                                 shape=(new_n_samples,) + bulk_avg.shape[1:],
        #                                                 # New shape with repeated samples
        #                                                 dtype='float32',
        #                                                 chunks=(bulk_avg.chunks[0],) + bulk_avg.shape[1:],
        #                                                 fill_value=0)
        #
        #     # Create the 'centers' dataset with the new shape
        #     new_centers = new_zarr_file.create_dataset('centers',
        #                                                shape=(new_n_samples,) + centers.shape[1:],
        #                                                # New shape with repeated samples
        #                                                dtype='float32',
        #                                                chunks=(centers.chunks[0],) + centers.shape[1:],
        #                                                fill_value=0)
        #
        #     # Populate the new datasets by copying the data multiple times (using the repeat_factor)
        #     new_inputs[:] = np.tile(inputs[:], (repeat_factor, 1, 1, 1, 1))  # Repeating the inputs dataset
        #     new_outputs[:] = np.tile(outputs[:], (repeat_factor, 1))  # Repeating the outputs dataset
        #     new_bulk_avg[:] = np.tile(bulk_avg[:], (repeat_factor, 1))  # Repeating the bulk_avg dataset
        #     new_centers[:] = np.tile(centers[:], (repeat_factor, 1))  # Repeating the centers dataset
        #
        #     print(f"New larger Zarr file created with {new_n_samples} samples.")
        # else:
        #     print(f"The larger Zarr file already exists at {new_zarr_file_path}.")
        #
        # zarr_file_path = new_zarr_file_path
        # ####################

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        import torch.autograd.profiler as profiler

        if DFMSim3D.model is None:
            nn_path = config["sim_config"]["nn_path"]
            study = load_study(nn_path)
            model_path = get_saved_model_path(nn_path, study.best_trial)
            model_kwargs = study.best_trial.user_attrs["model_kwargs"]
            DFMSim3D.model = study.best_trial.user_attrs["model_class"](**model_kwargs)
            if not torch.cuda.is_available():
                DFMSim3D.checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            else:
                DFMSim3D.checkpoint = torch.load(model_path)
            DFMSim3D.inverse_transform = get_inverse_transform(study, results_dir=nn_path)
            DFMSim3D.transform = get_transform(study, results_dir=nn_path)

        ##
        # Create dataset
        ##
        dataset_for_prediction = DFM3DDataset(zarr_path=zarr_file_path,
                                              init_transform=DFMSim3D.transform[0],
                                              input_transform=DFMSim3D.transform[1],
                                              output_transform=DFMSim3D.transform[2],
                                              return_centers_bulk_avg=True)

        dset_prediction_loader = torch.utils.data.DataLoader(dataset_for_prediction, batch_size=batch_size, shuffle=False)

        DFMSim3D.model.load_state_dict(DFMSim3D.checkpoint['best_model_state_dict'])

        #DFMSim3D.model.eval()

        DFMSim3D.model = DFMSim3D.model.to(memory_format=torch.channels_last_3d)
        DFMSim3D.model.to(device).eval()

        #model.half()

        #example_input = torch.randn(1, 6, 64, 64, 64)
        #scripted_model = torch.jit.trace(DFMSim3D.model, example_input)
        #scripted_model = torch.compile(DFMSim3D.model, backend="aot_eager")
        #output = scripted_model(data)

        # print(" torch.cuda.is_available() ", torch.cuda.is_available())

        #DFMSim3D.model = torch.compile(DFMSim3D.model)

        with torch.inference_mode():
            for i, sample in enumerate(dset_prediction_loader):
                # print("i ", i)
                inputs, targets, centers, bulk_features_avg = sample
                # print("inputs ", inputs.shape)
                # print("centers ", centers)
                # print("bulk features avg ", bulk_features_avg)

                inputs = inputs.to(memory_format=torch.channels_last_3d)  # Optimize for 3D convolution
                inputs = inputs.float().to(device)

                #inputs = inputs.contiguous()
                # sample_n = dataset_for_prediction._bulk_file_paths[i].split('/')[-2]
                # center = sample_center[sample_n]
                # print("torch.cuda.is_available() ", torch.cuda.is_available())
                # if torch.cuda.is_available():
                #     # print("cuda available")
                #     inputs = inputs.cuda()
                #     DFMSim3D.model = DFMSim3D.model.cuda()

                # with profiler.profile() as prof:
                #     #with profiler.record_function("conv3d"):
                #     predictions = DFMSim3D.model(inputs)
                # print(prof.key_averages().table(sort_by="cpu_time_total"))
                # exit()
                #with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                predictions = DFMSim3D.model(inputs)
                #predictions = model(inputs)

                predictions = np.squeeze(predictions)

                # print("dset_prediction_loader ", dset_prediction_loader._bulk_features_avg)

                # print("zarr predictions ", predictions)
                # print("torch.reshape(predictions, (*predictions.shape, 1, 1))) ", torch.reshape(predictions, (*predictions.shape, 1, 1)).shape)

                # if np.any(predictions < 0):
                #     print("inputs ", inputs)
                #     print("negative predictions ", predictions)

                inv_predictions = torch.squeeze(
                    DFMSim3D.inverse_transform(torch.reshape(predictions, (*predictions.shape, 1, 1))))
                # print("inv predictions shape ", inv_predictions.shape)
                # print("bulk_features_avg.shape ", bulk_features_avg.shape)
                #print("inv predictions ", inv_predictions)

                if dataset_for_prediction.init_transform is not None:
                    # print("dataset_for_prediction._bulk_features_avg ", dataset_for_prediction._bulk_features_avg)
                    if len(inv_predictions.shape) > 1:
                        bulk_features_avg = bulk_features_avg.view(-1, 1)

                    # print("bulk_features_avg ", bulk_features_avg)
                    # print("inv predictions ", inv_predictions)

                    inv_predictions *= bulk_features_avg

                # print("inv predictions scaled ", inv_predictions)
                # exit()
                # return
                # return pred_cond_tensors

                # print("inv prediction shape ", inv_predictions.shape)

                # pred_cond_tn = np.array([[inv_predictions[0], inv_predictions[1]],
                #                          [inv_predictions[1], inv_predictions[2]]])

                # if pred_cond_tn is not None:

                # print("list(centers.numpy()) ", centers.tolist())
                # print("type list(centers.numpy()) ", type(centers.tolist()))

                # print("centers ", centers)
                # print("centers.numpy ", centers.numpy())
                # print("centers.numpy.shape ", centers.numpy().shape)
                # print("inv_predictions.numpy() ", inv_predictions.numpy().shape)
                # print("len(inv_predictions.numpy().shape) ", len(inv_predictions.numpy().shape))

                inv_predictions_numpy = inv_predictions.numpy()

                #print("len(inv_predictions_numpy.shape) ", len(inv_predictions_numpy.shape))
                if len(inv_predictions_numpy.shape) == 1:
                    inv_predictions_numpy = np.expand_dims(inv_predictions_numpy, axis=0)

                #print("voigt_to_tn(inv_predictions_numpy) ", voigt_to_tn(inv_predictions_numpy))

                # print("voigt_to_tn(inv_predictions_numpy)", voigt_to_tn(inv_predictions_numpy).shape)
                # exit()

                if len(inv_predictions.numpy().shape) == 1:
                    dict_cond_tn_values.extend([list(voigt_to_tn(inv_predictions_numpy))])
                else:
                    dict_cond_tn_values.extend(list(voigt_to_tn(inv_predictions_numpy)))

                # print("list(inv_predictions.numpy()) ", list(inv_predictions.numpy()))

                centers_tuples = map(tuple, centers.numpy())
                # print("centers_tuples ", list(centers_tuples))

                dict_centers.extend(centers_tuples)
                # dict_cond_tn_values.extend(list(inv_predictions.numpy()))

                # result_dict = dict(zip(dict_centers, dict_cond_tn_values))
                #
                # print("result dict ", result_dict)
                #exit()

        # result_dict = dict(zip(map(tuple, np.array(dict_centers).flatten()), np.array(dict_cond_tn_values)))
        #
        # print("result dicts ", result_dict)

        # print("dict centers ", dict_centers)
        # print("dict_cond_tn_values ", dict_cond_tn_values)
        #exit()

        pred_cond_tensors = dict(zip(dict_centers, dict_cond_tn_values))

        return pred_cond_tensors

    @staticmethod
    def get_sliding_indices(volume_size, window_size, stride):
        starts = list(range(0, volume_size - window_size + 1, stride))
        if starts[-1] + window_size < volume_size:
            starts.append(volume_size - window_size)  # Add last window to cover edge
        print("starts ", starts)
        return starts

    @staticmethod
    def rasterize_at_once_zarr_move_by_stride(config, dfn_to_homogenization, bulk_cond_values, bulk_cond_points, hom_box_size, domain_box_size, fem_grid_rast):
        fem_grid_n_steps = fem_grid_rast.grid.shape
        bulk_cond_fem_rast = DFMSim3D._bulk_cond_to_rast_grid(bulk_cond_values, bulk_cond_points, fem_grid_rast.grid)

        fr_cond, fr_cross_section = [], []
        if len(dfn_to_homogenization) > 0:
            fr_cross_section, fr_cond = DFMSim3D.fr_conductivity(dfn_to_homogenization, cross_section_factor=1e-4)

        bulk_cond_fem_rast_voigt = tn_to_voigt(bulk_cond_fem_rast)
        bulk_cond_fem_rast_voigt = bulk_cond_fem_rast_voigt.reshape(*fem_grid_n_steps,
                                                                    bulk_cond_fem_rast_voigt.shape[-1]).T
        # print("bulk cond fem rast ", bulk_cond_fem_rast.shape)

        # print("len(dfn_to_homogenization) ", len(dfn_to_homogenization))

        if len(dfn_to_homogenization) == 0:
            rasterized_input = bulk_cond_fem_rast
        else:
            rasterized_input = DFMSim3D.rasterize(fem_grid_rast, dfn_to_homogenization, bulk_cond=bulk_cond_fem_rast,
                                                  fr_cond=fr_cond)

        rasterized_input_voigt = tn_to_voigt(rasterized_input)

        rasterized_input_voigt = rasterized_input_voigt.reshape(*fem_grid_n_steps, rasterized_input_voigt.shape[-1]).T

        #domain_box = config["sim_config"]["geometry"]["orig_domain_box"]
        subdomain_box = hom_box_size #config["sim_config"]["geometry"]["subdomain_box"]
        subdomain_pixel_size = 64
        pixel_stride = subdomain_pixel_size // config["sim_config"]["geometry"]["pixel_stride_div"]

        lx, ly, lz = domain_box_size, domain_box_size, domain_box_size

        batch_size = 1

        dict_centers = []
        dict_cond_tn_values = []


        x_size, y_size, z_size = rasterized_input_voigt.shape[1:]  # Assuming (channels, X, Y, Z)

        x_starts = DFMSim3D.get_sliding_indices(x_size, subdomain_pixel_size, pixel_stride)
        y_starts = DFMSim3D.get_sliding_indices(y_size, subdomain_pixel_size, pixel_stride)
        z_starts = DFMSim3D.get_sliding_indices(z_size, subdomain_pixel_size, pixel_stride)

        all_indices = list(itertools.product(x_starts, y_starts, z_starts))
        num_subdomains = len(all_indices)

        zarr_file_path = DFMSim3D.create_zarr_file(os.getcwd(), n_samples=num_subdomains, config_dict=config, centers=True)
        zarr_file = zarr.open(zarr_file_path, mode='r+')

        print("num subdomains ", num_subdomains)

        zarr_batch_size = 1
        n_channels = rasterized_input_voigt.shape[0]

        voxel_spacing = subdomain_box/subdomain_pixel_size

        for start in range(0, num_subdomains, zarr_batch_size):
            end = min(start + zarr_batch_size, num_subdomains)

            batch_centers = np.zeros((end - start, 3), dtype=np.float32)
            batch_inputs = np.zeros(
                (end - start, n_channels, subdomain_pixel_size, subdomain_pixel_size, subdomain_pixel_size),
                dtype=np.float32)
            batch_bulk_avg = np.zeros((end - start, n_channels), dtype=np.float32)

            batch_indices = all_indices[start:end]

            for batch_index, (x_start, y_start, z_start) in enumerate(batch_indices):
                x_end = x_start + subdomain_pixel_size
                y_end = y_start + subdomain_pixel_size
                z_end = z_start + subdomain_pixel_size

                # Compute voxel center index
                cx_idx = (x_start + x_end) / 2
                cy_idx = (y_start + y_end) / 2
                cz_idx = (z_start + z_end) / 2

                # Convert to real coordinates relative to domain center (0, 0, 0)
                center_x = (cx_idx * voxel_spacing) - lx / 2
                center_y = (cy_idx * voxel_spacing) - ly / 2
                center_z = (cz_idx * voxel_spacing) - lz / 2

                batch_centers[batch_index] = (center_x, center_y, center_z)
                batch_inputs[batch_index] = rasterized_input_voigt[:, x_start:x_end, y_start:y_end, z_start:z_end]
                batch_bulk_avg[batch_index] = np.mean(
                    bulk_cond_fem_rast_voigt[:, x_start:x_end, y_start:y_end, z_start:z_end], axis=(1, 2, 3))

            zarr_file["centers"][start:end] = batch_centers
            zarr_file["inputs"][start:end] = batch_inputs
            zarr_file["bulk_avg"][start:end] = batch_bulk_avg

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        import torch.autograd.profiler as profiler

        if DFMSim3D.model is None:
            nn_path = config["sim_config"]["nn_path"]
            study = load_study(nn_path)
            model_path = get_saved_model_path(nn_path, study.best_trial)
            model_kwargs = study.best_trial.user_attrs["model_kwargs"]
            DFMSim3D.model = study.best_trial.user_attrs["model_class"](**model_kwargs)
            if not torch.cuda.is_available():
                DFMSim3D.checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            else:
                DFMSim3D.checkpoint = torch.load(model_path)
            DFMSim3D.inverse_transform = get_inverse_transform(study, results_dir=nn_path)
            DFMSim3D.transform = get_transform(study, results_dir=nn_path)

        ##
        # Create dataset
        ##
        dataset_for_prediction = DFM3DDataset(zarr_path=zarr_file_path,
                                              init_transform=DFMSim3D.transform[0],
                                              input_transform=DFMSim3D.transform[1],
                                              output_transform=DFMSim3D.transform[2],
                                              return_centers_bulk_avg=True)

        dset_prediction_loader = torch.utils.data.DataLoader(dataset_for_prediction, batch_size=batch_size,
                                                             shuffle=False)

        DFMSim3D.model.load_state_dict(DFMSim3D.checkpoint['best_model_state_dict'])
        DFMSim3D.model = DFMSim3D.model.to(memory_format=torch.channels_last_3d)
        DFMSim3D.model.to(device).eval()
        with torch.inference_mode():
            for i, sample in enumerate(dset_prediction_loader):
                inputs, targets, centers, bulk_features_avg = sample
                inputs = inputs.to(memory_format=torch.channels_last_3d)  # Optimize for 3D convolution
                inputs = inputs.float().to(device)

                predictions = DFMSim3D.model(inputs)
                predictions = np.squeeze(predictions)

                inv_predictions = torch.squeeze(
                    DFMSim3D.inverse_transform(torch.reshape(predictions, (*predictions.shape, 1, 1))))

                if dataset_for_prediction.init_transform is not None:
                    if len(inv_predictions.shape) > 1:
                        bulk_features_avg = bulk_features_avg.view(-1, 1)
                    inv_predictions *= bulk_features_avg

                inv_predictions_numpy = inv_predictions.numpy()

                # print("len(inv_predictions_numpy.shape) ", len(inv_predictions_numpy.shape))
                if len(inv_predictions_numpy.shape) == 1:
                    inv_predictions_numpy = np.expand_dims(inv_predictions_numpy, axis=0)

                if len(inv_predictions.numpy().shape) == 1:
                    dict_cond_tn_values.extend([list(voigt_to_tn(inv_predictions_numpy))])
                else:
                    dict_cond_tn_values.extend(list(voigt_to_tn(inv_predictions_numpy)))

                centers_tuples = map(tuple, centers.numpy())
                dict_centers.extend(centers_tuples)


        pred_cond_tensors = dict(zip(dict_centers, dict_cond_tn_values))

        return pred_cond_tensors

    @staticmethod
    def fine_SRF_from_homogenization(dfn, config, sample_seed):
        fine_step = config["fine"]["step"]
        coarse_step = config["coarse"]["step"]
        print("fine step: {}, coarse step: {}".format(fine_step, coarse_step))

        orig_domain_box = config["sim_config"]["geometry"]["orig_domain_box"]

        # There is no homogenization on the coarsest level
        # if coarse_step == 0:
        #     subdomain_box = [0, 0, 0]
        # else:
        # subdomain_box, n_nonoverlap_subdomains, n_subdomains_per_axes = DFMSim3D._calculate_subdomains(coarse_step,
        #                                                                                            config["sim_config"]["geometry"])
        # print("subdomain box ", subdomain_box)
        # print("orig domain box ", orig_domain_box)
        # print("n_nonoverlap_subdomains ", n_nonoverlap_subdomains)

        reversed_level_params = list(np.squeeze(config["sim_config"]["level_parameters"]))[::-1]
        print("reversed level params ", reversed_level_params)

        hom_block_sizes = np.array(reversed_level_params[1:reversed_level_params.index(fine_step)+1]) * 1.5

        #print("reversed_level_params[:reversed_level_params.index(fine_step)] ", reversed_level_params[:reversed_level_params.index(fine_step)])
        #print("np.sum(reversed_level_params[:reversed_level_params.index(fine_step)]) ", np.sum(reversed_level_params[:reversed_level_params.index(fine_step)]))

        larger_domain_size = orig_domain_box[0] + np.sum(hom_block_sizes) + np.sum(reversed_level_params[:reversed_level_params.index(fine_step)])  # Add fine samples to ensure larger domains for interpolations
        larger_domain_reduced_by_homogenization = larger_domain_size
        for hb_size in hom_block_sizes:
            larger_domain_reduced_by_homogenization -= hb_size

        print("new larger_domain_size ", larger_domain_size)
        print("larger_domain_reduced_by_homogenization ", larger_domain_reduced_by_homogenization)

        # To homogenize at this level - bulk conductivity has to be generated on larger domain size
        #geometry = {}
        larger_domain_box = [larger_domain_size, larger_domain_size, larger_domain_size]
        #geometry["pixel_stride_div"] = config["sim_config"]["geometry"]["pixel_stride_div"]

        dfn_fine = dfn
        for i in range(0, reversed_level_params.index(fine_step)):
            dfn_to_homogenization_list = []
            dfn_to_fine_list = []
            current_fine_step = reversed_level_params[i]
            current_coarse_step = reversed_level_params[i+1]

            print("current fine step: {}, current coarse step: {}".format(current_fine_step, current_coarse_step))
            for fr in dfn_fine:
                ### dfn for homogenization - to get fine sample SRF
                if fr.r >= current_fine_step and fr.r <= current_coarse_step:
                    #print("to hom fr.r ", fr.r)
                    dfn_to_homogenization_list.append(fr)
                else:
                    #print("coarse/new fine fr.r ", fr.r)
                    dfn_to_fine_list.append(fr)

            dfn_to_homogenization = stochastic.FractureSet.from_list(dfn_to_homogenization_list)
            dfn_fine = stochastic.FractureSet.from_list(dfn_to_fine_list)
            #coarse_fr_media = FracturedMedia.fracture_cond_params(dfn_to_coarse, 1e-4, 0.00001)

            # print("cond_larger_domain_size % (current_coarse_step*1.5) ", cond_larger_domain_size % (current_coarse_step*1.5))
            # while cond_larger_domain_size % (current_coarse_step*1.5) != 0:
            #     cond_larger_domain_size += 1
            #
            # geometry["orig_domain_box"] = [cond_larger_domain_size, cond_larger_domain_size, cond_larger_domain_size]
            # print("fine orig domain box ", orig_domain_box)

            #dfn = stochastic.FractureSet.from_list(dfn_to_fine_list)
            # Cubic law transmisivity

            #subdomain_box, n_nonoverlap_subdomains, n_subdomains_per_axes = DFMSim3D._calculate_subdomains(current_coarse_step, geometry)
            #print("HOM subdomain box ", subdomain_box)
            #print("HOM n nonoverlap_subdomain ", n_nonoverlap_subdomains)
            #print("HOM n_subdomains_per_axes ", n_subdomains_per_axes)

            hom_box_size = current_coarse_step * 1.5

            print("larger domain box size ", larger_domain_size)
            print("hom box size ", hom_box_size)

            #domain_box_size = geometry["orig_domain_box"][0]
            #n_centers = np.round(domain_box_size / hom_box_size)
            #n_total_centers = np.round(n_centers * geometry["pixel_stride_div"] + 1)
            hom_boxes_per_domain = larger_domain_size / hom_box_size
            print("hom_boxes_per_domain ", hom_boxes_per_domain)

            # Generate SRF on the finest level's resolution
            if i == 0:
                n_steps_cond_grid_size = int(hom_boxes_per_domain * 16)
                #print("n_steps_cond_grid_size ", n_steps_cond_grid_size)
                n_steps_cond_grid = (n_steps_cond_grid_size, n_steps_cond_grid_size, n_steps_cond_grid_size)

                fem_grid_cond = fem.fem_grid(n_steps_cond_grid_size, n_steps_cond_grid, fem.Fe.Q(dim=3),
                                             origin=-n_steps_cond_grid_size / 2)

                print("FEM GRID COND ", fem_grid_cond)

                bulk_cond_values, bulk_cond_points = DFMSim3D.generate_grid_cond(fem_grid_cond, config, seed=sample_seed)

            ##############
            # Upscaling  #
            ##############
            if "nn_path" in config["sim_config"]:
                n_voxels_per_hom_block = config["sim_config"]["geometry"]["n_voxels"][0]
                #print("n_voxels_per_hom_block ", n_voxels_per_hom_block)
                fem_grid_n_steps = [int(n_voxels_per_hom_block * hom_boxes_per_domain)] * 3
                #print("fem_grid_n_steps ", fem_grid_n_steps)
                fem_grid_rast = fem.fem_grid(larger_domain_size, fem_grid_n_steps, fem.Fe.Q(dim=3), origin=-larger_domain_size / 2)
                print("fem grid rast ", fem_grid_rast)
                ##@TODO: Set the most suitable Surrogate for prediction
                cond_tensors = DFMSim3D.rasterize_at_once_zarr_move_by_stride(config, dfn_to_homogenization,
                                                                              bulk_cond_values, bulk_cond_points,
                                                                              hom_box_size, larger_domain_size,
                                                                              fem_grid_rast)
                #points = np.array(list(cond_tensors.keys()))
                #values = np.squeeze(np.array(list(cond_tensors.values())))
                #print("points ", points)
                #print("values ", values)

            else:
                if config["sim_config"]["use_larger_domain"]:
                    config["sim_config"]["geometry"]["domain_box"] = hom_domain_box
                    config["sim_config"]["geometry"]["fractures_box"] = hom_domain_box

                cond_tensors, pred_cond_tensors_homo = DFMSim3D.homogenization(config, dfn_to_homogenization,
                                                                               dfn_to_homogenization_list,
                                                                               bulk_cond_values, bulk_cond_points)
            bulk_cond_values, bulk_cond_points = np.squeeze(
                np.array(list(cond_tensors.values()))), np.array(list(cond_tensors.keys()))

        fr_media = FracturedMedia.fracture_cond_params(dfn_fine, 1e-4, 0.00001)
        return fr_media, bulk_cond_values, bulk_cond_points


    @staticmethod
    def calculate(config, seed):
        """
        Method that actually run the calculation, it's called from mlmc.tool.pbs_job.PbsJob.calculate_samples()
        Calculate fine and coarse sample and also extract their results
        :param config: dictionary containing simulation configuration, LevelSimulation.config_dict (set in level_instance)
        :param seed: random seed, int
        :return: List[fine result, coarse result], both flatten arrays (see mlmc.sim.synth_simulation.calculate())
        """
        sample_dir = Path(os.getcwd()).absolute()
        basename = os.path.basename(sample_dir)
        sample_idx = int(basename.split('S')[1])

        print("os environ get SCRATCHDIR ", os.environ.get("SCRATCHDIR"))

        scratch_sample_path = None
        if os.environ.get("SCRATCHDIR") is not None:
            scratch_sample_path = os.path.join(os.environ.get("SCRATCHDIR"), basename)

            shutil.move(sample_dir, scratch_sample_path)
            os.chdir(scratch_sample_path)

        current_dir = Path(os.getcwd()).absolute()

        sample_seed = config["sim_config"]["seed"] + seed
        #print("sample seed ", sample_seed)
        np.random.seed(sample_seed)

        gen_hom_samples = False
        if "generate_hom_samples" in config["sim_config"] and config["sim_config"]["generate_hom_samples"]:
            gen_hom_samples = True

        if gen_hom_samples:
            return DFMSim3D.calculate_hom_sample(config, current_dir, sample_idx, sample_seed)

        fine_step = config["fine"]["step"]
        coarse_step = config["coarse"]["step"]

        domain_size = config["sim_config"]["geometry"]["domain_box"][0]  # 15  # 100
        fem_grid_cond_domain_size = config["sim_config"]["geometry"]["domain_box"][0] + 3 #+ int(config["sim_config"]["geometry"]["domain_box"][0] * 0.1)
        fr_domain_size = config["sim_config"]["geometry"]["fractures_box"][0]
        fr_range = config["sim_config"]["geometry"]["pow_law_sample_range"]  # (5, fr_domain_size)

        sim_config = config["sim_config"]
        geom = sim_config["geometry"]

        subdomain_box, n_nonoverlap_subdomains, n_subdomains_per_axes = DFMSim3D._calculate_subdomains(coarse_step, config["sim_config"]["geometry"])

        print("calculated subdomain box: {}, n subdomains per axes: {}, n_nonoverlap_subdomains: {}".format(subdomain_box, n_subdomains_per_axes, n_nonoverlap_subdomains))

        config["sim_config"]["geometry"]["domain_box"] = config["sim_config"]["geometry"]["orig_domain_box"]
        config["sim_config"]["geometry"]["subdomain_box"] = subdomain_box
        config["sim_config"]["geometry"]["n_subdomains_per_axes"] = n_subdomains_per_axes
        config["sim_config"]["geometry"]["n_subdomains"] = n_subdomains_per_axes ** 3

        #@TODO: use larger domain to calculate hom samples
        orig_domain_box = config["sim_config"]["geometry"]["orig_domain_box"]
        if coarse_step > 0 and "use_larger_domain" in config["sim_config"] and config["sim_config"][
            "use_larger_domain"]:

            sub_domain_box = config["sim_config"]["geometry"]["subdomain_box"]
            orig_frac_box = config["sim_config"]["geometry"]["fractures_box"]

            print("subdomain box ", subdomain_box)
            print("orig domain box ", orig_domain_box)

            # config["sim_config"]["geometry"]["domain_box"] = [orig_domain_box[0] + 2 * sub_domain_box[0],
            #                                                   orig_domain_box[1] + 2 * sub_domain_box[1],
            #                                                   orig_domain_box[1] + 2 * sub_domain_box[1]]

            config["sim_config"]["geometry"]["domain_box"] = [orig_domain_box[0] + sub_domain_box[0],
                                                              orig_domain_box[1] + sub_domain_box[1],
                                                              orig_domain_box[1] + sub_domain_box[1]]
            fem_grid_cond_domain_size = orig_domain_box[0] + sub_domain_box[0] + fine_step
            hom_domain_box = [orig_domain_box[0] + sub_domain_box[0], orig_domain_box[1] + sub_domain_box[1], orig_domain_box[1] + sub_domain_box[1]]
            # if config["sim_config"]["rasterize_at_once"]:
            #     hom_domain_box = [orig_domain_box[0] + sub_domain_box[0]/2, orig_domain_box[1] + sub_domain_box[1]/2]
            print("use larger domain domain box ", config["sim_config"]["geometry"]["domain_box"])

        # subdomain_box, n_nonoverlap_subdomains, n_subdomains_per_axes = DFMSim3D._calculate_subdomains(coarse_step, config["sim_config"]["geometry"])
        # print("subdomain_box ", subdomain_box)
        # print("n_nonoverlap_subdomains ", n_nonoverlap_subdomains)
        # print("n_subdomains_per_axes ", n_subdomains_per_axes)

        ###################
        ### Fine sample ###
        ###################
        fine_sample_start_time = time.time()
        dfn = DFMSim3D.fracture_random_set(sample_seed, fr_range, sim_config["work_dir"], max_frac=geom["n_frac_limit"])
        dfn_list = []
        for fr in dfn:
            if fr.r >= list(np.squeeze(config["sim_config"]["level_parameters"], axis=1))[-1] and fr.r <= orig_domain_box[0]:
                dfn_list.append(fr)
        dfn = stochastic.FractureSet.from_list(dfn_list)
        print("level parameters ", config["sim_config"]["level_parameters"])

        bulk_cond_values_start_time = time.time()
        # If the finest level
        if list(np.squeeze(config["sim_config"]["level_parameters"], axis=1)).index(config["fine"]["step"]) == (len(np.squeeze(config["sim_config"]["level_parameters"], axis=1)) - 1):
            dfn_to_fine_list = []
            for fr in dfn:
                if fr.r >= fine_step:
                    dfn_to_fine_list.append(fr)
            dfn = stochastic.FractureSet.from_list(dfn_to_fine_list)

            #n_steps = config["sim_config"]["geometry"]["n_voxels"]
            #print("n steps ", n_steps)
            fraction_factor = 2
            #@TODO: Think the calculation of n_steps over
            #n_steps = (int(n_steps[0] * n_nonoverlap_subdomains / fraction_factor), int(n_steps[1] * n_nonoverlap_subdomains / fraction_factor), int(n_steps[2] * n_nonoverlap_subdomains / fraction_factor))
            #print("n steps for all nonoverlaping subdomains / fraction factor ", n_steps)

            # Cubic law transmisivity
            fr_media = FracturedMedia.fracture_cond_params(dfn, 1e-4, 0.00001)
            fem_grid_cond_domain_size = int(fem_grid_cond_domain_size)
            # n_steps_cond_grid = (fem_grid_cond_domain_size, fem_grid_cond_domain_size, fem_grid_cond_domain_size)
            # 16x16x16 cond values generated per homogenization block
            n_steps_cond_grid = (
            n_nonoverlap_subdomains * 16, n_nonoverlap_subdomains * 16, n_nonoverlap_subdomains * 16)
            print("n steps cond grid ", n_steps_cond_grid)
            # fem_grid_cond = fem.fem_grid(fem_grid_cond_domain_size, n_steps_cond_grid, fem.Fe.Q(dim=3),
            #                             origin=-fem_grid_cond_domain_size / 2)  # 27 cells
            fem_grid_cond = fem.fem_grid(fem_grid_cond_domain_size, n_steps_cond_grid, fem.Fe.Q(dim=3),
                                         origin=-fem_grid_cond_domain_size / 2)
            print("fem grid cond ", fem_grid_cond)
            generate_grid_cond_start_time = time.time()
            bulk_cond_values, bulk_cond_points = DFMSim3D.generate_grid_cond(fem_grid_cond, config, seed=sample_seed)
            print("time_generate_grid_cond ", time.time() - generate_grid_cond_start_time)

            print("bulk cond values shape ", bulk_cond_values.shape)
            print("bulk cond points shape ", bulk_cond_points.shape)
        else:
            fr_media, bulk_cond_values, bulk_cond_points = DFMSim3D.fine_SRF_from_homogenization(dfn, config, sample_seed)

        print("bulk cond values time ", time.time() - bulk_cond_values_start_time)

        #######################
        ## Bulk conductivity ##
        #######################
        fr_rad_values = []
        total_area = 0
        total_excluded_volume = 0
        for fr in dfn:
            #print("fr.r ", fr.r)
            total_area += 4 * fr.r**2
            total_excluded_volume += np.pi * fr.r**3
            fr_rad_values.append(fr.radius[0] * (fr.radius[1] ** 2) + (fr.radius[0] ** 2) * fr.radius[1])

        # Square fractures:
        # print("Whole sample mean fr rad values ", np.mean(fr_rad_values))
        # rho_3D = np.pi ** 2 / 2 * np.mean(fr_rad_values) * (len(dfn) / fr_range[1] ** 3)
        # rho_3D_new = total_excluded_volume * (len(dfn) / fr_range[1] ** 3)
        # print("rho_3D ", rho_3D)
        # print("rho_3D_new ", rho_3D_new)
        # #exit()
        #print("bulk step: {}, fr step: {}".format(bulk_step, fr_step))

        dimensions = config["sim_config"]["geometry"]["orig_domain_box"] #config["sim_config"]["geometry"]["domain_box"]

        fine_res = [0, 0, 0, 0, 0, 0]
        print("fine sample dimensions ", dimensions)

        if not gen_hom_samples:
            sim_run_start_time = time.time()
            if "flow_sim" in config["sim_config"] and config["sim_config"]["flow_sim"]:
                bc_pressure_gradient = [1, 0, 0]
                cond_file, fr_cond, fr_region_map = DFMSim3D._run_sample_flow(bc_pressure_gradient, fr_media, config, current_dir, bulk_cond_values, bulk_cond_points, dimensions, mesh_step=config["fine"]["step"])

                conv_check = check_conv_reasons(os.path.join(current_dir, "flow123.0.log"))
                if not conv_check:
                    raise Exception("fine sample not converged")

                print("cond file ", cond_file)
                fine_res = DFMSim3D.get_outflow(current_dir)

                fine_res = [fine_res[0], fine_res[0], fine_res[0], fine_res[0], fine_res[0], fine_res[0]]
                print("fine res ", fine_res)
                if os.path.exists("flow_fields.pvd"):
                    shutil.move("flow_fields.pvd", "flow_field_fine.pvd")
                if os.path.exists("flow_fields"):
                    shutil.move("flow_fields", "flow_fields_fine")
                if os.path.exists("flow_fields.msh"):
                    shutil.move("flow_fields.msh", "flow_fields_fine.msh")
                if os.path.exists("water_balance.yaml"):
                    shutil.move("water_balance.yaml", "water_balance_fine.yaml")
                if os.path.exists("water_balance.txt"):
                    shutil.move("water_balance.txt", "water_balance_fine.txt")

                if os.path.exists("bulk_cond_data.npy"):
                    shutil.move("bulk_cond_data.npy", "fine_bulk_cond_data.npy")
                if os.path.exists("fr_cond_data.npy"):
                    shutil.move("fr_cond_data.npy", "fine_fr_cond_data.npy")
                if os.path.exists("voxel_fracture_sizes.npy"):
                    shutil.move("voxel_fracture_sizes.npy", "fine_voxel_fracture_sizes.npy")
                pass
            else:
                fine_res, fr_cond, fr_region_map = DFMSim3D.get_equivalent_cond_tn(fr_media, config, current_dir, bulk_cond_values, bulk_cond_points, dimensions, mesh_step=config["fine"]["step"])
                conv_check = check_conv_reasons(os.path.join(current_dir, "flow123.0.log"))
                if not conv_check:
                    raise Exception("fine sample not converged")
                pass

            print("fine sim run time ", time.time() - sim_run_start_time)

            if os.path.exists("flow123.0.log"):
                shutil.move("flow123.0.log", "fine_flow123.0.log")
            if os.path.exists("flow_upscale_templ.yaml_stderr"):
                shutil.move("flow_upscale_templ.yaml_stderr", "fine_flow_upscale_templ.yaml_stderr")
            if os.path.exists("flow_upscale_templ.yaml_stdout"):
                shutil.move("flow_upscale_templ.yaml_stdout", "fine_flow_upscale_templ.yaml_stdout")
            if os.path.exists("flow_upscale_templ.yaml.yaml"):
                shutil.move("flow_upscale_templ.yaml.yaml", "fine_flow_upscale_templ.yaml.yaml")
            if os.path.exists("homo_cube.brep"):
                shutil.move("homo_cube.brep", "fine_homo_cube.brep")
            if os.path.exists("homo_cube_healed.msh2"):
                shutil.move("homo_cube_healed.msh2", "fine_homo_cube_healed.msh2")
            if os.path.exists("homo_cube.heal_stats.yaml"):
                shutil.move("homo_cube.heal_stats.yaml", "fine_homo_cube.heal_stats.yaml")
            if os.path.exists("homo_cube.msh2"):
                shutil.move("homo_cube.msh2", "fine_homo_cube.msh2")
            if os.path.exists("input_fields.msh"):
                shutil.move("input_fields.msh", "fine_input_fields.msh")
            if os.path.exists("bulk_cond_data.npy"):
                shutil.move("bulk_cond_data.npy", "fine_bulk_cond_data.npy")
            if os.path.exists("fr_cond_data.npy"):
                shutil.move("fr_cond_data.npy", "fine_fr_cond_data.npy")
            if os.path.exists("voxel_fracture_sizes.npy"):
                shutil.move("voxel_fracture_sizes.npy", "fine_voxel_fracture_sizes.npy")

        print("fine sample time ", time.time() - fine_sample_start_time)

        #####################
        ### Coarse sample ###
        #####################
        coarse_sample_start_time = time.time()
        coarse_res = [0, 0, 0, 0, 0, 0]
        # print("fine_step", config["fine"]["step"])
        # print("coarse_step", config["coarse"]["step"])
        #
        # print("fr range ", fr_range)
        #print("n frac limit ", geom["n_frac_limit"])

        no_homogenization_flag = config["sim_config"]["no_homogenization_flag"] if "no_homogenization_flag" in config["sim_config"] else False
        pred_cond_tensors_homo = None

        # rasterize bulk for homogenization
        if coarse_step > 0:
            dfn_to_homogenization_list = []
            dfn_to_coarse_list = []
            for fr in dfn:
                if fr.r <= coarse_step:
                    dfn_to_homogenization_list.append(fr)
                else:
                    #print("coarse fr.r ", fr.r)
                    dfn_to_coarse_list.append(fr)
            dfn_to_homogenization = stochastic.FractureSet.from_list(dfn_to_homogenization_list)
            dfn_to_coarse = stochastic.FractureSet.from_list(dfn_to_coarse_list)
            coarse_fr_media = FracturedMedia.fracture_cond_params(dfn_to_coarse, 1e-4, 0.00001)

            # ######
            # ## ONLY for PLOT purposes
            # ## rasterize fractures to homogenization
            # ######
            # coarse_fr_media = FracturedMedia.fracture_cond_params(dfn_to_homogenization, 1e-4, 0.00001)
            # cond_file, fr_cond = DFMSim3D.create_mesh_fields(fr_media, bulk_cond_values, bulk_cond_points, dimensions,
            #                                                  mesh_step=config["fine"]["step"],
            #                                                  sample_dir=sample_dir,
            #                                                  work_dir=config["sim_config"]["work_dir"],
            #                                                  outflow_problem=True, file_prefix="hom_"
            #                                                  )

            print("len(dfn_to_homogenization_list) ", len(dfn_to_homogenization_list))
            print("len(dfn_to_coarse_list) ", len(dfn_to_coarse_list))

            homogenization_start_time = time.time()
            # Carry out homogenization
            if not no_homogenization_flag:
                if "nn_path" in config["sim_config"]:
                    domain_size = config["sim_config"]["geometry"]["domain_box"]

                    #print("n voxels ", config["sim_config"]["geometry"]["n_voxels"])

                    n_steps_per_axes = config["sim_config"]["geometry"]["n_voxels"][0]
                    fem_grid_n_steps = [n_steps_per_axes * n_nonoverlap_subdomains] * 3

                    #print("fem_grid_n_steps ", fem_grid_n_steps)

                    #print("domain size ", domain_size)

                    import cProfile
                    import pstats
                    pr = cProfile.Profile()
                    pr.enable()

                    fem_grid_rast = fem.fem_grid(domain_size, fem_grid_n_steps, fem.Fe.Q(dim=3), origin=-domain_size[0] / 2)
                    cond_tensors = DFMSim3D.rasterize_at_once_zarr(config, dfn_to_homogenization, bulk_cond_values,
                                                                   bulk_cond_points, fem_grid_rast, n_subdomains_per_axes)

                    # hom_box_size= config["sim_config"]["geometry"]["subdomain_box"][0]
                    # domain_box_size = config["sim_config"]["geometry"]["orig_domain_box"][0] + hom_box_size
                    # cond_tensors_moved_by_stride = DFMSim3D.rasterize_at_once_zarr_move_by_stride(config, dfn_to_homogenization, bulk_cond_values,
                    #                                                bulk_cond_points, hom_box_size, domain_box_size, fem_grid_rast)
                    #
                    # # Extract keys and values
                    # points1 = np.array(list(cond_tensors.keys()))
                    # values1 = np.squeeze(np.array(list(cond_tensors.values())))
                    #
                    # points2 = np.array(list(cond_tensors_moved_by_stride.keys()))
                    # values2 = np.squeeze(np.array(list(cond_tensors_moved_by_stride.values())))
                    #
                    # # Get sort indices
                    # idx1 = np.lexsort(points1.T)
                    # idx2 = np.lexsort(points2.T)
                    #
                    # # Sort
                    # points1_sorted, values1_sorted = points1[idx1], values1[idx1]
                    # points2_sorted, values2_sorted = points2[idx2], values2[idx2]
                    #
                    # same_points = np.array_equal(points1_sorted, points2_sorted)
                    # same_values = np.allclose(values1_sorted, values2_sorted)
                    #
                    # print("points1_sorted ", points1_sorted)
                    # print("points2_sorted ", points2_sorted)
                    #
                    # print("sample points: {}, same values: {}".format(same_points, same_values))


                    pr.disable()
                    ps = pstats.Stats(pr).sort_stats('cumtime')
                    ps.print_stats(25)

                else:
                    print("=== COARSE PROBLEM ===")
                    if config["sim_config"]["use_larger_domain"]:
                        # print("hom domain box ", hom_domain_box)
                        config["sim_config"]["geometry"]["domain_box"] = hom_domain_box
                        config["sim_config"]["geometry"]["fractures_box"] = hom_domain_box
                        # print("config geometry ", config["sim_config"]["geometry"])

                    import cProfile
                    import pstats
                    pr = cProfile.Profile()
                    pr.enable()


                    cond_tensors, pred_cond_tensors_homo = DFMSim3D.homogenization(config, dfn_to_homogenization,
                                                                                   dfn_to_homogenization_list,
                                                                                   bulk_cond_values, bulk_cond_points)

                    pr.disable()
                    ps = pstats.Stats(pr).sort_stats('cumtime')
                    ps.print_stats(35)

                    DFMSim3D._save_tensors(pred_cond_tensors_homo, file=os.path.join(current_dir, DFMSim3D.PRED_COND_TN_FILE))

                    pred_hom_bulk_cond_values, pred_hom_bulk_cond_points = np.squeeze(
                        np.array(list(pred_cond_tensors_homo.values()))), np.array(list(pred_cond_tensors_homo.keys()))

                    # print("cond tensors rast ", cond_tensors)
                    # print("cond tensors homo ", cond_tensors_homo)
                    # print("pred cond tensors homo ", pred_cond_tensors_homo)
                    #
                    # exit()
            # Do not carry out homogenization at all
            else:
                # Use fine sample grid conductivity directly for upscaled sample
                #cond_tensors = dict(zip(list(bulk_cond_points), list(bulk_cond_values)))
                cond_tensors = dict(zip(map(tuple, bulk_cond_points), bulk_cond_values))

            print("homogenization time ", time.time() - homogenization_start_time)

            DFMSim3D._save_tensors(cond_tensors, file=os.path.join(current_dir, DFMSim3D.COND_TN_FILE))

            hom_bulk_cond_values, hom_bulk_cond_points = np.squeeze(np.array(list(cond_tensors.values()))), np.array(list(cond_tensors.keys()))


            coarse_sim_start_time = time.time()
            file_prefix = "coarse_"
            if "flow_sim" in config["sim_config"] and config["sim_config"]["flow_sim"]:
                bc_pressure_gradient = [1, 0, 0]
                cond_file, fr_cond, fr_region_map = DFMSim3D._run_sample_flow(bc_pressure_gradient, coarse_fr_media, config, current_dir, hom_bulk_cond_values, hom_bulk_cond_points, dimensions, mesh_step=config["coarse"]["step"])
                coarse_res = DFMSim3D.get_outflow(current_dir)

                coarse_res = [coarse_res[0], coarse_res[0], coarse_res[0], coarse_res[0], coarse_res[0], coarse_res[0]]

                conv_check = check_conv_reasons(os.path.join(current_dir, "flow123.0.log"))
                if not conv_check:
                    raise Exception("coarse sample not converged")

                if pred_cond_tensors_homo:
                    file_prefix = "pred_coarse_"
                    if os.path.exists("flow123.0.log"):
                        shutil.move("flow123.0.log", "hom_coarse_flow123.0.log")
                    if os.path.exists("flow_fields.pvd"):
                        shutil.move("flow_fields.pvd", "hom_coarse_flow_field_coarse.pvd")
                    if os.path.exists("flow_fields"):
                        shutil.move("flow_fields", "hom_coarse_flow_fields_coarse")
                    if os.path.exists("flow_fields.msh"):
                        shutil.move("flow_fields.msh", "hom_coarse_flow_fields_coarse.msh")
                    if os.path.exists("input_fields.msh"):
                        shutil.move("input_fields.msh", "hom_coarse_input_fields.msh")
                    if os.path.exists("water_balance.yaml"):
                        shutil.move("water_balance.yaml", "hom_coarse_water_balance_coarse.yaml")
                    if os.path.exists("water_balance.txt"):
                        shutil.move("water_balance.txt", "hom_coarse_water_balance_coarse.txt")
                    if os.path.exists("bulk_cond_data.npy"):
                        shutil.move("bulk_cond_data.npy", "hom_coarse_bulk_cond_data.npy")
                    if os.path.exists("fr_cond_data.npy"):
                        shutil.move("fr_cond_data.npy", "hom_coarse_fr_cond_data.npy")
                    if os.path.exists("voxel_fracture_sizes.npy"):
                        shutil.move("voxel_fracture_sizes.npy", "hom_coarse_voxel_fracture_sizes.npy")

                    bc_pressure_gradient = [1, 0, 0]
                    pred_cond_file, pred_fr_cond, pred_fr_region_map = DFMSim3D._run_sample_flow(bc_pressure_gradient, coarse_fr_media,
                                                                             config,
                                                                             current_dir, pred_hom_bulk_cond_values,
                                                                             pred_hom_bulk_cond_points, dimensions,
                                                                             mesh_step=config["coarse"]["step"], fr_region_map=fr_region_map)
                    pred_coarse_res = DFMSim3D.get_outflow(current_dir)

                    pred_coarse_res = [pred_coarse_res[0], pred_coarse_res[0], pred_coarse_res[0], pred_coarse_res[0],
                                       pred_coarse_res[0],
                                       pred_coarse_res[0]]

                    conv_check = check_conv_reasons(os.path.join(current_dir, "flow123.0.log"))
                    if not conv_check:
                        raise Exception("coarse sample not converged")

                    fine_res = coarse_res
                    coarse_res = pred_coarse_res

                if os.path.exists("flow123.0.log"):
                    shutil.move("flow123.0.log", "{}flow123.0.log".format(file_prefix))
                if os.path.exists("flow_fields.pvd"):
                    shutil.move("flow_fields.pvd", "{}flow_field_coarse.pvd".format(file_prefix))
                if os.path.exists("flow_fields"):
                    shutil.move("flow_fields", "{}flow_fields_coarse".format(file_prefix))
                if os.path.exists("flow_fields.msh"):
                    shutil.move("flow_fields.msh", "{}flow_fields_coarse.msh".format(file_prefix))
                if os.path.exists("input_fields.msh"):
                    shutil.move("input_fields.msh", "{}input_fields.msh".format(file_prefix))
                if os.path.exists("water_balance.yaml"):
                    shutil.move("water_balance.yaml", "{}water_balance_coarse.yaml".format(file_prefix))
                if os.path.exists("water_balance.txt"):
                    shutil.move("water_balance.txt", "{}_water_balance_coarse.txt".format(file_prefix))
                if os.path.exists("bulk_cond_data.npy"):
                    shutil.move("bulk_cond_data.npy", "{}bulk_cond_data.npy".format(file_prefix))
                if os.path.exists("fr_cond_data.npy"):
                    shutil.move("fr_cond_data.npy", "{}fr_cond_data.npy".format(file_prefix))
                if os.path.exists("voxel_fracture_sizes.npy"):
                    shutil.move("voxel_fracture_sizes.npy", "{}voxel_fracture_sizes.npy".format(file_prefix))
            else:
                coarse_res, fr_cond_coarse, fr_region_map_coarse = DFMSim3D.get_equivalent_cond_tn(coarse_fr_media, config, current_dir, hom_bulk_cond_values, hom_bulk_cond_points, dimensions, mesh_step=config["coarse"]["step"])
                conv_check = check_conv_reasons(os.path.join(current_dir, "flow123.0.log"))
                if not conv_check:
                    raise Exception("coarse sample not converged")

                if pred_cond_tensors_homo:
                    if os.path.exists("flow123.0.log"):
                        shutil.move("flow123.0.log", "hom_coarse_flow123.0.log")
                    if os.path.exists("flow_fields.pvd"):
                        shutil.move("flow_fields.pvd", "hom_coarse_flow_field_coarse.pvd")
                    if os.path.exists("flow_fields"):
                        shutil.move("flow_fields", "hom_coarse_flow_fields_coarse")
                    if os.path.exists("flow_fields.msh"):
                        shutil.move("flow_fields.msh", "hom_coarse_flow_fields_coarse.msh")
                    if os.path.exists("input_fields.msh"):
                        shutil.move("input_fields.msh", "hom_coarse_input_fields.msh")
                    if os.path.exists("water_balance.yaml"):
                        shutil.move("water_balance.yaml", "hom_coarse_water_balance_coarse.yaml")
                    if os.path.exists("water_balance.txt"):
                        shutil.move("water_balance.txt", "hom_coarse_water_balance_coarse.txt")
                    if os.path.exists("bulk_cond_data.npy"):
                        shutil.move("bulk_cond_data.npy", "hom_coarse_bulk_cond_data.npy")
                    if os.path.exists("fr_cond_data.npy"):
                        shutil.move("fr_cond_data.npy", "hom_coarse_fr_cond_data.npy")
                    if os.path.exists("voxel_fracture_sizes.npy"):
                        shutil.move("voxel_fracture_sizes.npy", "hom_coarse_voxel_fracture_sizes.npy")

                    pred_coarse_res, pred_fr_cond_coarse, pred_fr_region_map_coarse = DFMSim3D.get_equivalent_cond_tn(coarse_fr_media, config,
                                                                                           current_dir,
                                                                                           pred_hom_bulk_cond_values,
                                                                                           pred_hom_bulk_cond_points,
                                                                                           dimensions,
                                                                                           mesh_step=
                                                                                           config["coarse"]["step"], fr_region_map=fr_region_map_coarse)
                    conv_check = check_conv_reasons(os.path.join(current_dir, "flow123.0.log"))
                    if not conv_check:
                        raise Exception("coarse sample not converged")

                    fine_res = coarse_res
                    coarse_res = pred_coarse_res

                    file_prefix = "pred_coarse_"

                if os.path.exists("flow123.0.log"):
                    shutil.move("flow123.0.log", "{}flow123.0.log".format(file_prefix))
                if os.path.exists("flow_fields.pvd"):
                    shutil.move("flow_fields.pvd", "{}flow_field_coarse.pvd".format(file_prefix))
                if os.path.exists("flow_fields"):
                    shutil.move("flow_fields", "{}flow_fields_coarse".format(file_prefix))
                if os.path.exists("flow_fields.msh"):
                    shutil.move("flow_fields.msh", "{}flow_fields_coarse.msh".format(file_prefix))
                if os.path.exists("input_fields.msh"):
                    shutil.move("input_fields.msh", "{}input_fields.msh".format(file_prefix))
                if os.path.exists("water_balance.yaml"):
                    shutil.move("water_balance.yaml", "{}water_balance_coarse.yaml".format(file_prefix))
                if os.path.exists("water_balance.txt"):
                    shutil.move("water_balance.txt", "{}_water_balance_coarse.txt".format(file_prefix))
                if os.path.exists("bulk_cond_data.npy"):
                    shutil.move("bulk_cond_data.npy", "{}bulk_cond_data.npy".format(file_prefix))
                if os.path.exists("fr_cond_data.npy"):
                    shutil.move("fr_cond_data.npy", "{}fr_cond_data.npy".format(file_prefix))
                if os.path.exists("voxel_fracture_sizes.npy"):
                    shutil.move("voxel_fracture_sizes.npy", "{}voxel_fracture_sizes.npy".format(file_prefix))

            print("coarse sim time ", time.time() - coarse_sim_start_time)

            print("coarse sample time ", time.time() - coarse_sample_start_time)

            print("fine res ", fine_res)
            print("coarse res ", coarse_res)

            if os.path.exists("flow_fields.pvd"):
                os.remove("flow_fields.pvd")
            if os.path.exists("flow_fields"):
                shutil.rmtree("flow_fields")

            if os.path.exists("flow_fields.msh"):
                shutil.move("flow_fields.msh", "flow_fields_fine.msh")
            if os.path.exists("mesh_fine.msh"):
                shutil.move("mesh_fine.msh", "mesh_fine_fine.msh")

        if scratch_sample_path is not None:
            shutil.move(scratch_sample_path, sample_dir)

        return fine_res, coarse_res

    @staticmethod
    def generate_grid_cond(fem_grid, config, seed):
        bulk_conductivity = config["sim_config"]['bulk_conductivity']
        if "marginal_distr" in bulk_conductivity and bulk_conductivity["marginal_distr"] is not False:
            means, cov = DFMSim3D.calculate_cov(bulk_conductivity["marginal_distr"])
            bulk_conductivity["mean_log_conductivity"] = means
            bulk_conductivity["cov_log_conductivity"] = cov
            del bulk_conductivity["marginal_distr"]

        if "gstools" in config["sim_config"] and config["sim_config"]["gstools"]:
            bulk_model = GSToolsBulk3D(**bulk_conductivity)
            bulk_model.seed = seed
            print("bulkfieldsgstools")
        else:
            raise NotImplementedError

        return bulk_model.generate_field(fem_grid)

    @staticmethod
    def calculate_cov(marginal_distrs):
        # @TODO: 3D case not supported yet
        corr_coeff = marginal_distrs["corr_coeff"]

        cov = np.zeros((marginal_distrs["n_marginals"], marginal_distrs["n_marginals"]))
        means = np.zeros((marginal_distrs["n_marginals"]))
        for i in range(marginal_distrs["n_marginals"]):
            if "marginal_{}".format(i) in marginal_distrs:
                cov[i, i] = marginal_distrs["marginal_{}".format(i)]["std_log_conductivity"] ** 2
                means[i] = marginal_distrs["marginal_{}".format(i)]["mean_log_conductivity"]

        cov[1, 0] = cov[0, 1] = corr_coeff * np.sqrt(cov[0, 0] * cov[1, 1])
        return means, cov


    @staticmethod
    def _save_tensors(cond_tensors, file):
        with open(file, "w") as f:
            yaml.dump(cond_tensors, f)


    # @staticmethod
    # def _run_homogenization_sample(flow_problem, config, format="gmsh"):
    #     """
    #            Create random fields file, call Flow123d and extract results
    #            :param fields_file: Path to file with random fields
    #            :param ele_ids: Element IDs in computational mesh
    #            :param fine_input_sample: fields: {'field_name' : values_array, ..}
    #            :param flow123d: Flow123d command
    #            :param common_files_dir: Directory with simulations common files (flow_input.yaml, )
    #            :return: simulation result, ndarray
    #            """
    #     # if homogenization:
    #     start_time = time.time()
    #     outer_reg_names = []
    #     for reg in flow_problem.side_regions:
    #         outer_reg_names.append(reg.name)
    #         outer_reg_names.append(reg.sub_reg.name)
    #
    #     base = flow_problem.basename
    #     base = base + "_hom"
    #     if format == "vtk":
    #         base += "_vtk"
    #     outer_regions_list = outer_reg_names
    #     # flow_args = config["flow123d"]
    #     n_steps = config["sim_config"]["n_pressure_loads"]
    #     t = np.pi * np.arange(0, n_steps) / n_steps
    #     p_loads = np.array([np.cos(t), np.sin(t)]).T
    #
    #     in_f = in_file(base)
    #     out_dir =base
    #     # n_loads = len(self.p_loads)
    #     # flow_in = "flow_{}.yaml".format(self.base)
    #     params = dict(
    #         mesh_file=mesh_file(flow_problem.basename),
    #         fields_file=fields_file(flow_problem.basename),
    #         outer_regions=str(outer_regions_list),
    #         n_steps=len(p_loads))
    #
    #     out_dir = os.getcwd()
    #
    #     common_files_dir = config["fine"]["common_files_dir"]
    #
    #     if format == "vtk":
    #         substitute_placeholders(os.path.join(common_files_dir, DFMSim3D.YAML_TEMPLATE_H_VTK), in_f, params)
    #         #print("os.path.join(common_files_dir, DFMSim3D.YAML_TEMPLATE_H_VTK) ", os.path.join(common_files_dir, DFMSim3D.YAML_TEMPLATE_H_VTK))
    #     else:
    #         substitute_placeholders(os.path.join(common_files_dir, DFMSim3D.YAML_TEMPLATE_H), in_f, params)
    #
    #     flow_args = ["docker", "run", "-v", "{}:{}".format(os.getcwd(), os.getcwd()), *config["flow123d"]]
    #     #flow_args = ["singularity", "exec", "/storage/liberec3-tul/home/martin_spetlik/flow_3_1_0.sif", "flow123d"]
    #
    #     flow_args.extend(['--output_dir', out_dir, os.path.join(out_dir, in_f)])
    #
    #     # print("out dir ", out_dir)
    #     print("flow_args ", flow_args)
    #
    #     # if os.path.exists(os.path.join(out_dir, "flow123.0.log")):
    #     #     os.remove(os.path.join(out_dir, "flow123.0.log"))
    #
    #     # if os.path.exists(os.path.join(out_dir, DFMSim3D.FIELDS_FILE)):
    #     #     return True
    #     with open(base + "_stdout", "w") as stdout:
    #         with open(base + "_stderr", "w") as stderr:
    #             completed = subprocess.run(flow_args, stdout=stdout, stderr=stderr)
    #         print("Exit status: ", completed.returncode)
    #         status = completed.returncode == 0
    #     print("run homogenization sample TIME: {}".format(time.time() - start_time))
    #     conv_check = DFMSim3D.check_conv_reasons(os.path.join(out_dir, "flow123.0.log"))
    #     print("converged: ", conv_check)
    #
    #     return status, p_loads, outer_reg_names, conv_check  # and conv_check
    #     # return  status, p_loads, outer_reg_names  # and conv_check
    @staticmethod
    def replace_region_all_in_yaml(input_file, output_file, new_regions):
        class Flow123dLoader(yaml.SafeLoader):
            pass

        def flow123d_tag_constructor(loader, tag_suffix, node):
            if isinstance(node, yaml.MappingNode):
                return loader.construct_mapping(node)
            elif isinstance(node, yaml.SequenceNode):
                return loader.construct_sequence(node)
            else:
                return loader.construct_scalar(node)

        Flow123dLoader.add_multi_constructor('!', flow123d_tag_constructor)

        # Custom dumper to represent lists in inline format
        class InlineListDumper(yaml.SafeDumper):
            pass

        def represent_inline_list(dumper, data):
            return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

        InlineListDumper.add_representer(list, represent_inline_list)

        # Load and modify the YAML
        with open(input_file, 'r') as f:
            data = yaml.load(f, Loader=Flow123dLoader)

        input_fields = data['problem']['flow_equation']['input_fields']
        for field in input_fields:
            if isinstance(field, dict) and field.get('region') == 'ALL':
                field['region'] = new_regions

        # Dump with inline list format
        with open(output_file, 'w') as f:
            yaml.dump(data, f, sort_keys=False, Dumper=InlineListDumper)

    @staticmethod
    def _run_sample_flow(bc_pressure_gradient, fr_media, config, sample_dir, bulk_cond_values, bulk_cond_points, dimensions,
                    mesh_step, cond_file=None, center=[0, 0, 0], fr_region_map=None):

        fr_cond = None
        if cond_file is None:
            cond_file, fr_cond, fr_region_map, mesh_regions = DFMSim3D.create_mesh_fields(fr_media, bulk_cond_values, bulk_cond_points, dimensions,
                                                             mesh_step=mesh_step,
                                                             sample_dir=sample_dir,
                                                             work_dir=config["sim_config"]["work_dir"],
                                                             center=center, outflow_problem=True, fr_region_map=fr_region_map
                                                             )

        if "run_local" in config["sim_config"] and config["sim_config"]["run_local"]:
            flow_cfg = dotdict(
                flow_executable=[
                    "docker",
                    "run",
                    "-v",
                    "{}:{}".format(os.getcwd(), os.getcwd()),
                    # "flow123d/ci-gnu:3.9.0_49fc60d4d",
                    "flow123d/ci-gnu:4.0.0a01_94c428",
                    # "flow123d/ci-gnu:4.0.3dev_71cc9c",
                    "flow123d",
                    #        - flow123d/endorse_ci:a785dd
                    #        - flow123d/ci-gnu:4.0.0a_d61969
                    # "dbg",
                    # "run",
                    "--output_dir",
                    os.getcwd()
                ],
                mesh_file=cond_file,
                pressure_grad=bc_pressure_gradient,
            )
        else:
            flow_cfg = dotdict(
                flow_executable=[
                    "singularity",
                    "exec",
                    # "-v",
                    # "{}:{}".format(os.getcwd(), os.getcwd()),
                    "/storage/liberec3-tul/home/martin_spetlik/flow_4_0_0.sif",
                    "flow123d",
                    #        - flow123d/endorse_ci:a785dd
                    #        - flow123d/ci-gnu:4.0.0a_d61969
                    # "dbg",
                    # "run",
                    "--output_dir",
                    os.getcwd()
                ],
                mesh_file=cond_file,
                pressure_grad=bc_pressure_gradient,
            )
        f_template = "01_conductivity_3D.yaml"

        shutil.copy(os.path.join(config["sim_config"]["work_dir"], f_template), sample_dir)
        flow_cfg["input_regions"] = [item for item in list(mesh_regions.values()) if item.startswith("fam") or item == "box"]

        with workdir_mng(sample_dir):
            flow_out = call_flow(flow_cfg, f_template, flow_cfg)

        # Project to target grid
        print(flow_out)

        return cond_file, fr_cond, fr_region_map

    @staticmethod
    def _run_sample(bc_pressure_gradient, fr_media, config, sample_dir, bulk_cond_values, bulk_cond_points, dimensions, mesh_step, cond_file=None, center=[0,0,0], fr_region_map=None):

        fr_cond = None
        if cond_file is None:
            cond_file, fr_cond, fr_region_map, mesh_regions = DFMSim3D.create_mesh_fields(fr_media, bulk_cond_values, bulk_cond_points, dimensions,
                                    mesh_step=mesh_step,
                                    sample_dir=sample_dir,
                                    work_dir=config["sim_config"]["work_dir"], center=center, fr_region_map=fr_region_map)

        if "run_local" in config["sim_config"] and config["sim_config"]["run_local"]:
            flow_cfg = dotdict(
                flow_executable=[
                    "docker",
                    "run",
                    "-v",
                    "{}:{}".format(os.getcwd(), os.getcwd()),
                    # "flow123d/ci-gnu:3.9.0_49fc60d4d",
                    "flow123d/ci-gnu:4.0.0a01_94c428",
                    # "flow123d/ci-gnu:4.0.3dev_71cc9c",
                    "flow123d",
                    #        - flow123d/endorse_ci:a785dd
                    #        - flow123d/ci-gnu:4.0.0a_d61969
                    # "dbg",
                    # "run",
                    "--output_dir",
                    os.getcwd()
                ],
                mesh_file=cond_file,
                pressure_grad=bc_pressure_gradient,
            )
        else:
            flow_cfg = dotdict(
                flow_executable=[
                    "singularity",
                    "exec",
                    # "-v",
                    # "{}:{}".format(os.getcwd(), os.getcwd()),
                    "/storage/liberec3-tul/home/martin_spetlik/flow_4_0_0.sif",
                    "flow123d",
                    #        - flow123d/endorse_ci:a785dd
                    #        - flow123d/ci-gnu:4.0.0a_d61969
                    # "dbg",
                    # "run",
                    "--output_dir",
                    os.getcwd()
                ],
                mesh_file=cond_file,
                pressure_grad=bc_pressure_gradient,
            )

        f_template = "flow_upscale_templ.yaml"
        shutil.copy(os.path.join(config["sim_config"]["work_dir"], f_template), sample_dir)

        #print("flow_cfg ", flow_cfg)
        with workdir_mng(sample_dir):
            flow_out = call_flow(flow_cfg, f_template, flow_cfg)

        # Project to target grid
        #print(flow_out)

        return cond_file, fr_cond, fr_region_map

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
                #if flux_in > 1e-10:
                #    raise Exception("Possitive inflow at outlet region.")
                total_flux += flux  # flux field
                found = True

        # Get flow123d computing time
        # run_time = FlowSim.get_run_time(sample_dir)

        if not found:
            raise Exception
        return np.array([-total_flux])

    @staticmethod
    def result_format() -> List[QuantitySpec]:
        """
        Define simulation result format
        :return: List[QuantitySpec, ...]
        """
        #spec1 = QuantitySpec(name="cond_tn", unit="m", shape=(1, 1), times=[1], locations=['0'])
        spec1 = QuantitySpec(name="cond_tn", unit="m", shape=(6, 1), times=[1], locations=['0'])

        return [spec1]

    # @staticmethod
    # def excluded_area(r_min, r_max, kappa, coef=1):
    #     norm_coef = kappa / (r_min ** (-kappa) - r_max ** (-kappa))
    #     return coef * (norm_coef * (r_max ** (1 - kappa) - r_min ** (1 - kappa)) / (1 - kappa)) ** 2
    #
    # @staticmethod
    # def calculate_mean_excluded_volume(r_min, r_max, kappa, geom=False):
    #     #print("r min ", r_min)
    #     #print("r max ", r_max)
    #     #print("kappa ", kappa)
    #     if geom:
    #         # return 0.5 * (kappa / (r_min**(-kappa) - r_max**(-kappa)))**2 * 2*(((r_max**(2-kappa) - r_min**(2-kappa))) * ((r_max**(1-kappa) - r_min**(1-kappa))))/(kappa**2 - 3*kappa + 2)
    #         return ((r_max ** (1.5 * kappa - 0.5) - r_min ** (1.5 * kappa - 0.5)) / (
    #                 (-1.5 * kappa - 0.5) * (r_min ** (-1 * kappa) - r_max ** (-1 * kappa)))) ** 2
    #     else:
    #         return 0.5 * (kappa / (r_min ** (-kappa) - r_max ** (-kappa))) ** 2 * 2 * (
    #                     ((r_max ** (2 - kappa) - r_min ** (2 - kappa))) * (
    #             (r_max ** (1 - kappa) - r_min ** (1 - kappa)))) / (kappa ** 2 - 3 * kappa + 2)
    #
    # @staticmethod
    # def calculate_mean_fracture_size(r_min, r_max, kappa, power=1):
    #     f0 = (r_min ** (-kappa) - r_max ** (-kappa)) / kappa
    #     return (1 / f0) * (r_max ** (-kappa + power) - r_min ** (-kappa + power)) / (-kappa + power)

    # @staticmethod
    # def generate_fractures(config):
    #     #print("config ", config)
    #     sim_config = config["sim_config"]
    #     geom = sim_config["geometry"]
    #     lx, ly = geom["fractures_box"]
    #     fr_size_range = geom["pow_law_size_range"]
    #     pow_law_exp_3d = geom["pow_law_size_exp"]
    #     pow_law_sample_range = geom["pow_law_sample_range"]
    #     rho = False
    #     rho_2D = False
    #     if "rho" in geom:
    #         rho = geom["rho"]  # P_30 * V_ex
    #     if "rho_2D" in geom:
    #         rho_2D = geom["rho_2D"]  # P_30 * V_ex
    #     n_frac_limit = geom["n_frac_limit"]
    #     #print("n frac limit ", n_frac_limit)
    #     #print("fr size range ", fr_size_range)
    #     p_32 = geom["p_32"]
    #
    #     #print("lx: {}, ly: {} ".format(lx, ly))
    #
    #     # generate fracture set
    #     fracture_box = [lx, ly, 0]
    #     area = lx * ly
    #
    #     #print("pow_law_sample_range ", pow_law_sample_range)
    #
    #     if rho_2D is not False:
    #         sim_config["fracture_model"]["max_fr"] = pow_law_sample_range[1]
    #         A_ex = BothSample.excluded_area(pow_law_sample_range[0], pow_law_sample_range[1],
    #                                         kappa=pow_law_exp_3d - 1, coef=np.pi / 2)
    #         #print("A_ex ", A_ex)
    #         # rho_2D = N_f/A * A_ex, N_f/A = intensity
    #         #print("rho_2D ", rho_2D)
    #
    #         intensity = rho_2D / A_ex
    #         print("intensity ", intensity)
    #
    #         pop = fracture.Population(area, fracture.LineShape)
    #         pop.add_family("all",
    #                        fracture.FisherOrientation(0, 90, np.inf),
    #                        fracture.VonMisesOrientation(0, 0),
    #                        fracture.PowerLawSize(pow_law_exp_3d - 1, pow_law_sample_range, intensity)
    #                        )
    #
    #         pop.set_sample_range(pow_law_sample_range)
    #
    #     elif rho is not False:
    #         V_ex = BothSample.calculate_mean_excluded_volume(r_min=fr_size_range[0],
    #                                                          r_max=fr_size_range[1],
    #                                                          kappa=pow_law_exp_3d - 1)
    #
    #         v_ex = BothSample.calculate_mean_fracture_size(r_min=fr_size_range[0],
    #                                                        r_max=fr_size_range[1],
    #                                                        kappa=pow_law_exp_3d - 1, power=1)
    #
    #         R2 = BothSample.calculate_mean_fracture_size(r_min=fr_size_range[0],
    #                                                      r_max=fr_size_range[1],
    #                                                      kappa=pow_law_exp_3d - 1, power=2)
    #
    #         #print("V_ex ", V_ex)
    #         #print("rho ", rho)
    #         p_30 = rho / V_ex
    #         #print("p_30 ", p_30)
    #
    #         #print("v_ex ", v_ex)
    #         #print("R2 ", R2)
    #
    #         p_30 = rho / (v_ex * R2)
    #         #print("final P_30 ", p_30)
    #
    #         pop = fracture.Population(area, fracture.LineShape)
    #         pop.add_family("all",
    #                        fracture.FisherOrientation(0, 90, np.inf),
    #                        fracture.VonMisesOrientation(0, 0),
    #                        fracture.PowerLawSize(pow_law_exp_3d - 1, fr_size_range, p_30)
    #                        )
    #
    #         pop.set_sample_range(pow_law_sample_range)
    #
    #     else:
    #         pop = fracture.Population(area, fracture.LineShape)
    #         pop.add_family("all",
    #                        fracture.FisherOrientation(0, 90, np.inf),
    #                        fracture.VonMisesOrientation(0, 0),
    #                        fracture.PowerLawSize.from_mean_area(pow_law_exp_3d - 1, fr_size_range, p_32, pow_law_exp_3d)
    #                        )
    #         if n_frac_limit is not False:
    #             # pop.set_sample_range([None, np.min([lx, ly, self.config_dict["geometry"]["fr_max_size"]])],
    #             #                      sample_size=n_frac_limit)
    #             # pop.set_sample_range([None, max(lx, ly)],
    #             #                      sample_size=n_frac_limit)
    #
    #             print("fr size range ", fr_size_range)
    #
    #             pop.set_sample_range(fr_size_range,
    #                                  sample_size=n_frac_limit)
    #         elif pow_law_sample_range:
    #             pop.set_sample_range(pow_law_sample_range)
    #
    #     #print("total mean size: ", pop.mean_size())
    #     #print("size range:", pop.families[0].size.sample_range)
    #
    #
    #     pos_gen = fracture.UniformBoxPosition(fracture_box)
    #     fractures = pop.sample(pos_distr=pos_gen, keep_nonempty=False)
    #
    #     print("fractures len ", len(fractures))
    #
    #     #print("fractures ", fractures)
    #     #print("fr_size_range[0] ", fr_size_range[0])
    #
    #     fr_set = fracture.Fractures(fractures, fr_size_range[0] / 2)
    #
    #     return fr_set

    def make_summary(done_list):
        results = {problem.basename: problem.summary() for problem in done_list}

        #print("results ", results)

        with open("summary.yaml", "w") as f:
            yaml.dump(results, f)
