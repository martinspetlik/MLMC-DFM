import os
import os.path
import subprocess
import numpy as np
import shutil
import time
import copy
import torch
import joblib
import ruamel.yaml as yaml
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
from metamodel.cnn.datasets.dfm_dataset import DFMDataset
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
from bgem.upscale.fields import voigt_to_tn
import zarr
from bgem.upscale.voxelize import fr_conductivity
from bgem.upscale import *
import scipy.interpolate as sc_interpolate


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

    @staticmethod
    def rasterize_at_once(sample_dir, dataset_path, config, n_subdomains, fractures, hom_dir, seed=None, fields_file="fields_fine_to_rast.msh", mesh_file="mesh_fine_to_rast.msh" ):
        n_non_overlapping_subdomains = n_subdomains - ((n_subdomains-1)/2)

        # import psutil
        # float_size_bytes = 8  # Assuming double precision (8 bytes)
        # available_memory = psutil.virtual_memory().available
        # print(f"Available memory: {available_memory} bytes")
        # max_floats = (available_memory // float_size_bytes) * 0.8
        max_floats = 10000000000 # 10 Gb

        #print("n subdomains ", n_subdomains)

        n_pixels = 256
        total_n_pixels_x = n_pixels**2 * n_non_overlapping_subdomains **2
        #total_n_pixels_x = 4**2
        print("total n pixels x ", total_n_pixels_x)
        split_exp = 0
        if total_n_pixels_x >= max_floats:
            split_exp = 1
            tot_pixels = total_n_pixels_x

            tot_pixels /= 4**split_exp
            while tot_pixels >= max_floats:
                split_exp += 1
                tot_pixels /= 4 ** split_exp

        if split_exp > 1:
            raise Exception("Total number of pixels: {} does not fit into memory. Not supported yet".format(total_n_pixels_x))

        sample_0_path = DFMSim3D.split_domain(config, dataset_path, n_split_subdomains=4**split_exp, fractures=fractures, sample_dir=sample_dir, seed=seed, fields_file=fields_file, mesh_file=mesh_file)

        print("dataset path ", dataset_path)

        ####################
        ## Create dataset ##
        ####################
        process = subprocess.run(["bash", config["sim_config"]["create_dataset_script"], dataset_path, "{}".format(int(np.sqrt(total_n_pixels_x)))],
                                 capture_output=True, text=True)
        pred_cond_tensors = {}

        if process.returncode == 0:
            bulk_path = os.path.join(sample_0_path, "bulk.npz")
            fractures_path = os.path.join(sample_0_path, "fractures.npz")
            cross_section_path = os.path.join(sample_0_path, "cross_sections.npz")
            bulk = np.load(bulk_path)["data"]
            fractures = np.load(fractures_path)["data"]
            cross_section = np.load(cross_section_path)["data"]
            domain_box = config["sim_config"]["geometry"]["orig_domain_box"]
            subdomain_box = config["sim_config"]["geometry"]["subdomain_box"]

            #print("bulk.shape ", bulk.shape)
            #print("fractures.shape ", fractures.shape)

            final_dataset_path = os.path.join(hom_dir, "final_dataset")
            os.mkdir(final_dataset_path)
            subdomain_size = 256
            sample_center = {}
            lx, ly = domain_box
            lx += subdomain_box[0]
            ly += subdomain_box[1]
            k = 0

            #print("subdomain box ", subdomain_box)

            # print("bulk[0, ...] ", bulk[0, ...])
            print("rasterize n subdomains ", n_subdomains)
            for i in range(n_subdomains):
                #print("lx ", lx)
                center_y = -(subdomain_box[0] / 2 + (lx - subdomain_box[0]) / (n_subdomains - 1) * i - lx / 2)
                for j in range(n_subdomains):
                    sample_name = "sample_{}".format(k)
                    print("sample name ", sample_name)
                    f_dset_sample_dir = os.path.join(final_dataset_path, sample_name)
                    os.mkdir(f_dset_sample_dir)

                    # print("subdomain box ", subdomain_box)
                    # exit()

                    center_x = (subdomain_box[1] / 2 + (lx - subdomain_box[1]) / (n_subdomains - 1) * j - lx / 2)

                    #print("center(i: {}, j:{}) x: {}, y: {}".format(i, j, center_x, center_y))
                    # print("x from: {}, to: {}".format(i*int((subdomain_size)/2), i*int((subdomain_size)/2) + subdomain_size))
                    # print("y from: {} to: {}".format(j*int((subdomain_size)/2), j*int((subdomain_size)/2) + subdomain_size))
                    # print("j*int((subdomain_size)/2) + subdomain_size ", j*int((subdomain_size)/2) + subdomain_size)

                    bulk_subdomain = bulk[:, i*int((subdomain_size)/2): i*int((subdomain_size)/2) + subdomain_size,
                                     j*int((subdomain_size)/2): j*int((subdomain_size)/2) + subdomain_size]
                    fractures_subdomain = fractures[:, i * int((subdomain_size) / 2): i * int((subdomain_size) / 2) + subdomain_size,
                                     j * int((subdomain_size) / 2): j * int((subdomain_size) / 2) + subdomain_size]
                    cross_section_subdomain = cross_section[:,
                                          i * int((subdomain_size) / 2): i * int((subdomain_size) / 2) + subdomain_size,
                                          j * int((subdomain_size) / 2): j * int((subdomain_size) / 2) + subdomain_size]

                    #print("bulk_subdomain[0,...] ", bulk_subdomain[0,...])

                    np.savez_compressed(os.path.join(f_dset_sample_dir, "bulk"), data=bulk_subdomain)
                    np.savez_compressed(os.path.join(f_dset_sample_dir, "fractures"), data=fractures_subdomain)
                    np.savez_compressed(os.path.join(f_dset_sample_dir, "cross_sections"), data=cross_section_subdomain)

                    #print("sample name ", sample_name)
                    sample_center[sample_name] = (center_x, center_y)
                    k += 1

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
            dataset_for_prediction = DFMDataset(data_dir=final_dataset_path,
                                                # output_file_name=output_file_name,
                                                init_transform=DFMSim3D.transform[0],
                                                input_transform=DFMSim3D.transform[1],
                                                output_transform=DFMSim3D.transform[2],
                                                two_dim=True,
                                                cross_section=True,
                                                # input_channels=config[
                                                #     "input_channels"] if "input_channels" in config else None,
                                                # output_channels=config[
                                                #     "output_channels"] if "output_channels" in config else None,
                                                # fractures_sep=config[
                                                #     "fractures_sep"] if "fractures_sep" in config else False,
                                                # vel_avg=config["vel_avg"] if "vel_avg" in config else False
                                                )

            dset_prediction_loader = torch.utils.data.DataLoader(dataset_for_prediction, batch_size=1, shuffle=False)
            with torch.no_grad():
                DFMSim3D.model.load_state_dict(DFMSim3D.checkpoint['best_model_state_dict'])
                DFMSim3D.model.eval()

                for i, sample in enumerate(dset_prediction_loader):
                    inputs, targets = sample
                    print("inputs ", inputs.shape)
                    inputs = inputs.float()
                    sample_n = dataset_for_prediction._bulk_file_paths[i].split('/')[-2]
                    center = sample_center[sample_n]
                    # if args.cuda and torch.cuda.is_available():
                    #    inputs = inputs.cuda()
                    predictions = DFMSim3D.model(inputs)
                    predictions = np.squeeze(predictions)

                    print("predictions ", predictions)

                    # if np.any(predictions < 0):
                    #     print("inputs ", inputs)
                    #     print("negative predictions ", predictions)

                    inv_predictions = torch.squeeze(
                        DFMSim3D.inverse_transform(torch.reshape(predictions, (*predictions.shape, 1, 1))))


                    #print("inv predictions shape ", inv_predictions.shape)


                    if dataset_for_prediction.init_transform is not None:
                        inv_predictions *= dataset_for_prediction._bulk_features_avg

                    pred_cond_tn = np.array([[inv_predictions[0], inv_predictions[1]],
                                             [inv_predictions[1], inv_predictions[2]]])

                    if pred_cond_tn is not None:
                        pred_cond_tn_flatten = pred_cond_tn.flatten()

                        if not np.any(np.isnan(pred_cond_tn_flatten)):
                            pred_cond_tensors[center] = [pred_cond_tn_flatten[0],
                                                         (pred_cond_tn_flatten[1] + pred_cond_tn_flatten[2]) / 2,
                                                         pred_cond_tn_flatten[3]]

                            #print("pred cond tn: {}".format(pred_cond_tensors[center]))
                            # if pred_cond_tn_flatten[0] < 0:
                            #     print("inputs ", inputs)

        else:
            raise Exception(process.stderr)

        return pred_cond_tensors

    @staticmethod
    def create_zarr_file(dir_path, n_samples, config_dict):
        input_shape_n_voxels = config_dict["sim_config"]["geometry"]["n_voxels"]
        input_shape_n_channels = 6  # 6 channels for cond_tn
        n_cond_tn_channels = 6  # 1 channel for cross_section
        # n_cross_section_channels = 1
        output_shape = (6,)

        zarr_file_path = os.path.join(dir_path, DFMSim3D.ZARR_FILE)

        if not os.path.exists(zarr_file_path):
            zarr_file = zarr.open(zarr_file_path, mode='w')

            # # Create the 'inputs' dataset with the specified shape
            inputs = zarr_file.create_dataset('inputs',
                                              shape=(n_samples,) + (input_shape_n_channels, *input_shape_n_voxels),
                                              dtype='float32',
                                              chunks=(1, n_cond_tn_channels, *input_shape_n_voxels),
                                              fill_value=0)
            # inputs[:, :, :, :, :] = np.zeros((n_samples, n_cond_tn_channels, *input_shape_n_voxels,
            #                                                         ))  # Populate the first 6 channels
            # inputs[:, :, :, :, n_cond_tn_channels] = np.random.rand(n_samples,
            #                                                         *input_shape_n_voxels)  # Populate the last channel

            # Create the 'outputs' dataset with the specified shape
            outputs = zarr_file.create_dataset('outputs', shape=(n_samples,) + output_shape, dtype='float32',
                                               chunks=(1, n_cond_tn_channels), fill_value=0)
            # outputs[:, :] = np.zeros((n_samples, n_cond_tn_channels))  # Populate the first 6 channels
            # outputs[:, n_cond_tn_channels] = np.random.rand(n_samples)  # Populate the last channel

            # Assign metadata to indicate channel names
            inputs.attrs['channel_names'] = ['cond_tn_0', 'cond_tn_1', 'cond_tn_2', 'cond_tn_3', 'cond_tn_4',
                                             'cond_tn_5']
            outputs.attrs['channel_names'] = ['cond_tn_0', 'cond_tn_1', 'cond_tn_2', 'cond_tn_3', 'cond_tn_4',
                                              'cond_tn_5']

        return zarr_file_path

    @staticmethod
    def homogenization(config, dfn, bulk_cond_values, bulk_cond_points, seed=None, hom_dir_name="homogenization"):
        # if "rasterize_at_once" in config["sim_config"] and config["sim_config"]["rasterize_at_once"]  and "nn_path" in config["sim_config"]:
        #     pred_cond_tensors = DFMSim3D.rasterize_at_once(sample_dir, dataset_path, config, n_subdomains, fractures, h_dir, seed=seed)
        #     #print("pred cond tensors ", pred_cond_tensors)
        # print("config ", config)
        sample_dir = os.getcwd()

        print("homogenization method")
        print("os.getcwd() ", os.getcwd())
        # print("config[scratch_dir] ", config["sim_config"]["scratch_dir"])
        print("os environ get SCRATCHDIR ", os.environ.get("SCRATCHDIR"))

        hom_dir_abs_path = os.path.join(sample_dir, hom_dir_name)
        # hom_dir_name = "homogenization"
        # if "scratch_dir" in config["sim_config"] and config["sim_config"]["scratch_dir"] is not None:
        #     # hom_dir_abs_path = os.path.join(config["sim_config"]["scratch_dir"], hom_dir_name)
        #     hom_dir_abs_path = os.path.join(os.environ.get("SCRATCHDIR"), hom_dir_name)
        # else:
        #     hom_dir_abs_path = os.path.join(sample_dir, hom_dir_name)

        print("hom_dir_abs_path ", hom_dir_abs_path)

        dfn_to_homogenization = []
        #fr_rad_values = []
        for fr in dfn:
            if fr.r <= config["coarse"]["step"]:
                #print("fr.radius ", fr.radius)
                #fr_rad_values.append(fr.radius[0]* fr.radius[1]**2 + fr.radius[0]**2*fr.radius[1])
                dfn_to_homogenization.append(fr)
        if len(dfn_to_homogenization) > 0:
            dfn = stochastic.FractureSet.from_list(dfn_to_homogenization)
        else:
            dfn = []
        #print("len dfn_to_homogenization ", len(dfn_to_homogenization))

        #print("mean fr rad values ", np.mean(fr_rad_values))

        fr_media = FracturedMedia.fracture_cond_params(dfn, 1e-4, 0.00001)

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

        fine_flow = None

        zarr_file_path = DFMSim3D.create_zarr_file(sample_dir, n_samples=int(n_subdomains_per_axes**3), config_dict=config)

        k = 0
        if "rasterize_at_once" in config["sim_config"] and config["sim_config"]["rasterize_at_once"]:
            fields_file = config["sim_config"]["fields_file_to_rast"] if "fields_file_to_rast" in config["sim_config"] else "fields_fine_to_rast.msh"
            mesh_file = config["sim_config"]["mesh_file_to_rast"] if "mesh_file_to_rast" in config["sim_config"] else "mesh_fine_to_rast.msh"

            pred_cond_tensors = DFMSim3D.rasterize_at_once(sample_dir, dataset_path, config, n_subdomains, fractures,
                                                         h_dir, seed=seed, fields_file=fields_file, mesh_file=mesh_file)
            # print("pred cond tensors ", pred_cond_tensors)
        else:
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

                        # center_x = 0.0
                        # center_y = -7.5
                        # center_z = -7.5
                        # k = 10
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

                        while True:
                            try:
                                # subdomain_box_run_samples[0] += subdomain_box_run_samples[0] * 0.1
                                # subdomain_box_run_samples[1] += subdomain_box_run_samples[1] * 0.1
                                # subdomain_box_run_samples[2] += subdomain_box_run_samples[2] * 0.1
                                print("center x:{} y:{}, z:{}, k: {}".format(center_x, center_y, center_z, k))

                                bc_pressure_gradient = [1, 0, 0]
                                cond_file, fr_cond = DFMSim3D._run_sample(bc_pressure_gradient, copy.deepcopy(fr_media), config, hom_sample_dir,
                                                                          bulk_cond_values, bulk_cond_points, subdomain_box_run_samples, center=[center_x, center_y, center_z])
                                flux_response_0 = DFMSim3D.get_flux_response()  # bc_pressure_gradient, fr_media, fem_grid, config, sample_dir,
                                # sim_config)

                                bc_pressure_gradient = [0, 1, 0]
                                DFMSim3D._run_sample(bc_pressure_gradient, copy.deepcopy(fr_media), config, hom_sample_dir, bulk_cond_values,
                                                     bulk_cond_points,
                                                     subdomain_box_run_samples, cond_file=cond_file, center=[center_x, center_y, center_z])
                                flux_response_1 = DFMSim3D.get_flux_response()  # bc_pressure_gradient, fr_media, fem_grid, config, sample_dir,
                                # sim_config)

                                bc_pressure_gradient = [0, 0, 1]
                                DFMSim3D._run_sample(bc_pressure_gradient, copy.deepcopy(fr_media), config, hom_sample_dir, bulk_cond_values,
                                                     bulk_cond_points,
                                                     subdomain_box_run_samples, cond_file=cond_file, center=[center_x, center_y, center_z])

                                flux_response_2 = DFMSim3D.get_flux_response()  # bc_pressure_gradient, fr_media, fem_grid, config, sample_dir,
                                # sim_config)

                                bc_pressure_gradients = np.stack(([1, 0, 0], [0, 1, 0], [0, 0, 1]), axis=0)
                                flux_responses = np.squeeze(
                                    np.stack((flux_response_0, flux_response_1, flux_response_2), axis=0))



                                # exit()
                                break

                            except:
                                # subdomain_box_run_samples[0] += subdomain_box_run_samples[0] * 0.05
                                # subdomain_box_run_samples[1] += subdomain_box_run_samples[1] * 0.05
                                # subdomain_box_run_samples[2] += subdomain_box_run_samples[2] * 0.05
                                #
                                current_dir = os.getcwd()
                                center_x += center_x * 0.05
                                center_y += center_y * 0.05
                                center_z += center_z * 0.05


                                # Loop through the files in the directory and delete them
                                for filename in os.listdir(current_dir):
                                    file_path = os.path.join(current_dir, filename)

                                    # Check if it's a file and delete it
                                    if os.path.isfile(file_path):
                                        os.remove(file_path)
                                        print(f"Deleted: {file_path}")

                        equivalent_cond_tn_voigt = equivalent_posdef_tensor(np.array(bc_pressure_gradients),
                                                                            flux_responses)

                        # print("equivalent_cond_tn_voigt ", equivalent_cond_tn_voigt)

                        equivalent_cond_tn = voigt_to_tn(np.array([equivalent_cond_tn_voigt]))  # np.zeros((3, 3))
                        print("equivalent cond tn ", equivalent_cond_tn)
                        evals, evecs = np.linalg.eigh(equivalent_cond_tn)
                        print("evals equivalent cond tn ", evals)
                        assert np.all(evals) > 0

                        fem_grid_rast = fem.fem_grid(15, config["sim_config"]["geometry"]["n_voxels"], fem.Fe.Q(dim=3), origin=[center_x-subdomain_box[0]/2, center_y-subdomain_box[0]/2, center_z-subdomain_box[0]/2])

                        print("fem_grid_rast.grid ", fem_grid_rast.grid)
                        print("fem_grid_rast.grid center ", fem_grid_rast.grid.grid_center())

                        # bulk_cond_fem_rast = DFMSim3D._bulk_cond_to_rast_grid(bulk_cond_values, bulk_cond_points,
                        #                                                       fem_grid_rast.grid)

                        #rasterized_input = DFMSim3D.rasterize(fem_grid_rast, dfn, bulk_cond=bulk_cond_fem_rast,
                        #                                      fr_cond=fr_cond)

                        #rasterized_input_voigt = tn_to_voigt(rasterized_input)
                        #rasterized_input_voigt = rasterized_input_voigt.reshape(*config["sim_config"]["geometry"]["n_voxels"],
                        #                                                        rasterized_input_voigt.shape[-1]).T

                        fine_res = np.squeeze(equivalent_cond_tn_voigt)

                        DFMSim3D.rasterize_save_to_zarr(zarr_file_path, config, k, fine_res, bulk_cond_values,
                                                        bulk_cond_points, dfn, fr_cond,
                                                        fem_grid_rast, n_steps=config["sim_config"]["geometry"]["n_voxels"])

                        #np.save("equivalent_cond_tn", equivalent_cond_tn)
                        #np.savez_compressed("rasterized_input_voigt ", rasterized_input_voigt)

                        # if os.path.exists(zarr_file_path):
                        #     zarr_file = zarr.open(zarr_file_path, mode='r+')
                        #
                        #     zarr_file["inputs"][k, ...] = rasterized_input_voigt
                        #     zarr_file["outputs"][k, :] = fine_res

                        os.chdir(h_dir)



        try:
            shutil.move(h_dir, sample_dir)
        except:
            pass

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
    def fracture_random_set(seed, size_range, work_dir, max_frac=1e21):
        #script_dir = Path(__file__).absolute().parent
        rmin, rmax = size_range
        box_dimensions = (rmax, rmax, rmax)
        fr_cfg_path = os.path.join(work_dir, "fractures_conf.yaml")

        # with open() as f:
        #    pop_cfg = yaml.load(f, Loader=yaml.SafeLoader)
        fr_pop = stochastic.Population.from_cfg(fr_cfg_path, box_dimensions, shape=stochastic.EllipseShape())
        if fr_pop.mean_size() > max_frac:
            common_range = fr_pop.common_range_for_sample_size(sample_size=max_frac)
            print("common range ", common_range)
            fr_pop = fr_pop.set_sample_range(common_range)
        print(f"fr set range: {[rmin, rmax]}, fr_lim: {max_frac}, mean population size: {fr_pop.mean_size()}")
        pos_gen = stochastic.UniformBoxPosition(fr_pop.domain)
        np.random.seed(seed)
        fractures = fr_pop.sample(pos_distr=pos_gen, keep_nonempty=True)

        # for fr in fractures:
        #    fr.region = gmsh.Region.get("fr", 2)
        return fractures


    @staticmethod
    def homo_decovalex(fr_media: FracturedMedia, grid: fem.Grid):
        """
        Homogenize fr_media to the conductivity tensor field on grid.
        :return: conductivity_field, np.array, shape (n_elements, n_voight)
        """
        ellipses = [dmap.Ellipse(fr.normal, fr.center, fr.scale) for fr in fr_media.dfn]
        d_grid = dmap.Grid.make_grid(grid.origin, grid.step, grid.dimensions)
        fractures = dmap.map_dfn(d_grid, ellipses)
        fr_transmissivity = fr_media.fr_conductivity * fr_media.fr_cross_section
        k_iso_zyx = dmap.permIso(d_grid, fractures, fr_transmissivity, fr_media.conductivity)
        k_iso_xyz = grid.cell_field_C_like(k_iso_zyx)
        k_voigt = k_iso_xyz[:, None] * np.array([1, 1, 1, 0, 0, 0])[None, :]
        return k_voigt

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
            #print("fr: ", i, "tag: ", shape.dim_tags, "fr.r: ", fr.r)
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
    def ref_solution_mesh(work_dir, domain_dimensions, fractures, fr_step, bulk_step, center):
        factory = gmsh.GeometryOCC("homo_cube", verbose=False)
        gopt = options.Geometry()
        gopt.Tolerance = 0.0001
        gopt.ToleranceBoolean = 0.001

        print("domain dimensions ", domain_dimensions)

        box = factory.box(domain_dimensions, center)
        #fr_box = factory.box([15,15,15], center)

        fractures, fr_region_map = DFMSim3D.create_fractures_rectangles(factory, fractures, factory.rectangle())

        fractures_group = factory.group(*fractures).intersect(box)
        box_fr, fractures_fr = factory.fragment(box, fractures_group)
        print("box fr ", box_fr)
        print("fracture fr ", fractures_fr)

        print("bulk step: {}, fr step: {}".format(bulk_step, fr_step))

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

        print("mesh file name ", mesh_file)
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

            #print("fr values shape ", fr_values.shape)
            #print("bulk cond tn shape ", bulk_cond_tn.shape)

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

    @staticmethod
    def reference_solution(fr_media: FracturedMedia, dimensions, bc_gradients, mesh_step, sample_dir, work_dir):
        dfn = fr_media.dfn
        bulk_conductivity = fr_media.conductivity

        # Input crssection and conductivity
        print("mesh step ", mesh_step)
        mesh_file, fr_region_map = DFMSim3D.ref_solution_mesh(sample_dir, dimensions, dfn, fr_step=mesh_step, bulk_step=mesh_step)
        print("mesh file ", mesh_file)
        full_mesh = Mesh.load_mesh(mesh_file, heal_tol=0.001)  # gamma

        fr_cond, fr_map = DFMSim3D.fr_field(full_mesh, dfn, fr_region_map, fr_media.fr_conductivity, bulk_conductivity,
                          rnd_cond=False, field_dim=1)

        fr_cross_section, fr_map = DFMSim3D.fr_field(full_mesh, dfn, fr_region_map, fr_media.fr_cross_section, 1.0, field_dim=1)

        print("fr cond ", fr_cond)
        #print("fr el ids ", fr_el_ids)

        fields = dict(
            conductivity=fr_cond,
            cross_section=fr_cross_section)

        cond_file = full_mesh.write_fields(str(sample_dir / "input_fields.msh2"), fields)
        print("cond_file ", cond_file)
        cond_file = Path(cond_file)
        cond_file = cond_file.rename(cond_file.with_suffix(".msh"))

        print("final cond file ", cond_file)

        flow_cfg = dotdict(
            flow_executable=[
                "docker",
                "run",
                "-v",
                "{}:{}".format(os.getcwd(), os.getcwd()),
                "flow123d/ci-gnu:4.0.0a01_94c428",
                "flow123d",
                #        - flow123d/endorse_ci:a785dd
                #        - flow123d/ci-gnu:4.0.0a_d61969
                # "dbg",
                # "run",
                "--output_dir",
                os.getcwd()
            ],
            mesh_file=cond_file,
            pressure_grad=bc_gradients,
            # pressure_grad_0=bc_gradients[0],
            # pressure_grad_1=bc_gradients[1],
            # pressure_grad_2=bc_gradients[2]
        )
        f_template = "flow_upscale_templ.yaml"
        shutil.copy(os.path.join(work_dir, f_template), sample_dir)
        with workdir_mng(sample_dir):
            flow_out = call_flow(flow_cfg, f_template, flow_cfg)

        # Project to target grid
        print(flow_out)
        # vel_p0 = velocity_p0(target_grid, flow_out)
        # projection of fields
        return flow_out


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

            AB = nodes[1] - nodes[0]
            AC = nodes[2] - nodes[0]
            AD = nodes[3] - nodes[0]

            # Compute the cross product of vectors AC and AD
            cross_product = np.cross(AC, AD)

            # Compute the dot product of AB with the cross product
            dot_product = np.dot(AB, cross_product)

            # The volume is 1/6 of the absolute value of the dot product
            volume = abs(dot_product) / 6.0

            return volume

        else:
            assert False


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
    def rasterize(fem_grid, dfn, bulk_cond, fr_cond):
        #steps = 3 * [41]
        target_grid = fem_grid.grid #Grid(3 * [15], steps, origin=3 * [-7.5])  # test grid with center in (0,0,0)
        isec_corners = intersection_cell_corners(dfn, target_grid)
        # isec_probe = probe_fr_intersection(fr_set, target_grid)
        #cross_section, fr_cond = fr_conductivity(dfn)
        rasterized = isec_corners.interpolate(bulk_cond, fr_cond, source_grid=fem_grid.grid)
        DFMSim3D.plot_isec_fields2(isec_corners, bulk_cond, rasterized, "raster_field.vtk")

        for i_ax in range(3):
            assert np.all(bulk_cond[:, i_ax, i_ax] <= rasterized[:, i_ax, i_ax])
        for i_ax in range(3):
            assert np.all(rasterized[:, i_ax, i_ax].max() <= fr_cond[:, i_ax, i_ax].max())

        return rasterized

    @staticmethod
    def create_mesh_fields(fr_media, bulk_cond_values, bulk_cond_points, dimensions, mesh_step, sample_dir, work_dir, center=[0, 0, 0]):
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
            fr_cross_section, fr_cond = fr_conductivity(dfn, cross_section_factor=1e-4)

        print("bulk cond points.shape ", bulk_cond_points.shape)


        mesh_file, fr_region_map = DFMSim3D.ref_solution_mesh(sample_dir, dimensions, dfn, fr_step=mesh_step,
                                                              bulk_step=mesh_step, center=center)

        full_mesh = Mesh.load_mesh(mesh_file, heal_tol=0.001)

        #if len(dfn) > 0:
        fr_cond_tn, fr_map = DFMSim3D.fr_field(full_mesh, dfn, fr_region_map, fr_cond,
                                                   bulk_conductivity, rnd_cond=False, field_dim=3)

        cross_sections, fr_map_cs = DFMSim3D.fr_field(full_mesh, dfn, fr_region_map, fr_cross_section, 1.0, field_dim=1)
        cross_sections = np.array(cross_sections)

        #######################################
        ## Interpolate SRF to mesh elements  ##
        #######################################
        bulk_elements_barycenters = full_mesh.el_barycenters(elements=full_mesh._bulk_elements)
        print("bulk_elements_barycenters ", bulk_elements_barycenters.shape)
        full_mesh_bulk_cond_values = interp(bulk_elements_barycenters)

        print("full_mesh_bulk_cond_values ", full_mesh_bulk_cond_values.shape)

        zero_rows = np.where(np.all(full_mesh_bulk_cond_values == 0, axis=1))[0]
        assert len(zero_rows) == 0

        ##################
        ## Write fields ##
        ##################
        fr_cond_tn[-full_mesh_bulk_cond_values.shape[0]:, ...] = full_mesh_bulk_cond_values
        conductivity = fr_cond_tn.reshape(fr_cond_tn.shape[0], 9)

        fields = dict(conductivity=conductivity, cross_section=cross_sections.reshape(-1, 1))

        cond_file = full_mesh.write_fields(str(sample_dir / "input_fields.msh2"), fields)
        cond_file = Path(cond_file)
        cond_file = cond_file.rename(cond_file.with_suffix(".msh"))

        return cond_file, fr_cond


    @staticmethod
    def calculate_hom_sample(config, sample_dir, sample_idx, sample_seed):
        domain_size = 15  # 15  # 100
        fem_grid_cond_domain_size = 16
        fr_domain_size = 100
        fr_range = (5, fr_domain_size) #(5, fr_domain_size)
        coarse_step = 10 #config["coarse"]["step"]

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

            for fr in dfn:
                #print("fr.r ", fr.r)
                if fr.r <= coarse_step:
                    dfn_to_homogenization.append(fr)
            #print("len dfn_to_homogenization ", len(dfn_to_homogenization))

            dfn = stochastic.FractureSet.from_list(dfn_to_homogenization)

        # dfn = dfn_to_homogenization
        ########################
        ########################
        ########################

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

        fem_grid_cond = fem.fem_grid(fem_grid_cond_domain_size, n_steps_cond_grid, fem.Fe.Q(dim=3),
                                     origin=-fem_grid_cond_domain_size / 2)  # 27 cells
        fem_grid_rast = fem.fem_grid(domain_size, n_steps, fem.Fe.Q(dim=3),
                                     origin=-domain_size / 2)  # 27 cells


        #######################
        ## Bulk conductivity ##
        #######################
        bulk_cond_values, bulk_cond_points = DFMSim3D.generate_grid_cond(fem_grid_cond, config, seed=sample_seed)

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
                                                  bulk_cond_points, dimensions)
        flux_response_0 = DFMSim3D.get_flux_response()  # bc_pressure_gradient, fr_media, fem_grid, config, sample_dir,
        # sim_config)

        bc_pressure_gradient = [0, 1, 0]
        DFMSim3D._run_sample(bc_pressure_gradient, fr_media, config, sample_dir, bulk_cond_values, bulk_cond_points,
                             dimensions, cond_file=cond_file)
        flux_response_1 = DFMSim3D.get_flux_response()  # bc_pressure_gradient, fr_media, fem_grid, config, sample_dir,
        # sim_config)

        bc_pressure_gradient = [0, 0, 1]
        DFMSim3D._run_sample(bc_pressure_gradient, fr_media, config, sample_dir, bulk_cond_values, bulk_cond_points,
                             dimensions, cond_file=cond_file)
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

        #print("fine res shape ", fine_res.shape)

        DFMSim3D._remove_files()

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
        DFMSim3D.rasterize_save_to_zarr(zarr_file_path, config, sample_idx, fine_res, bulk_cond_values, bulk_cond_points, dfn, fr_cond,
                               fem_grid_rast, n_steps)

        # zarr_file_path = os.path.join(config["fine"]["common_files_dir"], DFMSim3D.ZARR_FILE)
        # if os.path.exists(zarr_file_path):
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
    def rasterize_save_to_zarr(zarr_file_path, config, sample_idx, fine_res, bulk_cond_values, bulk_cond_points, dfn, fr_cond, fem_grid_rast, n_steps):
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
            zarr_file["outputs"][sample_idx, :] = fine_res

    @staticmethod
    def _bulk_cond_to_rast_grid(bulk_cond_values, bulk_cond_points, grid_rast):
        from scipy.interpolate import griddata
        # Interpolate the scattered data onto the regular grid
        resized_data = griddata(bulk_cond_points, bulk_cond_values, grid_rast.barycenters(), method='nearest')
        return resized_data

    @staticmethod
    def _calculate_subdomains(coarse_step, geometry):
        if coarse_step > 0:
            hom_box_size = coarse_step * 1.5
            domain_box_size = geometry["orig_domain_box"][0]
            n_centers = np.round(domain_box_size / hom_box_size)
            n_total_centers = np.round(n_centers * 2 + 1)
            hom_box_size = domain_box_size / n_centers
        else:
            return geometry["subdomain_box"], 4, 4

        return [hom_box_size, hom_box_size, hom_box_size], int(n_centers), int(n_total_centers)

    @staticmethod
    def calculate(config, seed):
        """
        Method that actually run the calculation, it's called from mlmc.tool.pbs_job.PbsJob.calculate_samples()
        Calculate fine and coarse sample and also extract their results
        :param config: dictionary containing simulation configuration, LevelSimulation.config_dict (set in level_instance)
        :param seed: random seed, int
        :return: List[fine result, coarse result], both flatten arrays (see mlmc.sim.synth_simulation.calculate())
        """
        from bgem.upscale import fem_plot, fem, voigt_to_tn, tn_to_voigt, FracturedMedia, voxelize

        sample_dir = Path(os.getcwd()).absolute()
        #print("sample dir ", sample_dir)

        basename = os.path.basename(sample_dir)
        sample_idx = int(basename.split('S')[1])

        if "scratch_dir" in config and os.path.exists(config["scratch_dir"]):
            sample_dir = os.path.join(config["scratch_dir"], basename)

        print("sample dir ", sample_dir)
        #print("sample idx ", sample_idx)

        sample_seed = config["sim_config"]["seed"] + seed
        print("sample seed ", sample_seed)
        np.random.seed(sample_seed)

        gen_hom_samples = False
        if "generate_hom_samples" in config["sim_config"] and config["sim_config"]["generate_hom_samples"]:
            gen_hom_samples = True

        gen_hom_samples_split = False
        if "generate_hom_samples_split" in config["sim_config"] and config["sim_config"]["generate_hom_samples_split"]:
            gen_hom_samples_split = True

        if gen_hom_samples:
            return DFMSim3D.calculate_hom_sample(config, sample_dir, sample_idx, sample_seed)

        fine_step = config["fine"]["step"]
        coarse_step = config["coarse"]["step"]

        domain_size = config["sim_config"]["geometry"]["domain_box"][0]  # 15  # 100
        fem_grid_cond_domain_size = config["sim_config"]["geometry"]["domain_box"][0] + fine_step
        fr_domain_size = config["sim_config"]["geometry"]["fractures_box"][0]
        fr_range = config["sim_config"]["geometry"]["pow_law_sample_range"]  # (5, fr_domain_size)

        sim_config = config["sim_config"]
        geom = sim_config["geometry"]

        subdomain_box, n_nonoverlap_subdomains, n_subdomains_per_axes = DFMSim3D._calculate_subdomains(coarse_step,
                                                                   config["sim_config"]["geometry"])

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
        dfn = DFMSim3D.fracture_random_set(sample_seed, fr_range, sim_config["work_dir"], max_frac=geom["n_frac_limit"])

        n_steps = config["sim_config"]["geometry"]["n_voxels"]
        print("n steps ", n_steps)

        fraction_factor = 2

        #@TODO: Think the calculation of n_steps over
        #n_steps = (int(n_steps[0] * n_nonoverlap_subdomains / fraction_factor), int(n_steps[1] * n_nonoverlap_subdomains / fraction_factor), int(n_steps[2] * n_nonoverlap_subdomains / fraction_factor))

        #print("n steps for all nonoverlaping subdomains / fraction factor ", n_steps)

        # Cubic law transmissvity
        fr_media = FracturedMedia.fracture_cond_params(dfn, 1e-4, 0.00001)

        fem_grid_cond_domain_size = int(fem_grid_cond_domain_size)

        n_steps_cond_grid = (fem_grid_cond_domain_size, fem_grid_cond_domain_size, fem_grid_cond_domain_size)

        #n_steps_cond_grid = (2,2,2)

        print("n steps cond grid ", n_steps_cond_grid)

        #fem_grid_cond = fem.fem_grid(fem_grid_cond_domain_size, n_steps_cond_grid, fem.Fe.Q(dim=3),
        #                             origin=-fem_grid_cond_domain_size / 2)  # 27 cells

        fem_grid_cond = fem.fem_grid(fem_grid_cond_domain_size, n_steps_cond_grid, fem.Fe.Q(dim=3), origin=-fem_grid_cond_domain_size / 2)
        fem_grid_rast = fem.fem_grid(domain_size, n_steps, fem.Fe.Q(dim=3), origin=-domain_size / 2)

        #######################
        ## Bulk conductivity ##
        #######################

        #print("bulk step: {}, fr step: {}".format(bulk_step, fr_step))

        bulk_cond_values, bulk_cond_points = DFMSim3D.generate_grid_cond(fem_grid_cond, config, seed=sample_seed)

        print("bulk cond values shape ", bulk_cond_values.shape)
        print("bulk cond points shape ", bulk_cond_points.shape)

        #####################
        ### Coarse sample ###
        #####################
        print("fine_step", config["fine"]["step"])
        print("coarse_step", config["coarse"]["step"])

        print("fr range ", fr_range)
        #print("n frac limit ", geom["n_frac_limit"])

        fr_rad_values = []
        for fr in dfn:
            print("fr.r ", fr.r)
            fr_rad_values.append(fr.radius[0] * fr.radius[1] ** 2 + fr.radius[0] ** 2 * fr.radius[1])

        print("Whole sample mean fr rad values ", np.mean(fr_rad_values))
        rho_3D = np.pi**2/2 * np.mean(fr_rad_values) * (len(dfn)/fr_range[1]**3)
        print("rho_3D ", rho_3D)
        #     if fr.r <= coarse_step:
        #         dfn_to_homogenization.append(fr)
        # print("len dfn_to_homogenization ", len(dfn_to_homogenization))

        #dfn = stochastic.FractureSet.from_list(dfn_to_homogenization)

        #dfn = dfn_to_homogenization
        ########################
        ########################
        ########################
        #### Homogenization ####
        ########################
        fine_res, coarse_res = [], []

        if gen_hom_samples_split:
            config["sim_config"]["geometry"]["domain_box"] = hom_domain_box
            DFMSim3D.homogenization(config, dfn, bulk_cond_values, bulk_cond_points)
        else:


            #exit()


            #steps = (int(fine_step), int(fine_step), int(fine_step))

            #n_steps_coef = 1.5
            #n_steps = (int(domain_size/int(fine_step)*n_steps_coef), int(domain_size/int(fine_step)*n_steps_coef), int(domain_size/int(fine_step)*n_steps_coef))

            #n_steps = (64, 64, 64)
            #n_steps = (25, 25, 25)
            #n_steps = (4, 4, 4)


            #grid_cond = DFMSim3D.homo_decovalex(fr_media, fem_grid.grid)
            #grid_cond = grid_cond.reshape(*n_steps, grid_cond.shape[-1])

            #print("grid_cond shape ", grid_cond.shape)
            #@TODO: rasterize input

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

            print("sample_dir ", sample_dir)


            # bc_pressure_gradient = [1, 0, 0]
            # flux_response_0 = DFMSim3D.get_flux_response(bc_pressure_gradient, fr_media, fem_grid, config, sample_dir, sim_config)
            #
            # exit()

            bc_pressure_gradient = [1, 0, 0]
            cond_file, fr_cond = DFMSim3D._run_sample(bc_pressure_gradient, fr_media, config, sample_dir, bulk_cond_values, bulk_cond_points, dimensions)
            flux_response_0 = DFMSim3D.get_flux_response()#bc_pressure_gradient, fr_media, fem_grid, config, sample_dir,
                                                         #sim_config)


            bc_pressure_gradient = [0, 1, 0]
            DFMSim3D._run_sample(bc_pressure_gradient, fr_media, config, sample_dir, bulk_cond_values, bulk_cond_points,
                                 dimensions, cond_file=cond_file)
            flux_response_1 = DFMSim3D.get_flux_response()#bc_pressure_gradient, fr_media, fem_grid, config, sample_dir,
                                                         #sim_config)

            bc_pressure_gradient = [0, 0, 1]
            DFMSim3D._run_sample(bc_pressure_gradient, fr_media, config, sample_dir, bulk_cond_values, bulk_cond_points,
                                 dimensions, cond_file=cond_file)
            flux_response_2 = DFMSim3D.get_flux_response()#bc_pressure_gradient, fr_media, fem_grid, config, sample_dir,
                                                         #sim_config)

            #print("flux response ", flux_response_0)



            bc_pressure_gradients = np.stack(([1, 0, 0], [0, 1, 0], [0, 0, 1]), axis=0)
            flux_responses = np.squeeze(np.stack((flux_response_0, flux_response_1, flux_response_2), axis=0))


            equivalent_cond_tn_voigt = equivalent_posdef_tensor(np.array(bc_pressure_gradients), flux_responses)

            #print("equivalent_cond_tn_voigt ", equivalent_cond_tn_voigt)

            equivalent_cond_tn = voigt_to_tn(np.array([equivalent_cond_tn_voigt])) #np.zeros((3, 3))

            # # Map the Voigt vector to the symmetric matrix
            # equivalent_cond_tn[0, 0] = equivalent_cond_tn_voigt[0]  # xx
            # equivalent_cond_tn[1, 1] = equivalent_cond_tn_voigt[1]  # yy
            # equivalent_cond_tn[2, 2] = equivalent_cond_tn_voigt[2]  # zz
            # equivalent_cond_tn[1, 2] = equivalent_cond_tn[2, 1] = equivalent_cond_tn_voigt[3]  # yz or zy
            # equivalent_cond_tn[0, 2] = equivalent_cond_tn[2, 0] = equivalent_cond_tn_voigt[4]  # xz or zx
            # equivalent_cond_tn[0, 1] = equivalent_cond_tn[1, 0] = equivalent_cond_tn_voigt[5]  # xy or yx
            print("equivalent cond tn ", equivalent_cond_tn)
            evals, evecs = np.linalg.eigh(equivalent_cond_tn)
            print("evals equivalent cond tn ", evals)
            assert np.all(evals) > 0


            fine_res = np.squeeze(equivalent_cond_tn_voigt)

            print("fine res shape ", fine_res.shape)

            DFMSim3D._remove_files()

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


            #if coarse_step > 0:



            #######################
            ## save to zarr file  #
            #######################
            # Shape of the data

            zarr_file_path = os.path.join(config["fine"]["common_files_dir"], DFMSim3D.ZARR_FILE)
            if os.path.exists(zarr_file_path):
                zarr_file = zarr.open(zarr_file_path, mode='r+')
                # Write data to the specified slice or index
                rasterized_input = DFMSim3D.rasterize(fem_grid_rast, dfn, bulk_cond=bulk_cond_values, fr_cond=fr_cond)

                rasterized_input_voigt = tn_to_voigt(rasterized_input)
                rasterized_input_voigt = rasterized_input_voigt.reshape(*n_steps, rasterized_input_voigt.shape[-1]).T

                zarr_file["inputs"][sample_idx, ...] = rasterized_input_voigt
                zarr_file["outputs"][sample_idx, :] = fine_res

        return fine_res, coarse_res


    @staticmethod
    def generate_grid_cond(fem_grid, config, seed):
        bulk_conductivity = config["sim_config"]['bulk_conductivity']
        if "marginal_distr" in bulk_conductivity and bulk_conductivity["marginal_distr"] is not False:
            means, cov = DFMSim3D.calculate_cov(bulk_conductivity["marginal_distr"])

            bulk_conductivity["mean_log_conductivity"] = means
            bulk_conductivity["cov_log_conductivity"] = cov
            del bulk_conductivity["marginal_distr"]

        # print("BULKFIelds bulk_conductivity ", bulk_conductivity)
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
        print("save tensors cond tensors ", cond_tensors)
        print("file to save ", file)
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
    def _run_sample(bc_pressure_gradient, fr_media, config, sample_dir, bulk_cond_values, bulk_cond_points, dimensions, cond_file=None, center=[0,0,0]):

        fr_cond = None
        if cond_file is None:
            cond_file, fr_cond = DFMSim3D.create_mesh_fields(fr_media, bulk_cond_values, bulk_cond_points, dimensions,
                                    mesh_step=config["fine"]["step"],
                                    sample_dir=sample_dir,
                                    work_dir=config["sim_config"]["work_dir"],
                                                             center=center,
                                                        )

        flow_cfg = dotdict(
            flow_executable=[
                "docker",
                "run",
                "-v",
                "{}:{}".format(os.getcwd(), os.getcwd()),
                #"flow123d/ci-gnu:3.9.0_49fc60d4d",
                "flow123d/ci-gnu:4.0.0a01_94c428",
                #"flow123d/ci-gnu:4.0.3dev_71cc9c",
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
            # pressure_grad_0=bc_gradients[0],
            # pressure_grad_1=bc_gradients[1],
            # pressure_grad_2=bc_gradients[2]
        )
        f_template = "flow_upscale_templ.yaml"
        shutil.copy(os.path.join(config["sim_config"]["work_dir"], f_template), sample_dir)
        with workdir_mng(sample_dir):
            flow_out = call_flow(flow_cfg, f_template, flow_cfg)

        # Project to target grid
        print(flow_out)

        return cond_file, fr_cond


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
        spec1 = QuantitySpec(name="cond_tn", unit="m", shape=(3, 1), times=[1], locations=['0'])

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
