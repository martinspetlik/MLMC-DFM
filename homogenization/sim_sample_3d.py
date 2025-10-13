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
#import pyvista as pv
from homogenization.gstools_bulk_3D import GSToolsBulk3D, GSToolsBulk3DEffective
from homogenization.srf_from_population import SRFFromTensorPopulation
from bgem import stochastic
from bgem import stochastic
from bgem.gmsh import gmsh, options
from mesh_class import Mesh
from bgem.core import call_flow, dotdict, workdir as workdir_mng
from bgem.upscale import fem_plot, fem, voigt_to_tn, tn_to_voigt, FracturedMedia, voxelize
from bgem.upscale.homogenization import equivalent_posdef_tensor
from typing import *
import zarr
from bgem.stochastic import FractureSet, EllipseShape, PolygonShape
from bgem.upscale import *
import scipy.interpolate as sc_interpolate
import cProfile
import pstats
from datetime import datetime

#os.environ["CUDA_VISIBLE_DEVICES"]=""


# print("torch.get_num_threads() ", torch.get_num_threads())
# torch.set_num_threads(torch.get_num_threads())  # Max out CPU cores
# torch.backends.mkldnn.enabled = True  # Use MKL-DNN for Conv3D
# torch.backends.openmp.enabled = True  # Enable OpenMP

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
    COND_TN_POP_COORDS_FILE = 'cond_tn_pop_coords.npy'
    PRED_COND_TN_POP_FILE = 'pred_cond_tn_pop.npy'
    COND_TN_FILE = "cond_tensors.yaml"
    COND_TN_VALUES_FILE = "cond_tensors_values.npz"
    COND_TN_COORDS_FILE = "cond_tensors_coords.npz"
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
        if "nn_path_levels" in self.config_dict and fine_step in self.config_dict["nn_path_levels"]:
            config["sim_config"]["nn_path"] = self.config_dict["nn_path_levels"][fine_step]
        config["fine"]["common_files_dir"] = common_files_dir

        if coarse_step != 0:
            config["coarse"]["sample_cond_tns"] = samples_cond_tns
        config["coarse"]["common_files_dir"] = coarse_sim_common_files_dir
        #config["fields_used_params"] = self._fields_used_params  # Params for Fields instance, which is created in PbsJob

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

        # print("zarr_file[inputs][index, :].shape ", zarr_file["inputs"][index, :].shape)

        hom_block_bulk = bulk_cond_fem_rast_voigt

        # print("np.mean(hom_block_bulk, axis=(1, 2, 3)) ", np.mean(hom_block_bulk, axis=(1, 2, 3)))

        zarr_file["bulk_avg"][index, :] = np.mean(hom_block_bulk, axis=(1, 2, 3))

        # index += 1

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
    def homogenization(config, dfn_to_homogenize, dfn_to_homogenize_list, bulk_cond_values, bulk_cond_points, seed=None,
                       hom_dir_name="homogenization"):
        sample_dir = os.getcwd()

        print("homogenization method")
        print("os.getcwd() ", os.getcwd())
        # print("config[scratch_dir] ", config["sim_config"]["scratch_dir"])
        print("os environ get SCRATCHDIR ", os.environ.get("SCRATCHDIR"))

        # hom_dir_abs_path = os.path.join(sample_dir, hom_dir_name)
        # hom_dir_name = "homogenization"
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

        # fr_media = FracturedMedia.fracture_cond_params(dfn_to_homogenize, 1e-4, 0.00001)

        # for fr in dfn:
        #     print("hom fr.r ", fr.r)

        # hom_dir_name = "homogenization"
        # if "scratch_dir" in config["sim_config"] and config["sim_config"]["scratch_dir"] is not None:
        #     #hom_dir_abs_path = os.path.join(config["sim_config"]["scratch_dir"], hom_dir_name)
        #     hom_dir_abs_path = os.path.join(os.environ.get("SCRATCHDIR"), hom_dir_name)
        # else:
        # hom_dir_abs_path = os.path.join(sample_dir, hom_dir_name)

        # print("hom_dir_abs_path ", hom_dir_abs_path)

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
        # lx, ly = domain_box

        print("n_subdomains_per_axes ", n_subdomains_per_axes)
        # exit()
        # n_subdomains = int(np.floor(np.sqrt(n_subdomains)))

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
        # create_hom_samples_start_time = time.time()
        for i in range(n_subdomains_per_axes):
            center_x = subdomain_box[0] / 2 + (lx - subdomain_box[0]) / (n_subdomains_per_axes - 1) * i - lx / 2

            for j in range(n_subdomains_per_axes):
                # start_time = time.time()

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

                    # outer_cube = DFMSim3D.get_outer_cube(center_x, center_y, center_z, box_size_x, box_size_y, box_size_z)
                    # sim_config["geometry"]["outer_cube"] = outer_cube
                    # print("work_dir ", work_dir)

                    # print("outer polygon ", outer_polygon)
                    # print("center x:{} y:{}".format(center_x, center_y))

                    # sim_config["work_dir"] = work_dir
                    # config["homogenization"] = True

                    subdomain_box_run_samples = copy.deepcopy(subdomain_box)

                    # print("subdomain box run samples ", subdomain_box_run_samples)
                    # print("np.array(subdomain_box_run_samples)*1.1 ", np.array(subdomain_box_run_samples)*1.1)

                    cond_field_step = np.abs(bulk_cond_points[0][-1]) - np.abs(bulk_cond_points[1][-1])
                    subdomain_to_extract = np.array(subdomain_box_run_samples) * 1.1 + (2 * cond_field_step)
                    subdomain_bulk_cond_values, subdomain_bulk_cond_points = DFMSim3D.extract_subdomain(
                        bulk_cond_values, bulk_cond_points, (center_x, center_y, center_z), subdomain_to_extract)

                    try:
                        subdomain_dfn_to_homogenize = DFMSim3D.extract_dfn(dfn_to_homogenize, dfn_to_homogenize_list,
                                                                           (center_x, center_y, center_z),
                                                                           np.array(subdomain_box_run_samples) * 1.5)
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
                            # print("subdomain_box_run_samples ", subdomain_box_run_samples)

                            bc_pressure_gradient = [1, 0, 0]
                            cond_file, fr_cond, fr_region_map = DFMSim3D._run_sample(bc_pressure_gradient,
                                                                                     subdomain_fr_media, config,
                                                                                     hom_sample_dir,
                                                                                     subdomain_bulk_cond_values,
                                                                                     subdomain_bulk_cond_points,
                                                                                     subdomain_box_run_samples,
                                                                                     mesh_step=config["fine"]["step"],
                                                                                     center=[center_x, center_y,
                                                                                             center_z], regular_grid_interp=False)
                            flux_response_0 = DFMSim3D.get_flux_response()  # bc_pressure_gradient, fr_media, fem_grid, config, sample_dir,
                            # sim_config)

                            bc_pressure_gradient = [0, 1, 0]
                            DFMSim3D._run_sample(bc_pressure_gradient, subdomain_fr_media, config, hom_sample_dir,
                                                 subdomain_bulk_cond_values,
                                                 subdomain_bulk_cond_points,
                                                 subdomain_box_run_samples, mesh_step=config["fine"]["step"],
                                                 cond_file=cond_file, center=[center_x, center_y, center_z],
                                                 regular_grid_interp=False)
                            flux_response_1 = DFMSim3D.get_flux_response()  # bc_pressure_gradient, fr_media, fem_grid, config, sample_dir,
                            # sim_config)

                            bc_pressure_gradient = [0, 0, 1]
                            DFMSim3D._run_sample(bc_pressure_gradient, subdomain_fr_media, config, hom_sample_dir,
                                                 subdomain_bulk_cond_values,
                                                 subdomain_bulk_cond_points,
                                                 subdomain_box_run_samples, mesh_step=config["fine"]["step"],
                                                 cond_file=cond_file, center=[center_x, center_y, center_z],
                                                 regular_grid_interp=False)

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
                    # print("equivalent cond tn ", equivalent_cond_tn)
                    # evals, evecs = np.linalg.eigh(equivalent_cond_tn)
                    # print("evals equivalent cond tn ", evals)
                    # assert np.all(evals) > 0

                    cond_tensors[(new_center_x, new_center_y, new_center_z)] = equivalent_cond_tn

                    if "nn_path_for_block_hom" in config["sim_config"]:
                        # try:
                        fem_grid_rast = fem.fem_grid(subdomain_box[0], config["sim_config"]["geometry"]["n_voxels"],
                                                     fem.Fe.Q(dim=3),
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
                        # except:
                        #    DFMSim3D._remove_files()
                        #    os.chdir(h_dir)
                        #    continue

                    # print("equivalent_cond_tn ", tn_to_voigt(equivalent_cond_tn))
                    # print("equivalent_cond_tn_predictions ", equivalent_cond_tn_predictions)
                    # exit()

                    # print("fem_grid_rast.grid ", fem_grid_rast.grid)
                    # print("fem_grid_rast.grid center ", fem_grid_rast.grid.grid_center())

                    # bulk_cond_fem_rast = DFMSim3D._bulk_cond_to_rast_grid(bulk_cond_values, bulk_cond_points,
                    #                                                       fem_grid_rast.grid)

                    # rasterized_input = DFMSim3D.rasterize(fem_grid_rast, dfn, bulk_cond=bulk_cond_fem_rast,
                    #                                      fr_cond=fr_cond)

                    # rasterized_input_voigt = tn_to_voigt(rasterized_input)
                    # rasterized_input_voigt = rasterized_input_voigt.reshape(*config["sim_config"]["geometry"]["n_voxels"],
                    #                                                        rasterized_input_voigt.shape[-1]).T

                    # fine_res = np.squeeze(equivalent_cond_tn_voigt)
                    #
                    # DFMSim3D.rasterize_save_to_zarr(zarr_file_path, config, k, fine_res, bulk_cond_values,
                    #                                 bulk_cond_points, dfn, fr_cond,
                    #                                 fem_grid_rast, n_steps=config["sim_config"]["geometry"]["n_voxels"])

                    # np.save("equivalent_cond_tn", equivalent_cond_tn)
                    # np.savez_compressed("rasterized_input_voigt ", rasterized_input_voigt)

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
        """
        Generates a random set of fractures using the stochastic fracture population model.
        :param seed: Integer random seed for reproducibility of fracture generation
        :param size_range: Tuple (rmin, rmax) specifying the minimum and maximum fracture radii
        :param work_dir: Path to working directory containing the fracture configuration file
        :param max_frac: Maximum allowed number of fractures in the generated set (default: large number)
        :return: List of generated fracture objects
        """
        rmin, rmax = size_range
        box_dimensions = (rmax, rmax, rmax)
        fr_cfg_path = os.path.join(work_dir, "fractures_conf.yaml")

        # Load fracture population from configuration file
        fr_pop = stochastic.Population.from_cfg(fr_cfg_path, box_dimensions, shape=stochastic.EllipseShape())

        # Limit the sample size if it exceeds max_frac
        if fr_pop.mean_size() > max_frac:
            common_range = fr_pop.common_range_for_sample_size(sample_size=max_frac)
            fr_pop = fr_pop.set_sample_range(common_range)

        print(f"fr set range: {[rmin, rmax]}, fr_lim: {max_frac}, mean population size: {fr_pop.mean_size()}")

        # Generate fracture positions and sample fractures
        pos_gen = stochastic.UniformBoxPosition(fr_pop.domain)
        np.random.seed(seed)
        fractures = fr_pop.sample(pos_distr=pos_gen, keep_nonempty=True)

        return fractures

    @staticmethod
    def create_fractures_rectangles(gmsh_geom, fractures: FrozenSet, base_shape: gmsh.ObjectSet,
                                    shift=np.array([0, 0, 0])):
        """
        Creates rectangular fracture geometries from the given fracture definitions and inserts them into the GMSH geometry.

        :param gmsh_geom: GMSH geometry object used for construction and fragmentation
        :param fractures: Set of fracture definitions (typically containing orientation, size, and center info)
        :param base_shape: A base rectangle shape that is transformed for each fracture
        :param shift: Optional shift applied to the fracture center (default: [0, 0, 0])
        :return: Tuple (fracture_fragments, region_map), where fracture_fragments is a list of gmsh objects
                 representing the fragments, and region_map maps region names to fracture indices
        """

        # From given fracture data list 'fractures'.
        # transform the base_shape to fracture objects
        # fragment fractures by their intersections
        # return dict: fracture.region -> GMSHobject with corresponding fracture fragments
        if len(fractures) == 0:
            return [], []

        shapes = []
        region_map = {}
        for i, fr in enumerate(fractures):
            shape = base_shape.copy()
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

        #fr_dict.mesh_step(fr_step)

        #[print(k, v) for k, v in fr_dict.items()]
        #[print(k, v) for k, v in fr_bnd_dict.items()]

        geometry_set = list(fr_dict.values())

        for geometry_obj in geometry_set:
            geometry_obj.mesh_step(bulk_step)

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
        """
        Generate a 3D mesh with embedded fractures using Gmsh.
        :param work_dir: Directory where the resulting mesh file will be saved
        :param domain_dimensions: Tuple or list defining the dimensions (Lx, Ly, Lz) of the simulation domain
        :param fractures: List of fracture definitions to be inserted into the domain
        :param fr_step: Mesh size to be used on fracture surfaces
        :param bulk_step: Mesh size to be used in the bulk domain
        :param center: Center point (x, y, z) of the simulation domain
        :return: Tuple (mesh_file, fr_region_map) with the path to the .msh2 mesh file and region map of fractures
        """
        factory = gmsh.GeometryOCC("homo_cube", verbose=False)
        gopt = options.Geometry()
        gopt.Tolerance = 0.0001
        gopt.ToleranceBoolean = 0.001

        box = factory.box(domain_dimensions, center)

        fractures, fr_region_map = DFMSim3D.create_fractures_rectangles(factory, fractures, factory.rectangle())

        fractures_group = factory.group(*fractures).intersect(box)
        box_fr, fractures_fr = factory.fragment(box, fractures_group)

        fractures_fr.mesh_step(fr_step)  # .set_region("fractures")
        objects = [box_fr, fractures_fr]
        factory.write_brep(str(factory.model_name))

        #factory.mesh_options.CharacteristicLengthMin = bulk_step
        factory.mesh_options.CharacteristicLengthMax = bulk_step
        factory.mesh_options.Algorithm = options.Algorithm3d.Delaunay

        factory.make_mesh(objects, dim=3)

        f_name = factory.model_name + ".msh2"
        mesh_file = work_dir / (factory.model_name + ".msh2")
        factory.write_mesh(str(mesh_file), format=gmsh.MeshFormat.msh2)

        return mesh_file, fr_region_map

    @staticmethod
    def fr_cross_section(fractures, cross_to_r):
        return [cross_to_r * fr.r for fr in fractures]

    @staticmethod
    def fr_field(mesh, dfn, reg_id_to_fr, fr_values, bulk_value, rnd_cond=False, field_dim=3):
        """
        Provide fractures cond values and cross-section.
        :param mesh: The mesh object containing the domain and fractures
        :param dfn: Discrete Fracture Network structure or data used to identify fracture regions
        :param reg_id_to_fr: Dictionary mapping mesh region IDs to fracture indices
        :param fr_values: Array of tensor or scalar values assigned to fractures
        :param bulk_value: Scalar or tensor value used for the bulk (non-fracture) domain
        :param rnd_cond: Boolean flag to generate random conductivity values (currently not implemented)
        :param field_dim: Dimension of the field (1 for scalar, 3 for 3x3 tensor per element)
        :return: Tuple (field_vals, fr_map) where `field_vals` is an array of conductivity/tensor values per element and `fr_map` is the fracture index map for each element
        """
        fr_map = mesh.fr_map(dfn,
                             reg_id_to_fr)  # np.array of fracture indices of elements, n_frac for nonfracture elements

        if field_dim == 3:
            bulk_cond_tn = np.eye(3) * bulk_value
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

    # @staticmethod
    # def project_adaptive_source_quad(flow_out, grid: fem.Grid):
    #     grid_cell_volume = np.prod(grid.step) / 27
    #
    #     ref_el_2d = np.array([(0, 0), (1, 0), (0, 1)])
    #     ref_el_3d = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
    #
    #     pvd_content = pv.get_reader(flow_out.hydro.spatial_file.path)
    #     pvd_content.set_active_time_point(0)
    #     dataset = pvd_content.read()[0]  # Take first block of the Multiblock dataset
    #
    #     velocities = dataset.cell_data['velocity_p0']
    #     cross_section = dataset.cell_data['cross_section']
    #
    #     p_dataset = dataset.cell_data_to_point_data()
    #     p_dataset.point_data['velocity_magnitude'] = np.linalg.norm(p_dataset.point_data['velocity_p0'], axis=1)
    #     plane = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1))
    #     cut_dataset = p_dataset.clip_surface(plane)
    #
    #     plotter = pv.Plotter()
    #     plotter.add_mesh(p_dataset, color='white', opacity=0.3, label='Original Dataset')
    #     plotter.add_mesh(cut_dataset, scalars='velocity_magnitude', cmap='viridis', label='Velocity Magnitude')
    #
    #     # Add legend and show the plot
    #     plotter.add_scalar_bar(title='Velocity Magnitude')
    #     plotter.add_legend()
    #     plotter.show()
    #
    #     # num_cells = dataset.n_cells
    #     # shifts = np.zeros((num_cells, 3))
    #     # transform_matrices = np.zeros((num_cells, 3, 3))
    #     # volumes = np.zeros(num_cells)
    #
    #     weights_sum = np.zeros((grid.n_elements,))
    #     grid_velocities = np.zeros((grid.n_elements, 3))
    #     levels = np.zeros(dataset.n_cells, dtype=np.int32)
    #     # Loop through each cell
    #     for i in range(dataset.n_cells):
    #         cell = dataset.extract_cells(i)
    #         points = cell.points
    #
    #         if len(points) < 3:
    #             continue  # Skip cells with less than 3 vertices
    #
    #         # Shift: the first vertex of the cell
    #         shift = points[0]
    #         # shifts[i] = shift
    #
    #         transform_matrix = points[1:] - shift
    #         if len(points) == 4:  # Tetrahedron
    #             # For a tetrahedron, we use all three vectors formed from the first vertex
    #             # transform_matrices[i] = transform_matrix[:3].T
    #             # Volume calculation for a tetrahedron:
    #             volume = np.abs(np.linalg.det(transform_matrix[:3])) / 6
    #             ref_el = ref_el_3d
    #         elif len(points) == 3:  # Triangle
    #             # For a triangle, we use only two vectors
    #             # transform_matrices[i, :2] = transform_matrix.T
    #             # Area calculation for a triangle:
    #             volume = 0.5 * np.linalg.norm(np.cross(transform_matrix[0], transform_matrix[1])) * cross_section[i]
    #             ref_el = ref_el_2d
    #         level = max(int(np.log2(volume / grid_cell_volume) / 3.0), 0)
    #         levels[i] = level
    #         ref_barycenters = DFMSim3D.refine_barycenters(ref_el[None, :, :], level)
    #         barycenters = shift[None, :] + ref_barycenters @ transform_matrix
    #         grid_indices = grid.project_points(barycenters)
    #         weights_sum[grid_indices] += volume
    #         grid_velocities[grid_indices] += volume * velocities[i]
    #     print(np.bincount(levels))
    #     grid_velocities = grid_velocities / weights_sum[:, None]
    #     return grid_velocities

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
    def get_flux_response():  # bc_pressure_gradients, fr_media, fem_grid, config, sample_dir, sim_config):
        # Load the simulation output mesh and compute volume-averaged velocity response (flux)

        # Output GMSH mesh handler
        out_mesh = gmsh_io.GmshIO()

        # Read the output mesh file containing flow fields
        with open("flow_fields.msh", "r") as f:
            out_mesh.read(f)

        # Select the first time step
        time_idx = 0
        time, field_cs = out_mesh.element_data['cross_section'][time_idx]

        # Compute element volumes and associated region IDs
        ele_reg_vol = {
            eid: (tags[0] - 10000, DFMSim3D.element_volume(out_mesh, nodes))
            for eid, (tele, tags, nodes) in out_mesh.elements.items()
        }

        # Extract elementwise velocity field
        velocity_field = out_mesh.element_data['velocity_p0']

        # Only one group assumed; adjust here if multiple material groups are needed
        n_groups = 1

        # Number of pressure gradient directions (1 if not running multiple load cases)
        n_directions = 1  # len(bc_pressure_gradients)

        # Initialize arrays to hold the integrated flux and total volume
        flux_response = np.zeros((n_groups, n_directions, 3))
        total_volume = np.zeros((n_groups, n_directions))

        print("Averaging velocities ...")

        # Iterate over all timesteps in the velocity field
        for i_time, (time, velocity) in velocity_field.items():
            i_time = int(i_time)

            for eid, ele_vel in velocity.items():
                # Get region ID and volume for the current element
                reg_id, vol = ele_reg_vol[eid]

                # Get cross-sectional scaling factor
                cs = field_cs[eid][0]

                # Compute the effective volume contribution
                volume = cs * vol

                # Group index is hardcoded to 0 (single group case)
                i_group = 0

                # Accumulate negative volumetric flux vector
                flux_response[i_group, i_time, :] += -(volume * np.array(ele_vel[0:3]))

                # Accumulate total volume for normalization
                total_volume[i_group, i_time] += volume

        # Normalize flux by total volume to get average
        flux_response /= total_volume

        # Remove singleton direction dimension and transpose for output
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
        """
        Rasterize fracture and bulk conductivity tensors onto a regular grid.
        :param fem_grid: Finite element mesh grid object
        :param dfn: Discrete fracture network object
        :param bulk_cond: Bulk conductivity tensor values on the grid
        :param fr_cond: Fracture conductivity tensor values
        :return: Rasterized conductivity tensor values on the target grid
        """
        target_grid = fem_grid.grid

        isec_corners = DFMSim3D.intersection_cell_corners_vec(dfn, target_grid)

        rasterized = isec_corners.interpolate(bulk_cond, fr_cond, source_grid=fem_grid.grid)

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
    def create_mesh_fields(fr_media, bulk_cond_values, bulk_cond_points, dimensions, mesh_step, sample_dir, work_dir,
                           center=[0, 0, 0], outflow_problem=False, file_prefix="", fr_region_map=None,
                           regular_grid_interp=True, config={}, sample_seed=None):
        """
        Create conductivity fields on a mesh from fracture and bulk media information.

        :param fr_media: Fracture media object containing DFN and bulk conductivity info
        :param bulk_cond_values: Values of bulk conductivity tensor (Nx9 or Nx3x3 array)
        :param bulk_cond_points: Coordinates corresponding to bulk conductivity values (Nx3 array)
        :param dimensions: Domain dimensions (x, y, z) for the simulation cube
        :param mesh_step: Mesh resolution for both fracture and bulk parts
        :param sample_dir: Path where mesh and field files will be written
        :param work_dir: Working directory for temporary files (unused here)
        :param center: Center of the simulation domain [x, y, z]
        :param outflow_problem: If True, apply mesh generation tailored for outflow boundary condition
        :param file_prefix: Prefix for output .npy field data files
        :param fr_region_map: Optional region mapping from mesh generator; regenerated if None
        :param regular_grid_interp: Whether to use RegularGridInterpolator (True) or LinearNDInterpolator (False)
        :param config: Configuration dictionary for fallback bulk conductivity generation
        :param sample_seed: Seed for reproducible bulk field generation (if config is used)
        :return: Tuple of (field_mesh_file, fracture_conductivity_tensors, fracture_region_map, mesh_region_ids)
        """
        # Extract DFN and bulk conductivity from fracture media object
        dfn = fr_media.dfn
        bulk_conductivity = fr_media.conductivity

        print("bulk cond points ", bulk_cond_points)

        # Prepare bulk conductivity interpolation if bulk data is provided
        if len(bulk_cond_values) > 0 and len(bulk_cond_points) > 0:
            cond_field_step = np.abs(bulk_cond_points[0][-1]) - np.abs(bulk_cond_points[1][-1])
            subdomain_to_extract = np.array(dimensions) * 1.1 + cond_field_step

            print("subdomain to extract ", subdomain_to_extract)

            # # Extract subdomain of bulk conductivity field
            # subdomain_bulk_cond_values, subdomain_bulk_cond_points = DFMSim3D.extract_subdomain(
            #     bulk_cond_values, bulk_cond_points, (center[0], center[1], center[2]), subdomain_to_extract
            # )
            subdomain_bulk_cond_values = bulk_cond_values
            subdomain_bulk_cond_points = bulk_cond_points

            # Choose interpolation method for bulk field
            if regular_grid_interp:
                # Regular grid interpolation
                x_unique = np.unique(subdomain_bulk_cond_points[:, 0])
                y_unique = np.unique(subdomain_bulk_cond_points[:, 1])
                z_unique = np.unique(subdomain_bulk_cond_points[:, 2])
                nx, ny, nz = len(x_unique), len(y_unique), len(z_unique)

                sort_idx = np.lexsort((subdomain_bulk_cond_points[:, 2], subdomain_bulk_cond_points[:, 1],
                                       subdomain_bulk_cond_points[:, 0]))
                values_sorted = subdomain_bulk_cond_values[sort_idx]
                values_grid = values_sorted.reshape((nx, ny, nz, 3, 3))

                print("values_grid.shape ", values_grid.shape)
                print("x_unique ", x_unique)

                interp = sc_interpolate.RegularGridInterpolator((x_unique, y_unique, z_unique), values_grid)
            else:
                # Scattered interpolation
                interp = sc_interpolate.LinearNDInterpolator(subdomain_bulk_cond_points, subdomain_bulk_cond_values,
                                                             fill_value=0)

        ###########################
        ## Fracture conductivity ##
        ###########################
        fr_cond, fr_cross_section = [], []
        if len(dfn) > 0:
            fr_cross_section, fr_cond = DFMSim3D.fr_conductivity(dfn, cross_section_factor=1e-4)

        # Convert to pandas DataFrame format (or compatible structure)
        fr_cond = DFMSim3D.make_fr_cond_pd(fr_cond)

        # Sanity check for invalid conductivity tensors (eigenvalues ≤ 0)
        n = 0
        for i in range(fr_cond.shape[0]):
            evals, evecs = np.linalg.eigh(fr_cond[i])
            if np.any(evals <= 0):
                print("evals equivalent cond tn ", evals)
                print("cond_tensors[i].shape", fr_cond[i])
                n += 1

        # Generate mesh if it doesn't already exist or if region map is missing
        if not os.path.exists(os.path.join(sample_dir, "homo_cube_healed.msh2")) or fr_region_map is None:
            if outflow_problem:
                mesh_file, fr_region_map = DFMSim3D.ref_solution_mesh_outflow_problem(
                    sample_dir, dimensions, dfn, fr_step=mesh_step, bulk_step=mesh_step, center=center
                )
            else:
                mesh_file, fr_region_map = DFMSim3D.ref_solution_mesh(
                    sample_dir, dimensions, dfn, fr_step=mesh_step, bulk_step=mesh_step, center=center
                )
        else:
            mesh_file = sample_dir / ("homo_cube_healed.msh2")

        # Load and heal the mesh
        full_mesh = Mesh.load_mesh(mesh_file, heal_tol=0.001)

        # Generate fracture conductivity field on the mesh
        fr_cond_tn, fr_map = DFMSim3D.fr_field(full_mesh, dfn, fr_region_map, fr_cond, bulk_conductivity,
                                               rnd_cond=False, field_dim=3)

        # Generate fracture cross-section field (1D)
        cross_sections, fr_map_cs = DFMSim3D.fr_field(full_mesh, dfn, fr_region_map, fr_cross_section, 1.0, field_dim=1)
        cross_sections = np.array(cross_sections)

        #######################################
        ## Interpolate SRF to mesh elements  ##
        #######################################
        bulk_elements_barycenters = full_mesh.el_barycenters(elements=full_mesh._bulk_elements)

        # print("bulk_elements_barycenters.shape ", bulk_elements_barycenters.shape)
        # print("bulk_elements_barycenters[0] min: {}, max: {}".format(np.min(bulk_elements_barycenters[:, 0]),
        #                                                              np.max(bulk_elements_barycenters[:, 0])))
        # from scipy.spatial import Delaunay
        # # Create Delaunay triangulation of known points
        # tri = Delaunay(bulk_cond_points)
        # # Check if barycenters are inside the convex hull
        # inside_mask = tri.find_simplex(bulk_elements_barycenters) >= 0
        # # inside_mask is a boolean array, True if inside hull, False if outside
        # outside_mask = ~inside_mask
        #
        # print(f"Points inside convex hull: {np.sum(inside_mask)}")
        # print(f"Points outside convex hull: {np.sum(outside_mask)}")
        #
        # # Given unique grid axes (1D arrays)
        # x_unique = np.unique(subdomain_bulk_cond_points[:, 0])
        # y_unique = np.unique(subdomain_bulk_cond_points[:, 1])
        # z_unique = np.unique(subdomain_bulk_cond_points[:, 2])
        #
        # print("x_unique ", x_unique)
        # print("y_unique ", y_unique)
        # print("z_unique ", z_unique)
        #
        # # Compute bounding box corners
        # bbox_min = np.array([x_unique.min(), y_unique.min(), z_unique.min()])
        # bbox_max = np.array([x_unique.max(), y_unique.max(), z_unique.max()])
        #
        # # Your points to check (M x 3)
        # points = bulk_elements_barycenters
        #
        # # Test if points lie inside bounding box (inclusive)
        # inside_bbox_mask = np.all((points >= bbox_min) & (points <= bbox_max), axis=1)
        #
        # # Summary
        # print(f"Points inside bounding box: {np.sum(inside_bbox_mask)} / {len(points)}")
        # print(f"Points outside bounding box: {len(points) - np.sum(inside_bbox_mask)}")
        #
        # outside_points = points[~inside_bbox_mask]
        # print("outside points ", outside_points)

        # Interpolate bulk conductivity to barycenters
        if len(bulk_cond_values) > 0 and len(bulk_cond_points) > 0:
            full_mesh_bulk_cond_values = interp(bulk_elements_barycenters)
            zero_rows = np.where(np.all(full_mesh_bulk_cond_values == 0, axis=1))[0]

            # Fill missing (zero) values with nearest neighbor interpolation
            if len(zero_rows) > 0:
                print("ZERO ROWS")
                from scipy.interpolate import NearestNDInterpolator
                nn_interp = NearestNDInterpolator(bulk_cond_points, bulk_cond_values)
                full_mesh_bulk_cond_values[zero_rows] = nn_interp(bulk_elements_barycenters[zero_rows])
        else:
            # Generate random or predefined bulk conductivity
            full_mesh_bulk_cond_values, bulk_cond_points = DFMSim3D.generate_grid_cond(bulk_elements_barycenters,
                                                                                       config, seed=sample_seed)

        ##################
        ## Write fields ##
        ##################
        # Combine fracture and bulk conductivity into one field
        fr_cond_tn[-full_mesh_bulk_cond_values.shape[0]:, ...] = full_mesh_bulk_cond_values
        conductivity = fr_cond_tn.reshape(fr_cond_tn.shape[0], 9)

        #######################
        ## Map fracture sizes #
        #######################

        # Get unique conductivity tensors and associated fracture sizes
        fr_cond_unique = fr_cond
        fracture_size_unique = dfn.radius[:, 0]  # shape (N_unique,)

        # Extract fracture-only portion of the full tensor field
        voxel_fr_cond = fr_cond_tn[:-full_mesh_bulk_cond_values.shape[0], ...]

        # Flatten tensors for comparison
        unique_flat = fr_cond_unique.reshape(len(fr_cond_unique), -1)
        voxel_flat = voxel_fr_cond.reshape(len(voxel_fr_cond), -1)

        # Compare each voxel tensor to all unique tensors using broadcasting
        matches = np.all(np.isclose(voxel_flat[:, None, :], unique_flat[None, :, :], rtol=1e-5), axis=2)

        # Get matching index for each voxel
        voxel_to_unique_idx = np.argmax(matches, axis=1)

        # Assign fracture size to each voxel based on match
        voxel_fracture_size = fracture_size_unique[voxel_to_unique_idx]

        # Save voxel-level fields for further analysis
        np.save("{}voxel_fracture_sizes".format(file_prefix), voxel_fracture_size)
        np.save("{}fr_cond_data".format(file_prefix), fr_cond_tn[:-full_mesh_bulk_cond_values.shape[0], ...])
        np.save("{}bulk_cond_data".format(file_prefix), fr_cond_tn[-full_mesh_bulk_cond_values.shape[0]:, ...])

        # Create dictionary of fields to export
        fields = dict(conductivity=conductivity, cross_section=cross_sections.reshape(-1, 1))

        # Write fields into mesh file
        cond_file = full_mesh.write_fields(str(sample_dir / "input_fields.msh2"), fields)
        cond_file = Path(cond_file)
        cond_file = cond_file.rename(cond_file.with_suffix(".msh"))

        return cond_file, fr_cond, fr_region_map, full_mesh.regions

    @staticmethod
    def calculate_hom_sample(config, sample_dir, sample_idx, sample_seed):
        """
        Calculates a homogenized conductivity tensor sample from DFM models
        :param config: Dictionary with simulation configuration, including geometry, mesh steps, and paths
        :param sample_dir: Path to the directory where sample-specific data will be stored
        :param sample_idx: Index of the current sample in the dataset
        :param sample_seed: Random seed used for reproducibility of the fracture network
        :return: Tuple containing (fine_res, coarse_res), the fine-scale and coarse-scale conductivity tensors
        """
        domain_size = 15  # Size of the rasterization domain
        fem_grid_cond_domain_size = 16  # Domain size for the conductivity FEM grid
        fr_domain_size = 100  # Domain size for generating fractures
        fr_range = (config["fine"]["step"], fr_domain_size)  # Fracture radius range
        coarse_step = 10  # Coarse discretization step
        fine_step = config["fine"]["step"]  # Fine discretization step

        sim_config = config["sim_config"]
        geom = sim_config["geometry"]

        #########
        # DFN
        ########
        dfn = []
        if geom["n_frac_limit"] > 0:
            # Generate random fractures using DFMSim3D with a limit on number
            dfn = DFMSim3D.fracture_random_set(sample_seed, fr_range, sim_config["work_dir"],
                                               max_frac=geom["n_frac_limit"])
            dfn_to_homogenization = []

            # Filter fractures that fall within the size range relevant for homogenization
            for fr in dfn:
                if fine_step <= fr.r <= coarse_step:
                    dfn_to_homogenization.append(fr)

            dfn = stochastic.FractureSet.from_list(dfn_to_homogenization)

        # Number of grid steps for rasterization
        n_steps = config["sim_config"]["geometry"]["n_voxels"]

        # Compute fracture media conductivity using the cubic law
        fr_media = FracturedMedia.fracture_cond_params(dfn, 1e-4, 0.00001)

        n_steps_cond_grid = (fem_grid_cond_domain_size, fem_grid_cond_domain_size, fem_grid_cond_domain_size)

        # Create FEM grids for the conductivity and raster domains
        fem_grid_cond = fem.fem_grid(fem_grid_cond_domain_size, n_steps_cond_grid, fem.Fe.Q(dim=3),
                                     origin=-fem_grid_cond_domain_size / 2)

        fem_grid_rast = fem.fem_grid(domain_size, n_steps, fem.Fe.Q(dim=3),
                                     origin=-domain_size / 2)

        #######################
        ## Bulk conductivity ##
        #######################
        # Generate bulk conductivity field on the FEM grid
        bulk_cond_values, bulk_cond_points = DFMSim3D.generate_grid_cond(fem_grid_cond.grid.barycenters(), config,
                                                                         seed=sample_seed)

        ##################################
        ## Create mesh and input fields ##
        ##################################
        dimensions = (domain_size, domain_size, domain_size)

        # Run 3 simulations with different pressure gradient boundary conditions
        bc_pressure_gradient = [1, 0, 0]
        cond_file, fr_cond, fr_region_map = DFMSim3D._run_sample(bc_pressure_gradient, fr_media, config, sample_dir,
                                                                 bulk_cond_values, bulk_cond_points, dimensions,
                                                                 mesh_step=config["fine"]["step"])
        flux_response_0 = DFMSim3D.get_flux_response()

        bc_pressure_gradient = [0, 1, 0]
        DFMSim3D._run_sample(bc_pressure_gradient, fr_media, config, sample_dir, bulk_cond_values, bulk_cond_points,
                             dimensions, cond_file=cond_file, mesh_step=config["fine"]["step"])
        flux_response_1 = DFMSim3D.get_flux_response()

        bc_pressure_gradient = [0, 0, 1]
        DFMSim3D._run_sample(bc_pressure_gradient, fr_media, config, sample_dir, bulk_cond_values, bulk_cond_points,
                             dimensions, cond_file=cond_file, mesh_step=config["fine"]["step"])
        flux_response_2 = DFMSim3D.get_flux_response()

        # Stack pressure gradients and flux responses to compute the equivalent tensor
        bc_pressure_gradients = np.stack(([1, 0, 0], [0, 1, 0], [0, 0, 1]), axis=0)
        flux_responses = np.squeeze(np.stack((flux_response_0, flux_response_1, flux_response_2), axis=0))

        # Calculate equivalent symmetric positive-definite tensor in Voigt notation
        equivalent_cond_tn_voigt = equivalent_posdef_tensor(np.array(bc_pressure_gradients), flux_responses)

        # Convert Voigt-form tensor to full 3x3 tensor
        equivalent_cond_tn = voigt_to_tn(np.array([equivalent_cond_tn_voigt]))

        # Sanity check: ensure positive definiteness
        evals, evecs = np.linalg.eigh(equivalent_cond_tn)
        assert np.all(evals) > 0

        fine_res = np.squeeze(equivalent_cond_tn_voigt)

        # # Update config with tensor population file paths if they exist
        # cond_tn_pop = os.path.join(config["fine"]["common_files_dir"], DFMSim3D.COND_TN_POP_FILE)
        # if os.path.exists(cond_tn_pop):
        #     config["fine"]["cond_tn_pop_file"] = cond_tn_pop
        #
        # if "nn_path" in config["sim_config"]:
        #     pred_cond_tn_pop = os.path.join(config["fine"]["common_files_dir"], DFMSim3D.PRED_COND_TN_POP_FILE)
        #     if os.path.exists(pred_cond_tn_pop):
        #         config["fine"]["pred_cond_tn_pop_file"] = pred_cond_tn_pop
        #
        # sample_cond_tns = os.path.join(config["fine"]["common_files_dir"], DFMSim3D.SAMPLES_COND_TNS_DIR)
        # if os.path.exists(sample_cond_tns):
        #     config["fine"]["sample_cond_tns"] = sample_cond_tns

        coarse_res = [0, 0, 0, 0, 0, 0]  # Placeholder for coarse resolution result

        #######################
        ## save to zarr file  #
        #######################
        # Save voxelized inputs and output tensor to Zarr format
        zarr_file_path = os.path.join(config["fine"]["common_files_dir"], DFMSim3D.ZARR_FILE)
        DFMSim3D.rasterize_save_to_zarr(zarr_file_path, config, sample_idx, bulk_cond_values, bulk_cond_points, dfn,
                                        fr_cond,
                                        fem_grid_rast, n_steps, fine_res)

        zarr_file_path = os.path.join(config["fine"]["common_files_dir"], DFMSim3D.ZARR_FILE)
        print("zarr file path ", zarr_file_path)
        if os.path.exists(zarr_file_path):
            zarr_file = zarr.open(zarr_file_path, mode='r+')

            #######
            ## bulk cond values to fem grid rast
            #######
            # Interpolate bulk conductivity values to the raster grid
            bulk_cond_fem_rast = DFMSim3D._bulk_cond_to_rast_grid(bulk_cond_values, bulk_cond_points,
                                                                  fem_grid_rast.grid)

            # Rasterize combined input: fractures + interpolated bulk conductivity
            rasterized_input = DFMSim3D.rasterize(fem_grid_rast, dfn, bulk_cond=bulk_cond_fem_rast, fr_cond=fr_cond)

            # Convert tensor field to Voigt form and rearrange for storage
            rasterized_input_voigt = tn_to_voigt(rasterized_input)
            rasterized_input_voigt = rasterized_input_voigt.reshape(*n_steps, rasterized_input_voigt.shape[-1]).T

            # Save input and output data to Zarr
            zarr_file["inputs"][sample_idx, ...] = rasterized_input_voigt
            zarr_file["outputs"][sample_idx, :] = fine_res

        return fine_res, coarse_res

    @staticmethod
    def rasterize_save_to_zarr(zarr_file_path, config, sample_idx, bulk_cond_values, bulk_cond_points, dfn, fr_cond,
                               fem_grid_rast, n_steps, fine_res=None):
        """
        Rasterizes the simulation inputs (bulk and fracture conductivity) and stores them into a Zarr file.

        :param zarr_file_path: Path to the Zarr file where data should be written
        :param config: Configuration object or dictionary for controlling options
        :param sample_idx: Index in the Zarr dataset at which to store the sample
        :param bulk_cond_values: Tensor or array with bulk conductivity values
        :param bulk_cond_points: Coordinates corresponding to bulk_cond_values
        :param dfn: Discrete Fracture Network definition used for rasterization
        :param fr_cond: Fracture conductivity tensor or values
        :param fem_grid_rast: Rasterized FEM grid used for mapping conductivities
        :param n_steps: Shape tuple of the FEM grid (e.g. (64, 64, 64))
        :param fine_res: (Optional) Output value(s) to store under "outputs" dataset
        :return: None
        """

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
        """
        Interpolate bulk conductivity tensor values from scattered points onto a regular grid.

        :param bulk_cond_values: Array of bulk conductivity tensors at scattered points
        :param bulk_cond_points: Coordinates of the points corresponding to bulk_cond_values
        :param grid_rast: Raster grid object providing barycenter points for interpolation
        :return: Interpolated conductivity tensors at the grid barycenters
        """
        # Interpolate the scattered data onto the regular grid
        x_unique = np.unique(bulk_cond_points[:, 0])
        y_unique = np.unique(bulk_cond_points[:, 1])
        z_unique = np.unique(bulk_cond_points[:, 2])
        nx, ny, nz = len(x_unique), len(y_unique), len(z_unique)

        sort_idx = np.lexsort((bulk_cond_points[:, 2], bulk_cond_points[:, 1], bulk_cond_points[:, 0]))
        # points_sorted = subdomain_bulk_cond_points[sort_idx]
        values_sorted = bulk_cond_values[sort_idx]
        values_grid = values_sorted.reshape((nx, ny, nz, 3, 3))

        interp = sc_interpolate.RegularGridInterpolator((x_unique, y_unique, z_unique), values_grid, method='nearest')
        target_points = grid_rast.barycenters()  # shape: (M, 3)

        # from scipy.spatial import Delaunay
        # # Create Delaunay triangulation of known points
        # tri = Delaunay(bulk_cond_points)
        # # Check if barycenters are inside the convex hull
        # inside_mask = tri.find_simplex(target_points) >= 0
        # # inside_mask is a boolean array, True if inside hull, False if outside
        # outside_mask = ~inside_mask
        #
        # print(f"Points inside convex hull: {np.sum(inside_mask)}")
        # print(f"Points outside convex hull: {np.sum(outside_mask)}")
        #
        # # Given unique grid axes (1D arrays)
        # # x_unique = np.unique(bulk_cond_points[:, 0])
        # # y_unique = np.unique(subdomain_bulk_cond_points[:, 1])
        # # z_unique = np.unique(subdomain_bulk_cond_points[:, 2])
        #
        # print("x_unique ", x_unique)
        # # print("y_unique ", y_unique)
        # # print("z_unique ", z_unique)
        #
        # # Compute bounding box corners
        # bbox_min = np.array([x_unique.min(), y_unique.min(), z_unique.min()])
        # bbox_max = np.array([x_unique.max(), y_unique.max(), z_unique.max()])
        #
        # # Your points to check (M x 3)
        # points = target_points
        #
        # # Test if points lie inside bounding box (inclusive)
        # inside_bbox_mask = np.all((points >= bbox_min) & (points <= bbox_max), axis=1)
        #
        # # Summary
        # print(f"Points inside bounding box: {np.sum(inside_bbox_mask)} / {len(points)}")
        # print(f"Points outside bounding box: {len(points) - np.sum(inside_bbox_mask)}")
        #
        # outside_points = points[~inside_bbox_mask]
        # print("outside points ", outside_points)

        resized_data = interp(target_points)
        return resized_data

    @staticmethod
    def get_equivalent_cond_tn(fr_media, config, sample_dir, bulk_cond_values, bulk_cond_points, dimensions, mesh_step, fr_region_map=None, sample_seed=None):
        bc_pressure_gradient = [1, 0, 0]
        cond_file, fr_cond, fr_region_map = DFMSim3D._run_sample(bc_pressure_gradient, fr_media, config, sample_dir, bulk_cond_values,
                                                  bulk_cond_points, dimensions, mesh_step, sample_seed=sample_seed)
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
    def get_fracture_to_matrix_ratio(fr_cond, bulk_cond_fem_rast, available_ratios):
        fracture_cond = fr_cond[:, 0, 0]  # shape (N,)
        fracture_mean = np.mean(fracture_cond)
        fracture_median = np.median(fracture_cond)
        # 2. Extract matrix conductivity from the bulk (e.g. (1,1) component)
        matrix_cond = bulk_cond_fem_rast[:, 1, 1]  # shape (N,)
        matrix_cond = matrix_cond[matrix_cond > 0]  # avoid invalid values
        matrix_mean = np.mean(matrix_cond)
        matrix_median = np.median(matrix_cond)

        print("fracture mean ", fracture_mean)
        print("fracture median ", fracture_median)

        print("matrix mean ", matrix_mean)
        print("matrix median ", matrix_median)

        sample_ratio_mean = fracture_mean / matrix_mean
        sample_ratio_median = fracture_median / matrix_median

        print("sample_ratio_mean ", sample_ratio_mean)
        print("sample_ratio_median ", sample_ratio_median)


        # 4. Find closest available ratio
        log_sample_ratio = np.log10(sample_ratio_mean)
        print("log_sample_ratio ", log_sample_ratio)
        closest_ratio = available_ratios[np.argmin(np.abs(np.log10(available_ratios) - log_sample_ratio))]
        print("closest ratio mean ", closest_ratio)

        # log_sample_ratio = np.log10(sample_ratio_median)
        # print("log_sample_ratio ", log_sample_ratio)
        # closest_ratio = available_ratios[np.argmin(np.abs(np.log10(available_ratios) - log_sample_ratio))]
        # print("closest ratio median ", closest_ratio)
        return closest_ratio

    @staticmethod
    def _prepare_raster_inputs(config, dfn_to_homogenization, bulk_cond_values, bulk_cond_points, fem_grid_rast):
        """Prepares FEM raster, fracture conductivity, and reshaped Voigt tensors."""
        fem_grid_n_steps = fem_grid_rast.grid.shape
        print("fem grid n steps ", fem_grid_n_steps)

        # Convert bulk conductivity to raster grid
        bulk_cond_fem_rast = DFMSim3D._bulk_cond_to_rast_grid(
            bulk_cond_values, bulk_cond_points, fem_grid_rast.grid
        )

        # Compute fracture conductivity if needed
        fr_cond, fr_cross_section = [], []
        if len(dfn_to_homogenization) > 0:
            fr_cross_section, fr_cond = DFMSim3D.fr_conductivity(
                dfn_to_homogenization, cross_section_factor=1e-4
            )

        # Convert to Voigt
        bulk_cond_fem_rast_voigt = tn_to_voigt(bulk_cond_fem_rast)
        bulk_cond_fem_rast_voigt = bulk_cond_fem_rast_voigt.reshape(
            *fem_grid_n_steps, bulk_cond_fem_rast_voigt.shape[-1]
        ).T

        # Rasterize including fractures if needed
        if len(dfn_to_homogenization) == 0:
            rasterized_input = bulk_cond_fem_rast
        else:
            rasterized_input = DFMSim3D.rasterize(
                fem_grid_rast,
                dfn_to_homogenization,
                bulk_cond=bulk_cond_fem_rast,
                fr_cond=fr_cond,
            )

        rasterized_input_voigt = tn_to_voigt(rasterized_input)
        rasterized_input_voigt = rasterized_input_voigt.reshape(
            *fem_grid_n_steps, rasterized_input_voigt.shape[-1]
        ).T

        # Pick NN model path (depending on fracture/matrix ratio)
        closest_frac_bulk_cond_ratio = None
        if "nn_path_cond_frac" in config["sim_config"]:
            available_cond_frac_ratios = list(config["sim_config"]["nn_path_cond_frac"].keys())
            print("available ratios ", available_cond_frac_ratios)
            closest_frac_bulk_cond_ratio = DFMSim3D.get_fracture_to_matrix_ratio(
                fr_cond, bulk_cond_fem_rast, available_cond_frac_ratios
            )

        return rasterized_input_voigt, bulk_cond_fem_rast_voigt, closest_frac_bulk_cond_ratio

    @staticmethod
    def _load_or_init_model(config, closest_frac_bulk_cond_ratio):
        """Loads the neural network model + transforms if not already loaded."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if DFMSim3D.model is None:
            if closest_frac_bulk_cond_ratio is not None:
                nn_path = config["sim_config"]["nn_path_cond_frac"][closest_frac_bulk_cond_ratio]
            else:
                nn_path = config["sim_config"]["nn_path"]
            print("nn_path ", nn_path)

            study = load_study(nn_path)
            model_path = get_saved_model_path(nn_path, study.best_trial)
            model_kwargs = study.best_trial.user_attrs["model_kwargs"]

            DFMSim3D.model = study.best_trial.user_attrs["model_class"](**model_kwargs)

            if not torch.cuda.is_available():
                DFMSim3D.checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
            else:
                DFMSim3D.checkpoint = torch.load(model_path)

            DFMSim3D.inverse_transform = get_inverse_transform(study, results_dir=nn_path)
            DFMSim3D.transform = get_transform(study, results_dir=nn_path)

        DFMSim3D.model.load_state_dict(DFMSim3D.checkpoint["best_model_state_dict"])
        DFMSim3D.model = DFMSim3D.model.to(memory_format=torch.channels_last_3d)
        DFMSim3D.model.to(device).eval()

        return device

    @staticmethod
    def _predict_from_zarr(zarr_file_path, batch_size=1):
        """Runs inference using DFMSim3D.model and returns predicted conductivity tensors."""
        dataset_for_prediction = DFM3DDataset(
            zarr_path=zarr_file_path,
            init_transform=DFMSim3D.transform[0],
            input_transform=DFMSim3D.transform[1],
            output_transform=DFMSim3D.transform[2],
            return_centers_bulk_avg=True,
        )

        dset_prediction_loader = torch.utils.data.DataLoader(
            dataset_for_prediction, batch_size=batch_size, shuffle=False
        )

        dict_centers, dict_cond_tn_values = [], []

        device = next(DFMSim3D.model.parameters()).device

        with torch.inference_mode():
            for _, sample in enumerate(dset_prediction_loader):
                inputs, targets, centers, bulk_features_avg = sample
                inputs = inputs.to(memory_format=torch.channels_last_3d).float().to(device)

                predictions = DFMSim3D.model(inputs)
                predictions = np.squeeze(predictions)

                inv_predictions = torch.squeeze(
                    DFMSim3D.inverse_transform(torch.reshape(predictions, (*predictions.shape, 1, 1)))
                )

                if dataset_for_prediction.init_transform is not None:
                    if len(inv_predictions.shape) > 1:
                        bulk_features_avg = bulk_features_avg.view(-1, 1)
                    inv_predictions *= bulk_features_avg

                inv_predictions_numpy = inv_predictions.numpy()
                if len(inv_predictions_numpy.shape) == 1:
                    inv_predictions_numpy = np.expand_dims(inv_predictions_numpy, axis=0)

                if len(inv_predictions.numpy().shape) == 1:
                    dict_cond_tn_values.extend([list(voigt_to_tn(inv_predictions_numpy))])
                else:
                    dict_cond_tn_values.extend(list(voigt_to_tn(inv_predictions_numpy)))

                centers_tuples = map(tuple, centers.numpy())
                dict_centers.extend(centers_tuples)

        return dict(zip(dict_centers, dict_cond_tn_values))


    @staticmethod
    def rasterize_at_once_zarr(config, dfn_to_homogenization, bulk_cond_values, bulk_cond_points, fem_grid_rast,
                               n_subdomains_per_axes):
        zarr_file_path = DFMSim3D.create_zarr_file(
            os.getcwd(), n_samples=int(n_subdomains_per_axes ** 3), config_dict=config, centers=True
        )

        rasterized_input_voigt, bulk_cond_fem_rast_voigt, closest_frac_bulk_cond_ratio = DFMSim3D._prepare_raster_inputs(config, dfn_to_homogenization, bulk_cond_values, bulk_cond_points,
                                            fem_grid_rast)

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

        num_subdomains = num_subdomains_x * num_subdomains_y * num_subdomains_z
        # Generate all index combinations
        all_indices = itertools.product(range(num_subdomains_x), range(num_subdomains_y), range(num_subdomains_z))

        zarr_batch_size = 1

        # Process in batches
        for start in range(0, num_subdomains, zarr_batch_size):
            end = min(start + zarr_batch_size, num_subdomains)

            # Preallocate batch arrays
            batch_centers = np.zeros((end - start, 3), dtype=np.float32)
            batch_inputs = np.zeros((end - start, *(
            rasterized_input_voigt.shape[0], subdomain_pixel_size, subdomain_pixel_size, subdomain_pixel_size)),
                                    dtype=np.float32)
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

                batch_centers[batch_index] = (center_x, center_y, center_z)
                batch_inputs[batch_index] = rasterized_input_voigt[:,
                                            i * pixel_stride: i * pixel_stride + subdomain_pixel_size,
                                            j * pixel_stride: j * pixel_stride + subdomain_pixel_size,
                                            l * pixel_stride: l * pixel_stride + subdomain_pixel_size]

                batch_bulk_avg[batch_index] = np.mean(bulk_cond_fem_rast_voigt[:,
                                                      i * pixel_stride: i * pixel_stride + subdomain_pixel_size,
                                                      j * pixel_stride: j * pixel_stride + subdomain_pixel_size,
                                                      l * pixel_stride: l * pixel_stride + subdomain_pixel_size],
                                                      axis=(1, 2, 3))

            # Batch write to Zarr
            zarr_file["centers"][start:end] = batch_centers
            zarr_file["inputs"][start:end] = batch_inputs
            zarr_file["bulk_avg"][start:end] = batch_bulk_avg

        DFMSim3D._load_or_init_model(config, closest_frac_bulk_cond_ratio)
        return DFMSim3D._predict_from_zarr(zarr_file_path, batch_size=1)

    @staticmethod
    def rasterize_at_once_zarr_irregular_stride(config, dfn_to_homogenization, bulk_cond_values, bulk_cond_points,
                                                fem_grid_rast, hom_block_centers):
        zarr_file_path = DFMSim3D.create_zarr_file(
            os.getcwd(), n_samples=int(len(hom_block_centers) ** 3), config_dict=config, centers=True
        )

        rasterized_input_voigt, bulk_cond_fem_rast_voigt, closest_frac_bulk_cond_ratio = DFMSim3D._prepare_raster_inputs(config, dfn_to_homogenization,
                                                                       bulk_cond_values, bulk_cond_points, fem_grid_rast)

        domain_box = config["sim_config"]["geometry"]["domain_box"]
        subdomain_box = config["sim_config"]["geometry"]["subdomain_box"]
        subdomain_pixel_size = config["sim_config"]["geometry"]["n_voxels"][0]

        orig_domain_min = -np.array(config["sim_config"]["geometry"]["orig_domain_box"]) / 2
        orig_domain_max = np.array(config["sim_config"]["geometry"]["orig_domain_box"]) / 2

        # derived
        pixel_size = subdomain_box[0] / subdomain_pixel_size  # units per pixel
        total_pixels = int(np.round(domain_box[0] / pixel_size))  # ~150
        half_block_px = subdomain_pixel_size // 2  # 32

        # Map centers from units -> pixel coordinates (float)
        domain_min = -domain_box[0] / 2
        centers_pixels_f = (hom_block_centers - domain_min) / pixel_size

        # Round or keep as floats depending on whether you require integer pixel centers
        centers_pixels = np.ceil(centers_pixels_f).astype(int)

        print(
            "pixel size:{}, total_pixels: {}, half_block_px: {}, domain_min: {}, centers_pixels_f: {}, center_pixels: {}".format(
                pixel_size, total_pixels, half_block_px, domain_min, centers_pixels_f, centers_pixels))

        zarr_file = zarr.open(zarr_file_path, mode='r+')

        all_indices = np.array(list(itertools.product(centers_pixels, repeat=3)))
        num_subdomains = all_indices.shape[0]

        print("rasterized_input_voigt.shape ", rasterized_input_voigt.shape)
        print("bulk_cond_fem_rast_voigt.shape ", bulk_cond_fem_rast_voigt.shape)

        # Process each subdomain one at a time (batch size = 1)
        for idx in range(num_subdomains):
            print("idx ", idx)
            px, py, pz = all_indices[idx]

            print("px: {}, py: {}, pz: {}".format(px, py, pz))

            # Compute subdomain slices (clamped)
            sx = max(0, min(px - half_block_px, rasterized_input_voigt.shape[1] - subdomain_pixel_size))
            sy = max(0, min(py - half_block_px, rasterized_input_voigt.shape[2] - subdomain_pixel_size))
            sz = max(0, min(pz - half_block_px, rasterized_input_voigt.shape[3] - subdomain_pixel_size))

            print("sx: {}, sy: {}, sz: {}".format(sx, sy, sz))
            # Physical center in units
            center_phys = np.array([px, py, pz], dtype=np.float32) * pixel_size + domain_min
            # Clamp to domain bounds
            center_phys = np.clip(center_phys, orig_domain_min, orig_domain_max)

            print("center_phys ", center_phys)

            # Extract subdomain
            subdomain_input = rasterized_input_voigt[:,
                              sx:sx + subdomain_pixel_size,
                              sy:sy + subdomain_pixel_size,
                              sz:sz + subdomain_pixel_size]

            # print("sx: {},  sx + subdomain_pixel_size: {}".format(sx, sx + subdomain_pixel_size))
            # print("sy: {},  sy + subdomain_pixel_size: {}".format(sy, sy + subdomain_pixel_size))
            # print("sz: {},  sz + subdomain_pixel_size: {}".format(sz, sz + subdomain_pixel_size))

            # Compute bulk average
            subdomain_bulk_avg = np.mean(bulk_cond_fem_rast_voigt[:,
                                         sx:sx + subdomain_pixel_size,
                                         sy:sy + subdomain_pixel_size,
                                         sz:sz + subdomain_pixel_size], axis=(1, 2, 3))

            # Write batch to Zarr
            zarr_file["centers"][idx] = center_phys
            zarr_file["inputs"][idx] = subdomain_input
            zarr_file["bulk_avg"][idx] = subdomain_bulk_avg

        DFMSim3D._load_or_init_model(config, closest_frac_bulk_cond_ratio)
        return DFMSim3D._predict_from_zarr(zarr_file_path, batch_size=1)

    @staticmethod
    def fine_SRF_from_homogenization(dfn, config, sample_seed):
        """
        Generate a fine-scale stochastic random field (SRF) from homogenization.

        :param dfn: List of fracture objects with attribute `r`.
        :param config: Simulation configuration dictionary containing geometry and SRF setup.
        :param sample_seed: Random seed for reproducibility of SRF generation.
        :return: Tuple (bulk_cond_values, bulk_cond_points)
            - bulk_cond_values: np.ndarray, conductivity tensor values after homogenization.
            - bulk_cond_points: np.ndarray, corresponding spatial coordinates of the conductivity tensors.
        """

        # Extract steps from config
        fine_step = config["fine"]["step"]
        coarse_step = config["coarse"]["step"]
        print(f"fine step: {fine_step}, coarse step: {coarse_step}")

        orig_domain_box = config["sim_config"]["geometry"]["orig_domain_box"]

        # Reverse level parameters (to align with finest to coarsest order)
        reversed_level_params = list(np.squeeze(config["sim_config"]["level_parameters"]))[::-1]

        # Identify current and previous level indices
        current_level_index = int(reversed_level_params.index(fine_step))
        assert current_level_index == len(reversed_level_params) - 1, "Only the coarsest level is supported"

        previous_level_index = current_level_index - 1
        print(f"current level index: {current_level_index}, previous level index: {previous_level_index}")

        # Prepare config for homogenization
        config_for_homogenization = copy.deepcopy(config)
        config_for_homogenization["fine"]["step"] = reversed_level_params[previous_level_index]
        config_for_homogenization["coarse"]["step"] = reversed_level_params[current_level_index]
        config_for_homogenization = DFMSim3D.configure_homogenization_geometry_params(config_for_homogenization)

        # Update geometry boxes
        sub_domain_box = config_for_homogenization["sim_config"]["geometry"]["subdomain_box"]

        domain_box = [
            orig_domain_box[0] + sub_domain_box[0],
            orig_domain_box[1] + sub_domain_box[1],
            orig_domain_box[2] + sub_domain_box[2],
        ]
        config_for_homogenization["sim_config"]["geometry"]["domain_box"] = domain_box
        config_for_homogenization["sim_config"]["geometry"]["hom_domain_box"] = domain_box

        import re
        config_for_homogenization["fine"]["cond_tn_pop_file"] = re.sub(r"l_step_\d+", f'l_step_{config_for_homogenization["fine"]["step"]}', config_for_homogenization["fine"]["cond_tn_pop_file"])
        config_for_homogenization["fine"]["cond_tn_pop_coords_file"] = re.sub(r"l_step_\d+",
                                                                       f'l_step_{config_for_homogenization["fine"]["step"]}',
                                                                       config_for_homogenization["fine"]["cond_tn_pop_coords_file"])

        print("use larger domain domain box", domain_box)

        # Split DFN into homogenization and fine lists
        dfn_to_homogenization_list = []
        dfn_to_fine_list = []

        for fr in dfn:
            if reversed_level_params[previous_level_index] <= fr.r <= reversed_level_params[current_level_index]:
                print("to hom fr.r", fr.r)
                dfn_to_homogenization_list.append(fr)
            elif reversed_level_params[current_level_index] < fr.r < orig_domain_box[0]:
                print("coarse/new fine fr.r", fr.r)
                dfn_to_fine_list.append(fr)

        # Generate SRF at the finest level
        if previous_level_index == 0:
            # Define FEM grid for conditional SRF generation
            hom_boxes_per_domain = config_for_homogenization["sim_config"]["geometry"]["n_nonoverlap_subdomains"]
            n_steps_cond_grid_size = int(hom_boxes_per_domain * 16)
            n_steps_cond_grid = (n_steps_cond_grid_size,) * 3

            fem_grid_cond = fem.fem_grid(
                n_steps_cond_grid_size,
                n_steps_cond_grid,
                fem.Fe.Q(dim=3),
                origin=-n_steps_cond_grid_size / 2,
            )
            print("fine SRF FEM GRID COND", fem_grid_cond)

            # SRF generation using gstools if enabled
            if config_for_homogenization["sim_config"].get("gstools_effective", False):
                print("gstools effective")
                femgrid_barycenters = fem_grid_cond.grid.barycenters()

                bulk_cond_values, bulk_cond_points = DFMSim3D.generate_grid_cond(
                    (femgrid_barycenters[:, 0],
                     femgrid_barycenters[:, 1],
                     femgrid_barycenters[:, 2]),
                    config_for_homogenization,
                    seed=sample_seed,
                    mode="fft",
                )
                bulk_cond_points = femgrid_barycenters
            else:
                bulk_cond_values, bulk_cond_points = DFMSim3D.generate_grid_cond(
                    fem_grid_cond.grid.barycenters(),
                    config,
                    seed=sample_seed,
                )

        elif previous_level_index in config["sim_config"]["levels_fine_srf_from_population"]:
            # Generate SRF from precomputed tensor population
            bulk_model = SRFFromTensorPopulation(config_for_homogenization)
            bulk_cond_values, bulk_cond_points = bulk_model.generate_field(
                reuse_sample=False,
                location_population=False,
            )

        # -----------------
        # Upscaling process
        # -----------------
        cond_tensors_for_coarse, _ = DFMSim3D.coarse_bulk_srf_generation(
            config_for_homogenization,
            bulk_cond_values,
            bulk_cond_points,
            dfn_to_homogenization_list,
        )

        # Flatten dictionary into arrays
        bulk_cond_values = np.squeeze(np.array(list(cond_tensors_for_coarse.values())))
        bulk_cond_points = np.array(list(cond_tensors_for_coarse.keys()))

        print("bulk cond values ", bulk_cond_values)
        print("bulk cond points ", bulk_cond_points)

        return bulk_cond_values, bulk_cond_points

    @staticmethod
    def coarse_bulk_srf_generation(config, bulk_cond_values, bulk_cond_points, dfn_to_homogenization_list, no_homogenization_flag=False):
        print("##############")
        print("coarse_bulk_srf_generation")
        print("###############")
        sim_config = config["sim_config"]
        subdomain_box = config["sim_config"]["geometry"]["subdomain_box"]
        hom_block_centers = config["sim_config"]["geometry"]["hom_block_centers"]
        n_nonoverlap_subdomains = config["sim_config"]["geometry"]["n_nonoverlap_subdomains"]
        n_subdomains_per_axes = config["sim_config"]["geometry"]["n_subdomains_per_axes"]
        hom_domain_box = config["sim_config"]["geometry"]["hom_domain_box"]
        pred_cond_tensors_homo = None

        print("bulk_cond_values.shape ", bulk_cond_values.shape)

        print("subdomain_box: {}, hom_block_centers: {}, n_nonoverlap_subdomains: {}, n_subdomains_per_axes: {}, hom_domain_box: {}".format(subdomain_box,
                                                                                                                                            hom_block_centers,
                                                                                                                                            n_nonoverlap_subdomains,
                                                                                                                                            n_subdomains_per_axes,
                                                                                                                                            hom_domain_box) )

        # Do not carry out homogenization at all
        if no_homogenization_flag:
            return dict(zip(map(tuple, bulk_cond_points), bulk_cond_values))

        dfn_to_homogenization = stochastic.FractureSet.from_list(dfn_to_homogenization_list)


        # Carry out homogenization
        # Use Surrogate
        if "nn_path" in config["sim_config"]:
            domain_size = config["sim_config"]["geometry"]["domain_box"]

            print('domain_size ', domain_size)

            # Homogenization block size is fixed
            if "hom_box_fixed" in sim_config and sim_config["hom_box_fixed"]:
                print("HOM BOX FIXED RASTERISATION")
                n_steps_per_axes = config["sim_config"]["geometry"]["n_voxels"][0]
                print("(domain_size[0] / subdomain_box[0]) ", (domain_size[0] / subdomain_box[0]))
                print("n_steps_per_axes ", n_steps_per_axes)

                num_voxels_per_axis = int(np.ceil((domain_size[0] / subdomain_box[0]) * n_steps_per_axes))
                print("num_voxels_per_axis ", num_voxels_per_axis)
                fem_grid_n_steps = [num_voxels_per_axis] * 3
                print("fem_grid_n_steps ", fem_grid_n_steps)

                pr = cProfile.Profile()
                pr.enable()

                print("(domain_size[0] - subdomain_box[0]) / 2 ", (domain_size[0] - subdomain_box[0]) / 2)
                hom_block_centers = hom_block_centers - ((domain_size[0] - subdomain_box[0]) / 2)
                print("shifted hom block centers ", hom_block_centers)

                fem_grid_rast = fem.fem_grid(domain_size, fem_grid_n_steps, fem.Fe.Q(dim=3),
                                             origin=-domain_size[0] / 2)
                cond_tensors = DFMSim3D.rasterize_at_once_zarr_irregular_stride(config, dfn_to_homogenization,
                                                                                bulk_cond_values,
                                                                                bulk_cond_points, fem_grid_rast,
                                                                                hom_block_centers)

                pr.disable()
                ps = pstats.Stats(pr).sort_stats('cumtime')
                ps.print_stats(25)

            else:
                n_steps_per_axes = config["sim_config"]["geometry"]["n_voxels"][0]
                fem_grid_n_steps = [n_steps_per_axes * n_nonoverlap_subdomains] * 3

                pr = cProfile.Profile()
                pr.enable()

                fem_grid_rast = fem.fem_grid(domain_size, fem_grid_n_steps, fem.Fe.Q(dim=3),
                                             origin=-domain_size[0] / 2)
                cond_tensors = DFMSim3D.rasterize_at_once_zarr(config, dfn_to_homogenization, bulk_cond_values,
                                                               bulk_cond_points, fem_grid_rast,
                                                               n_subdomains_per_axes)

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

            # pr = cProfile.Profile()
            # pr.enable()

            cond_tensors, pred_cond_tensors_homo = DFMSim3D.homogenization(config, dfn_to_homogenization,
                                                                           dfn_to_homogenization_list,
                                                                           bulk_cond_values, bulk_cond_points)

            # pr.disable()
            # ps = pstats.Stats(pr).sort_stats('cumtime')
            # ps.print_stats(35)
            #
            # # print("cond tensors ", cond_tensors)
            # # print("pred cond tensors homo ", pred_cond_tensors_homo)
            #
            # print("current dir ", current_dir)
            #
            # DFMSim3D._save_tensors(pred_cond_tensors_homo,
            #                        file=os.path.join(current_dir, DFMSim3D.PRED_COND_TN_FILE))
            #
            # pred_hom_bulk_cond_values, pred_hom_bulk_cond_points = np.squeeze(
            #     np.array(list(pred_cond_tensors_homo.values()))), np.array(list(pred_cond_tensors_homo.keys()))
            #
            # # print("cond tensors rast ", cond_tensors)
            # # print("cond tensors homo ", cond_tensors_homo)
            # # print("pred cond tensors homo ", pred_cond_tensors_homo)
            # #
            # # exit()

        return cond_tensors, pred_cond_tensors_homo


    @staticmethod
    def fine_bulk_srf_generation(config, fem_grid_cond_domain_size, sample_seed, dfn=None):
        fine_step = config["fine"]["step"]
        coarse_step = config["coarse"]["step"]
        sim_config = config["sim_config"]
        dimensions = config["sim_config"]["geometry"]["orig_domain_box"]
        n_nonoverlap_subdomains = config["sim_config"]["geometry"]["n_nonoverlap_subdomains"]

        bulk_cond_values_for_fine_sample, bulk_cond_points_for_fine_sample = None, None

        # If the finest level
        if list(np.squeeze(config["sim_config"]["level_parameters"], axis=1)).index(config["fine"]["step"]) == (
                len(np.squeeze(config["sim_config"]["level_parameters"], axis=1)) - 1):
            print("If the finest level")

            fem_grid_cond_domain_size = int(fem_grid_cond_domain_size)
            print("fem grid cond domain size ", fem_grid_cond_domain_size)
            # n_steps_cond_grid = (fem_grid_cond_domain_size, fem_grid_cond_domain_size, fem_grid_cond_domain_size)
            # 16x16x16 cond values generated per homogenization block
            n_steps_cond_grid = (n_nonoverlap_subdomains * 16, n_nonoverlap_subdomains * 16, n_nonoverlap_subdomains * 16)

            # 1LMC
            if coarse_step == 0:
                n_steps_cond_grid = (fem_grid_cond_domain_size, fem_grid_cond_domain_size, fem_grid_cond_domain_size)
                bulk_cond_values = {}
                bulk_cond_points = {}
            else:
                fem_grid_cond = fem.fem_grid(fem_grid_cond_domain_size, n_steps_cond_grid, fem.Fe.Q(dim=3),
                                             origin=-fem_grid_cond_domain_size / 2)
                print("fem grid cond ", fem_grid_cond)
                generate_grid_cond_start_time = time.time()

                if "gstools_effective" in sim_config and sim_config["gstools_effective"]:
                    print("gstools effective")
                    femgrid_barycenters = fem_grid_cond.grid.barycenters()
                    bulk_cond_values, bulk_cond_points = DFMSim3D.generate_grid_cond(
                        (femgrid_barycenters[:, 0], femgrid_barycenters[:, 1], femgrid_barycenters[:, 2]),
                        config, seed=sample_seed,
                        mode="fft")
                    bulk_cond_points = femgrid_barycenters
                else:
                    bulk_cond_values, bulk_cond_points = DFMSim3D.generate_grid_cond(fem_grid_cond.grid.barycenters(),
                                                                                     config, seed=sample_seed)
                print("time_generate_grid_cond ", time.time() - generate_grid_cond_start_time)
                print("bulk cond values ", bulk_cond_values.shape)
                print("bulk cond points ", bulk_cond_points.shape)

        # Generate fine SRF based on population of tensors
        elif list(np.squeeze(config["sim_config"]["level_parameters"], axis=1)).index(config["fine"]["step"]) in \
                config["sim_config"]["levels_fine_srf_from_population"]:
            print("Generate fine SRF based on population of tensors")

            assert "cond_tn_pop_file" in config["fine"] or "pred_cond_tn_pop_file" in config["fine"]

            bulk_model = SRFFromTensorPopulation(config)

            location_population = False
            if "generate_srf_from_hom_location_population" in config["sim_config"]:
                level_index = list(np.squeeze(config["sim_config"]["level_parameters"], axis=1)).index(config["fine"]["step"])
                if level_index in config["sim_config"]["generate_srf_from_hom_location_population"]:
                    location_population = config["sim_config"]["generate_srf_from_hom_location_population"][level_index]

            bulk_cond_values, bulk_cond_points = bulk_model.generate_field(reuse_sample=False, location_population=location_population)

            ###########
            # Bulk cond values for fine sample - use the exactly same 'hom centers' as coarse sample on a finer level
            ###########
            bulk_cond_values_for_fine_sample, bulk_cond_points_for_fine_sample = DFMSim3D.extract_subdomain(
                bulk_cond_values, bulk_cond_points, (0, 0, 0), dimensions)
            print('bulk_cond_points_for_fine_sample ', bulk_cond_points_for_fine_sample)
            print("bulk_cond_points_for_fine_sample.shape ", bulk_cond_points_for_fine_sample.shape)

        else:
            bulk_cond_values, bulk_cond_points = DFMSim3D.fine_SRF_from_homogenization(dfn, config, sample_seed)

        return bulk_cond_values, bulk_cond_points, bulk_cond_values_for_fine_sample, bulk_cond_points_for_fine_sample

    @staticmethod
    def config_cond_pop_files(config):
        # Update config with tensor population file paths if they exist
        cond_tn_pop = os.path.join(config["fine"]["common_files_dir"], DFMSim3D.COND_TN_POP_FILE)
        if os.path.exists(cond_tn_pop):
            config["fine"]["cond_tn_pop_file"] = cond_tn_pop
        cond_tn_coords_pop = os.path.join(config["fine"]["common_files_dir"], DFMSim3D.COND_TN_POP_COORDS_FILE)
        if os.path.exists(cond_tn_coords_pop):
            config["fine"]["cond_tn_pop_coords_file"] = cond_tn_coords_pop
        sample_cond_tns = os.path.join(config["fine"]["common_files_dir"], DFMSim3D.SAMPLES_COND_TNS_DIR)
        if os.path.exists(sample_cond_tns):
            config["fine"]["sample_cond_tns"] = sample_cond_tns

        return config

    @staticmethod
    def _calculate_subdomains(coarse_step, geometry, hom_box_size=None, fixed=False):
        """
        Calculate subdomains based on the domain size and homogenization box size.

        :param coarse_step: Step size for the coarse grid. If <= 0, a fallback geometry is returned.
        :param geometry: Dictionary containing geometry info. Must include "orig_domain_box" and optionally "subdomain_box".
        :param hom_box_size: Size of homogenization box. If None, defaults to 1.5 * coarse_step.
        :param fixed: If True, uses the fixed-box logic ("hom_box_fixed" mode). If False, uses the adaptive-box logic.
        :return: Tuple containing:
            - subdomain_box (list of float): Size of subdomain in each dimension [x, y, z].
            - n_nonoverlap_subdomains (int): Number of non-overlapping subdomains along each axis.
            - n_subdomains_per_axes (int): Number of total subdomains along each axis.
            - hom_block_centers (list of float): List of center positions (only non-empty if fixed=True).
        """
        if coarse_step <= 0:
            return geometry["subdomain_box"], 4, 4, [] if fixed else None

        if hom_box_size is None:
            hom_box_size = coarse_step * 1.5

        domain_box_size = geometry["orig_domain_box"][0]

        # Fixed hom block size
        if fixed:
            n_centers = domain_box_size / hom_box_size
            n_subdomains_per_axes = int(np.round(n_centers * 2 + 1))
            n_nonoverlap_subdomains = int(n_centers + 1)
            centers = np.linspace(0, domain_box_size, n_subdomains_per_axes)

            return (
                [hom_box_size] * 3,
                n_nonoverlap_subdomains,
                n_subdomains_per_axes,
                centers,
            )
        # Hom block size is adjusted to cover the domain with overlap 1/2
        else:
            n_centers = np.round(domain_box_size / hom_box_size)
            n_subdomains_per_axes = int(np.round(n_centers * geometry["pixel_stride_div"] + 1))
            hom_box_size = domain_box_size / n_centers

            return (
                [hom_box_size] * 3,
                int(n_centers + 1),
                n_subdomains_per_axes,
                []
            )

    @staticmethod
    def configure_homogenization_geometry_params(config):
        """
        Configure the simulation geometry including subdomain information.
        :param config: Configuration dictionary containing "sim_config" and "geometry".
        """
        fine_step = config["fine"]["step"]
        coarse_step = config["coarse"]["step"]
        sim_config = config["sim_config"]

        # Homogenization block size is determined by fine step and a multiplication factor
        if "hom_box_fine_step_mult" in sim_config:
            subdomain_box, n_nonoverlap_subdomains, n_subdomains_per_axes, hom_block_centers = \
                DFMSim3D._calculate_subdomains(
                    coarse_step,
                    config["sim_config"]["geometry"],
                    hom_box_size=fine_step * sim_config["hom_box_fine_step_mult"],
                    fixed=False,
                )
        # Fixed homogenization block size
        elif sim_config.get("hom_box_fixed", False):
            subdomain_box, n_nonoverlap_subdomains, n_subdomains_per_axes, hom_block_centers = \
                DFMSim3D._calculate_subdomains(
                    coarse_step,
                    config["sim_config"]["geometry"],
                    fixed=True,
                )
        else:
            subdomain_box, n_nonoverlap_subdomains, n_subdomains_per_axes, hom_block_centers = \
                DFMSim3D._calculate_subdomains(
                    coarse_step,
                    config["sim_config"]["geometry"],
                )

        print(
            "_calculate_subdomains_hom_box_fixed "
            "subdomain box: {}, n subdomains per axes: {}, n_nonoverlap_subdomains: {}, hom_block_centers: {}".format(
                subdomain_box, n_subdomains_per_axes, n_nonoverlap_subdomains, hom_block_centers
            )
        )

        config["sim_config"]["geometry"]["domain_box"] = config["sim_config"]["geometry"]["orig_domain_box"]
        config["sim_config"]["geometry"]["subdomain_box"] = subdomain_box
        config["sim_config"]["geometry"]["n_subdomains_per_axes"] = n_subdomains_per_axes
        config["sim_config"]["geometry"]["n_nonoverlap_subdomains"] = n_nonoverlap_subdomains
        config["sim_config"]["geometry"]["n_subdomains"] = n_subdomains_per_axes ** 3
        config["sim_config"]["geometry"]["hom_block_centers"] = hom_block_centers

        return config

    @staticmethod
    def calculate(config, seed):
        """
        Method that actually run the calculation, it's called from mlmc.tool.pbs_job.PbsJob.calculate_samples()
        Calculate fine and coarse sample and also extract their results
        :param config: dictionary containing simulation configuration, LevelSimulation.config_dict (set in level_instance)
        :param seed: random seed, int
        :return: List[fine result, coarse result], both flatten arrays (see mlmc.sim.synth_simulation.calculate())
        """
        sample_seed = config["sim_config"]["seed"] + seed
        np.random.seed(sample_seed)

        sample_dir = Path(os.getcwd()).absolute()
        basename = os.path.basename(sample_dir)
        sample_idx = int(basename.split('S')[1])

        config = DFMSim3D.config_cond_pop_files(config)

        scratch_sample_path = None
        if os.environ.get("SCRATCHDIR") is not None:
            scratch_sample_path = os.path.join(os.environ.get("SCRATCHDIR"), basename)
            shutil.move(sample_dir, scratch_sample_path)
            os.chdir(scratch_sample_path)
        current_dir = Path(os.getcwd()).absolute()

        ############################################
        # Generate homogenization samples - used for creating a dataset to learn a surrogate
        if "generate_hom_samples" in config["sim_config"] and config["sim_config"]["generate_hom_samples"]:
            return DFMSim3D.calculate_hom_sample(config, current_dir, sample_idx, sample_seed)
        ############################################

        fine_step = config["fine"]["step"]
        coarse_step = config["coarse"]["step"]

        domain_size = config["sim_config"]["geometry"]["domain_box"][0]  # 15  # 100
        fem_grid_cond_domain_size = config["sim_config"]["geometry"]["domain_box"][0] + 3 #+ int(config["sim_config"]["geometry"]["domain_box"][0] * 0.1)
        fr_domain_size = config["sim_config"]["geometry"]["fractures_box"][0]
        fr_range = config["sim_config"]["geometry"]["pow_law_sample_range"]  # (5, fr_domain_size)

        config = DFMSim3D.configure_homogenization_geometry_params(config)

        sim_config = config["sim_config"]
        geom = sim_config["geometry"]

        #@TODO: use larger domain to calculate hom samples
        orig_domain_box = config["sim_config"]["geometry"]["orig_domain_box"]
        if coarse_step > 0 and "use_larger_domain" in config["sim_config"] and config["sim_config"]["use_larger_domain"]:
            sub_domain_box = config["sim_config"]["geometry"]["subdomain_box"]
            config["sim_config"]["geometry"]["domain_box"] = [orig_domain_box[0] + sub_domain_box[0],
                                                              orig_domain_box[1] + sub_domain_box[1],
                                                              orig_domain_box[2] + sub_domain_box[2]]
            fem_grid_cond_domain_size = orig_domain_box[0] + sub_domain_box[0] + fine_step
            hom_domain_box = [orig_domain_box[0] + sub_domain_box[0], orig_domain_box[1] + sub_domain_box[1], orig_domain_box[2] + sub_domain_box[2]]
            print("use larger domain domain box ", config["sim_config"]["geometry"]["domain_box"])

            config["sim_config"]["geometry"]["hom_domain_box"] = hom_domain_box

        dimensions = config["sim_config"]["geometry"]["orig_domain_box"]

        ###################
        ### Fine sample ###
        ###################
        fine_res = [0, 0, 0, 0, 0, 0]
        fine_sample_start_time = time.time()

        print("list(np.squeeze(config[sim_config][level_parameters], axis=1))[-1] ", list(np.squeeze(config["sim_config"]["level_parameters"], axis=1))[-1])

        ### Discrete fracture network generation ###
        dfn = DFMSim3D.fracture_random_set(sample_seed, fr_range, sim_config["work_dir"], max_frac=geom["n_frac_limit"])
        ##############################################

        ### Bulk conductivity generation ###
        bulk_cond_values, bulk_cond_points, bulk_cond_values_for_fine_sample, bulk_cond_points_for_fine_sample = DFMSim3D.fine_bulk_srf_generation(config, fem_grid_cond_domain_size, sample_seed, dfn)

        print("bulk_cond_values_for_fine_sample ", bulk_cond_values_for_fine_sample)
        if bulk_cond_values_for_fine_sample is None:
            bulk_cond_values_for_fine_sample = bulk_cond_values
            bulk_cond_points_for_fine_sample = bulk_cond_points

        #####################################

        dfn = stochastic.FractureSet.from_list([
            fr for fr in dfn
            if (list(np.squeeze(config["sim_config"]["level_parameters"], axis=1))[-1] <= fr.r <= orig_domain_box[0])
               and (fr.r >= fine_step)
        ])
        fr_media = FracturedMedia.fracture_cond_params(dfn, 1e-4, 0.00001)

        ### Simulation run ###
        if "flow_sim" in config["sim_config"] and config["sim_config"]["flow_sim"]:
            bc_pressure_gradient = [1, 0, 0]
            cond_file, fr_cond, fr_region_map = DFMSim3D._run_sample_flow(bc_pressure_gradient, fr_media, config, current_dir,
                                                                          bulk_cond_values_for_fine_sample, bulk_cond_points_for_fine_sample, dimensions, mesh_step=config["fine"]["step"], sample_seed=sample_seed)

            conv_check = check_conv_reasons(os.path.join(current_dir, "flow123.0.log"))
            if not conv_check:
                raise Exception("fine sample not converged")

            print("cond file ", cond_file)
            fine_res = DFMSim3D.get_outflow(current_dir)

            fine_res = [fine_res[0], fine_res[0], fine_res[0], fine_res[0], fine_res[0], fine_res[0]]
            print("fine res ", fine_res)

            file_mapping = {
                "flow_fields.pvd": "flow_field_fine.pvd",
                "flow_fields": "flow_fields_fine",
                "flow_fields.msh": "flow_fields_fine.msh",
                "water_balance.yaml": "water_balance_fine.yaml",
                "water_balance.txt": "water_balance_fine.txt",
            }
            DFMSim3D.rename_files(file_mapping)

        else:
            fine_res, fr_cond, fr_region_map = DFMSim3D.get_equivalent_cond_tn(fr_media, config, current_dir, bulk_cond_values_for_fine_sample, bulk_cond_points_for_fine_sample, dimensions, mesh_step=config["fine"]["step"], sample_seed=sample_seed)
            conv_check = check_conv_reasons(os.path.join(current_dir, "flow123.0.log"))
            if not conv_check:
                raise Exception("fine sample not converged")

        file_mapping = {
            "flow123.0.log": "fine_flow123.0.log",
            "flow_upscale_templ.yaml_stderr": "fine_flow_upscale_templ.yaml_stderr",
            "flow_upscale_templ.yaml_stdout": "fine_flow_upscale_templ.yaml_stdout",
            "flow_upscale_templ.yaml.yaml": "fine_flow_upscale_templ.yaml.yaml",
            "homo_cube.brep": "fine_homo_cube.brep",
            "homo_cube_healed.msh2": "fine_homo_cube_healed.msh2",
            "homo_cube.heal_stats.yaml": "fine_homo_cube.heal_stats.yaml",
            "homo_cube.msh2": "fine_homo_cube.msh2",
            "input_fields.msh": "fine_input_fields.msh",
            "bulk_cond_data.npy": "fine_bulk_cond_data.npy",
            "fr_cond_data.npy": "fine_fr_cond_data.npy",
            "voxel_fracture_sizes.npy": "fine_voxel_fracture_sizes.npy"
        }

        DFMSim3D.rename_files(file_mapping)

        #################################################################


        #####################
        ### Coarse sample ###
        #####################
        coarse_sample_start_time = time.time()
        coarse_res = [0, 0, 0, 0, 0, 0]

        no_homogenization_flag = config["sim_config"]["no_homogenization_flag"] if "no_homogenization_flag" in config["sim_config"] else False

        # Return for the coarsest level
        if coarse_step == 0:
            return fine_res, coarse_res

        ### DFN preparation - for homogenization and for coarse sample ###
        dfn_to_homogenization_list = []
        dfn_to_coarse_list = []
        hom_max_r = coarse_step
        if "hom_box_fine_step_mult" in sim_config:  # homogenization block size determined by fine_step
            hom_max_r = fine_step * sim_config["hom_box_fine_step_mult"] * (
                        2 / 3)  # maximal fracture size is 2/3 homogenization block size

        print("hom_max_r ", hom_max_r)
        for fr in dfn:
            if fr.r <= hom_max_r:
                print("to hom fr.r ", fr.r)
                dfn_to_homogenization_list.append(fr)
            else:
                print("coarse fr.r ", fr.r)
                dfn_to_coarse_list.append(fr)

        dfn_to_coarse = stochastic.FractureSet.from_list(dfn_to_coarse_list)
        coarse_fr_media = FracturedMedia.fracture_cond_params(dfn_to_coarse, 1e-4, 0.00001)
        ###############################################################################

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

        cond_tensors_for_coarse, pred_cond_tensors_homo = DFMSim3D.coarse_bulk_srf_generation(config, bulk_cond_values, bulk_cond_points, dfn_to_homogenization_list, no_homogenization_flag)


        print("homogenization time ", time.time() - homogenization_start_time)

        print("homogenized cond tensors ", np.array(list(cond_tensors_for_coarse.values())).shape)

        DFMSim3D._save_tensors(cond_tensors_for_coarse, file=os.path.join(current_dir, DFMSim3D.COND_TN_FILE))

        DFMSim3D._save_tensor_values(cond_tensors_for_coarse, file=os.path.join(config["coarse"]["common_files_dir"], DFMSim3D.COND_TN_VALUES_FILE))
        DFMSim3D._save_tensor_coords(cond_tensors_for_coarse, file=os.path.join(config["coarse"]["common_files_dir"], DFMSim3D.COND_TN_COORDS_FILE))

        hom_bulk_cond_values, hom_bulk_cond_points = np.squeeze(np.array(list(cond_tensors_for_coarse.values()))), np.array(list(cond_tensors_for_coarse.keys()))

        if pred_cond_tensors_homo:
            pred_hom_bulk_cond_values, pred_hom_bulk_cond_points = np.squeeze(
                np.array(list(pred_cond_tensors_homo.values()))), np.array(list(pred_cond_tensors_homo.keys()))

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
                hom_coarse_mapping = {
                    "flow123.0.log": "hom_coarse_flow123.0.log",
                    "flow_fields.pvd": "hom_coarse_flow_field_coarse.pvd",
                    "flow_fields": "hom_coarse_flow_fields_coarse",
                    "flow_fields.msh": "hom_coarse_flow_fields_coarse.msh",
                    "input_fields.msh": "hom_coarse_input_fields.msh",
                    "water_balance.yaml": "hom_coarse_water_balance_coarse.yaml",
                    "water_balance.txt": "hom_coarse_water_balance_coarse.txt",
                    "bulk_cond_data.npy": "hom_coarse_bulk_cond_data.npy",
                    "fr_cond_data.npy": "hom_coarse_fr_cond_data.npy",
                    "voxel_fracture_sizes.npy": "hom_coarse_voxel_fracture_sizes.npy"
                }
                DFMSim3D.rename_files(hom_coarse_mapping)

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

            files_to_rename = [
                "flow123.0.log",
                "flow_fields.pvd",
                "flow_fields",
                "flow_fields.msh",
                "input_fields.msh",
                "water_balance.yaml",
                "water_balance.txt",
                "bulk_cond_data.npy",
                "fr_cond_data.npy",
                "voxel_fracture_sizes.npy"
            ]
            DFMSim3D.rename_files(files_to_rename, file_prefix=file_prefix)

        else:
            coarse_res, fr_cond_coarse, fr_region_map_coarse = DFMSim3D.get_equivalent_cond_tn(coarse_fr_media, config, current_dir, hom_bulk_cond_values, hom_bulk_cond_points, dimensions, mesh_step=config["coarse"]["step"])
            conv_check = check_conv_reasons(os.path.join(current_dir, "flow123.0.log"))
            if not conv_check:
                raise Exception("coarse sample not converged")

            if pred_cond_tensors_homo:
                hom_coarse_mapping = {
                    "flow123.0.log": "hom_coarse_flow123.0.log",
                    "flow_fields.pvd": "hom_coarse_flow_field_coarse.pvd",
                    "flow_fields": "hom_coarse_flow_fields_coarse",
                    "flow_fields.msh": "hom_coarse_flow_fields_coarse.msh",
                    "input_fields.msh": "hom_coarse_input_fields.msh",
                    "water_balance.yaml": "hom_coarse_water_balance_coarse.yaml",
                    "water_balance.txt": "hom_coarse_water_balance_coarse.txt",
                    "bulk_cond_data.npy": "hom_coarse_bulk_cond_data.npy",
                    "fr_cond_data.npy": "hom_coarse_fr_cond_data.npy",
                    "voxel_fracture_sizes.npy": "hom_coarse_voxel_fracture_sizes.npy"
                }
                DFMSim3D.rename_files(hom_coarse_mapping)

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

            files_to_rename = [
                "flow123.0.log",
                "flow_fields.pvd",
                "flow_fields",
                "flow_fields.msh",
                "input_fields.msh",
                "water_balance.yaml",
                "water_balance.txt",
                "bulk_cond_data.npy",
                "fr_cond_data.npy",
                "voxel_fracture_sizes.npy"]
            DFMSim3D.rename_files(files_to_rename, file_prefix=file_prefix)


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
    def generate_grid_cond(barycenters, config, seed, mode=None):
        """
        Generates a 3D conductivity field using one of the GSTools models based on the given configuration.

        :param barycenters: Array-like of shape (N, 3), coordinates of cell centers (e.g. from a grid or mesh)
        :param config: Dictionary containing simulation configuration, including conductivity model parameters
        :param seed: Integer random seed to ensure reproducibility of the random field generation
        :param mode: Optional override for the conductivity generation mode (e.g., 'log-normal', 'normal')
        :return: A NumPy array representing the generated bulk conductivity field at the given barycenters
        """
        bulk_conductivity = config["sim_config"]['bulk_conductivity']

        # Override mode if explicitly specified
        if mode is not None:
            bulk_conductivity['mode'] = mode

        # if "cond_tn_pop_file" in config["fine"] or "pred_cond_tn_pop_file" in config["fine"]:
        #     #@TODO: sample from saved population of conductivity tensors
        #     bulk_conductivity = config["sim_config"]['bulk_conductivity']
        #     #config_dict["mean_log_conductivity"] = bulk_conductivity["mean_log_conductivity"]
        #     bulk_model = SRFFromTensorPopulation(config)
        #     # bulk_cond_tn_pop_file = config_dict["fine"]["cond_tn_pop_file"]
        #     #bulk_model = BulkChoose(finer_level_path)
        # else:
        # Handle marginal distribution if provided
        if "marginal_distr" in bulk_conductivity and bulk_conductivity["marginal_distr"] is not False:
            means, cov = DFMSim3D.calculate_cov(bulk_conductivity["marginal_distr"])
            bulk_conductivity["mean_log_conductivity"] = means
            bulk_conductivity["cov_log_conductivity"] = cov
            del bulk_conductivity["marginal_distr"]

        # Choose the appropriate conductivity generation model based on configuration
        if "gstools" in config["sim_config"] and config["sim_config"]["gstools"]:
            bulk_model = GSToolsBulk3D(**bulk_conductivity)
            bulk_model.seed = seed

        # Uncomment if vector field support is added later
        # elif "gstools_vector" in config["sim_config"] and config["sim_config"]["gstools_vector"]:
        #     bulk_model = GSToolsBulk3DVectorField(**bulk_conductivity)
        #     bulk_model.seed = seed
        #     print("bulkfieldsgstools vector")

        elif "gstools_effective" in config["sim_config"] and config["sim_config"]["gstools_effective"]:
            bulk_model = GSToolsBulk3DEffective(**bulk_conductivity)
            bulk_model.structured = True
            bulk_model.seed = seed
            print("bulkfieldsgstools effective")
        else:
            raise NotImplementedError("No valid GSTools model specified in the configuration.")

        return bulk_model.generate_field(barycenters)

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

    @staticmethod
    def file_name_with_timestamp(file):
        # Extract directory and base filename
        directory, base_filename = os.path.split(file)
        name, ext = os.path.splitext(base_filename)

        # Create timestamped filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        new_filename = f"{name}_{timestamp}{ext}"
        return os.path.join(directory, new_filename)

    @staticmethod
    def _save_tensor_values(cond_tensors, file):
        full_path = DFMSim3D.file_name_with_timestamp(file)

        values_to_save = np.array([v[0] for v in cond_tensors.values()])
        np.savez_compressed(full_path, data=values_to_save)

    @staticmethod
    def _save_tensor_coords(cond_tensors, file):
        full_path = DFMSim3D.file_name_with_timestamp(file)

        coords_to_save = np.array(list(cond_tensors.keys()))
        np.savez_compressed(full_path, data=coords_to_save)

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
                    mesh_step, cond_file=None, center=[0, 0, 0], fr_region_map=None, sample_seed=None, regular_grid_interp=True):

        fr_cond = None
        if cond_file is None:
            cond_file, fr_cond, fr_region_map, mesh_regions = DFMSim3D.create_mesh_fields(fr_media, bulk_cond_values, bulk_cond_points, dimensions,
                                                             mesh_step=mesh_step,
                                                             sample_dir=sample_dir,
                                                             work_dir=config["sim_config"]["work_dir"],
                                                             center=center,
                                                             outflow_problem=True, fr_region_map=fr_region_map,
                                                             config=config, sample_seed=sample_seed, regular_grid_interp=regular_grid_interp
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
    def _run_sample(
            bc_pressure_gradient,
            fr_media,
            config,
            sample_dir,
            bulk_cond_values,
            bulk_cond_points,
            dimensions,
            mesh_step,
            cond_file=None,
            center=[0, 0, 0],
            fr_region_map=None,
            sample_seed=None,
            regular_grid_interp=True):
        """
        Runs a single simulation sample: generates the mesh with conductivity values,
        sets up the simulation using Flow123d, and executes it via Docker or Singularity.

        :param bc_pressure_gradient: Array-like or dict specifying the pressure gradient as boundary condition
        :param fr_media: Fracture media object containing fracture geometries and properties
        :param config: Dictionary containing simulation settings
        :param sample_dir: Directory to store simulation outputs for this sample
        :param bulk_cond_values: Values of bulk conductivity at given points
        :param bulk_cond_points: Spatial locations corresponding to bulk_cond_values
        :param dimensions: Tuple specifying the simulation domain size (x, y, z)
        :param mesh_step: Mesh resolution in each spatial direction
        :param cond_file: Optional; path to an existing conductivity mesh file. If None, it will be created
        :param center: Optional; center of the simulation domain (default is [0, 0, 0])
        :param fr_region_map: Optional; predefined fracture-to-region mapping. If None, it will be generated
        :param sample_seed: Optional; random seed used during mesh generation
        :return: Tuple of (cond_file, fr_cond, fr_region_map) where:
            - cond_file: Path to the generated or provided conductivity mesh file
            - fr_cond: Fracture conductivity array or object (if generated)
            - fr_region_map: Mapping of fracture regions (if generated)
        """
        fr_cond = None

        if cond_file is None:
            cond_file, fr_cond, fr_region_map, mesh_regions = DFMSim3D.create_mesh_fields(
                fr_media, bulk_cond_values, bulk_cond_points, dimensions,
                mesh_step=mesh_step,
                sample_dir=sample_dir,
                work_dir=config["sim_config"]["work_dir"],
                center=center,
                fr_region_map=fr_region_map,
                config=config,
                sample_seed=sample_seed, regular_grid_interp=regular_grid_interp
            )

        if "run_local" in config["sim_config"] and config["sim_config"]["run_local"]:
            flow_cfg = dotdict(
                flow_executable=[
                    "docker",
                    "run",
                    "-v",
                    "{}:{}".format(os.getcwd(), os.getcwd()),
                    "flow123d/ci-gnu:4.0.0a01_94c428",
                    "flow123d",
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
                    "/storage/liberec3-tul/home/martin_spetlik/flow_4_0_0.sif",
                    "flow123d",
                    "--output_dir",
                    os.getcwd()
                ],
                mesh_file=cond_file,
                pressure_grad=bc_pressure_gradient,
            )

        f_template = "flow_upscale_templ.yaml"
        shutil.copy(os.path.join(config["sim_config"]["work_dir"], f_template), sample_dir)

        with workdir_mng(sample_dir):
            flow_out = call_flow(flow_cfg, f_template, flow_cfg)

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

    def make_summary(done_list):
        results = {problem.basename: problem.summary() for problem in done_list}

        #print("results ", results)

        with open("summary.yaml", "w") as f:
            yaml.dump(results, f)

    @staticmethod
    def rename_files(files: Union[List[str], Dict[str, str]], file_prefix: Optional[str] = "") -> None:
        """
        Rename or move files using either a mapping or a list with optional prefix.

        :param files: Either:
                      - A list of existing filenames (prefix will be applied)
                      - A dictionary {source_filename: target_filename} for explicit renames
        :param file_prefix: Optional prefix to add when a list is provided. Ignored if a dictionary is passed.
        :return: None
        """
        if isinstance(files, dict):
            # Use explicit mapping
            for src, dst in files.items():
                if os.path.exists(src):
                    shutil.move(src, dst)
                    print(f"Renamed '{src}' -> '{dst}'")
        elif isinstance(files, list):
            # Apply prefix to each file
            if file_prefix is None:
                file_prefix = ""
            for src in files:
                if os.path.exists(src):
                    dst = f"{file_prefix}{src}"
                    shutil.move(src, dst)
                    print(f"Renamed '{src}' -> '{dst}'")
        else:
            raise ValueError("files must be a list or a dictionary")



