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
from homogenization.both_sample import FlowProblem, BothSample
#from metamodel.cnn.datasets import create_dataset
from metamodel.cnn.postprocess.optuna_results import load_study, load_models, get_saved_model_path, get_inverse_transform, get_transform
from metamodel.cnn.datasets.dfm_dataset import DFMDataset
#from npy_append_array import NpyAppendArray


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

            yaml_template_h_vtk = os.path.join(common_files_dir, DFMSim.YAML_TEMPLATE_H_VTK)
            shutil.copyfile(self.base_yaml_file_homogenization_vtk, yaml_template_h_vtk)

            yaml_template = os.path.join(common_files_dir, DFMSim.YAML_TEMPLATE)
            shutil.copyfile(self.base_yaml_file, yaml_template)

            yaml_file = os.path.join(common_files_dir, DFMSim.YAML_FILE)
            self._substitute_yaml(yaml_template, yaml_file)

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
            samples_cond_tns = os.path.join(coarse_sim_common_files_dir, DFMSim.SAMPLES_COND_TNS_DIR)
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
                               task_size=1/fine_step/5,  #len(fine_mesh_data['points']) / job_weight,
                               calculate=DFMSim.calculate,
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

    @staticmethod
    def get_outer_polygon(center_x, center_y, box_size_x, box_size_y):
        bl_corner = [center_x - box_size_x / 2, center_y - box_size_y / 2]
        br_corner = [center_x + box_size_x / 2, center_y - box_size_y / 2]
        tl_corner = [center_x - box_size_x / 2, center_y + box_size_y / 2]
        tr_corner = [center_x + box_size_x / 2, center_y + box_size_y / 2]

        # print("center x: {}, y: {}".format(center_x, center_y))

        return [bl_corner, br_corner, tr_corner, tl_corner]


    @staticmethod
    def split_domain(config, dataset_path, n_split_subdomains, fractures, sample_dir, seed=None):
        domain_box = config["sim_config"]["geometry"]["domain_box"]
        x_subdomains = int(np.sqrt(n_split_subdomains))
        subdomain_box = [domain_box[0]/x_subdomains, domain_box[1]/x_subdomains]

        if x_subdomains == 1:
            sample_0_path = os.path.join(dataset_path, "sample_0")
            os.mkdir(sample_0_path)
            shutil.copy(os.path.join(sample_dir, "fields_fine_to_rast.msh"), os.path.join(sample_0_path, "fields_fine.msh"))
            shutil.copy(os.path.join(sample_dir, "mesh_fine_to_rast.msh"), os.path.join(sample_0_path, "mesh_fine.msh"))
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
                    outer_polygon = DFMSim.get_outer_polygon(center_x, center_y, box_size_x, box_size_y)
                    config["sim_config"]["geometry"]["outer_polygon"] = outer_polygon

                    while True:
                        if fractures is None:
                            fractures = DFMSim.generate_fractures(config)

                        fine_flow = FlowProblem.make_fine((config["fine"]["step"],
                                                           config["sim_config"]["geometry"]["fr_max_size"]),
                                                          fractures,
                                                          config, seed=None)#config["sim_config"]["seed"] + seed)
                        fine_flow.fr_range = [config["fine"]["step"], config["coarse"]["step"]]

                        cond_fields = config["center_cond_field"]
                        if "center_larger_cond_field" in config:
                            cond_fields = config["center_larger_cond_field"]

                        center_cond_field = DFMSim.eliminate_far_points(outer_polygon,
                                                                        cond_fields,
                                                                        fine_step=config["fine"]["step"])
                        try:
                            fine_flow.make_mesh(center_box=([center_x, center_y], subdomain_box))
                        except:
                            pass

                        if not os.path.exists("mesh_fine.msh"):
                            box_size_x += box_size_x * 0.05
                            box_size_y += box_size_y * 0.05
                            outer_polygon = DFMSim.get_outer_polygon(center_x, center_y, box_size_x, box_size_y)
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
    def rasterize_at_once(sample_dir, dataset_path, config, n_subdomains, fractures, hom_dir, seed=None):
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

        sample_0_path = DFMSim.split_domain(config, dataset_path, n_split_subdomains=4**split_exp, fractures=fractures, sample_dir=sample_dir, seed=seed)

        ####################
        ## Create dataset ##
        ####################
        process = subprocess.run(["bash", config["sim_config"]["create_dataset_script"], dataset_path, "{}".format(int(np.sqrt(total_n_pixels_x)))],
                                 capture_output=True, text=True)
        pred_cond_tensors = {}

        if process.returncode == 0:
            bulk_path = os.path.join(sample_0_path, "bulk.npz")
            fractures_path = os.path.join(sample_0_path, "fractures.npz")
            bulk = np.load(bulk_path)["data"]
            fractures = np.load(fractures_path)["data"]
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
            #print("rasterize n subdomains ", n_subdomains)
            for i in range(n_subdomains):
                #print("lx ", lx)
                center_y = -(subdomain_box[0] / 2 + (lx - subdomain_box[0]) / (n_subdomains - 1) * i - lx / 2)
                for j in range(n_subdomains):
                    sample_name = "sample_{}".format(k)
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

                    #print("bulk_subdomain[0,...] ", bulk_subdomain[0,...])

                    np.savez_compressed(os.path.join(f_dset_sample_dir, "bulk"), data=bulk_subdomain)
                    np.savez_compressed(os.path.join(f_dset_sample_dir, "fractures"), data=fractures_subdomain)

                    #print("sample name ", sample_name)
                    sample_center[sample_name] = (center_x, center_y)
                    k += 1

            if DFMSim.model is None:
                nn_path = config["sim_config"]["nn_path"]
                study = load_study(nn_path)
                model_path = get_saved_model_path(nn_path, study.best_trial)
                model_kwargs = study.best_trial.user_attrs["model_kwargs"]
                DFMSim.model = study.best_trial.user_attrs["model_class"](**model_kwargs)
                if not torch.cuda.is_available():
                    DFMSim.checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                else:
                    DFMSim.checkpoint = torch.load(model_path)
                DFMSim.inverse_transform = get_inverse_transform(study, results_dir=nn_path)
                DFMSim.transform = get_transform(study, results_dir=nn_path)

            ##
            # Create dataset
            ##
            dataset_for_prediction = DFMDataset(data_dir=final_dataset_path,
                                                init_transform=DFMSim.transform[0],
                                                input_transform=DFMSim.transform[1],
                                                output_transform=DFMSim.transform[2],
                                                two_dim=True)

            dset_prediction_loader = torch.utils.data.DataLoader(dataset_for_prediction, shuffle=False)
            with torch.no_grad():
                DFMSim.model.load_state_dict(DFMSim.checkpoint['best_model_state_dict'])
                DFMSim.model.eval()

                for i, sample in enumerate(dset_prediction_loader):
                    inputs, targets = sample
                    # print("intputs ", inputs)
                    inputs = inputs.float()
                    sample_n = dataset_for_prediction._bulk_file_paths[i].split('/')[-2]
                    center = sample_center[sample_n]
                    # if args.cuda and torch.cuda.is_available():
                    #    inputs = inputs.cuda()
                    predictions = DFMSim.model(inputs)
                    predictions = np.squeeze(predictions)

                    # if np.any(predictions < 0):
                    #     print("inputs ", inputs)
                    #     print("negative predictions ", predictions)

                    inv_predictions = torch.squeeze(
                        DFMSim.inverse_transform(torch.reshape(predictions, (*predictions.shape, 1, 1))))

                    #print("inv predictions ", inv_predictions)

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
    def homogenization(config, fractures=None, seed=None):
        #print("config ", config)
        sample_dir = os.getcwd()
        print("homogenization method")
        print("os.getcwd() ", os.getcwd())

        hom_dir_name = "homogenization"
        if "scratch_dir" in config and os.path.exists(config["scratch_dir"]):
            hom_dir_abs_path = os.path.join(config["scratch_dir"], hom_dir_name)
        else:
            hom_dir_abs_path = os.path.join(sample_dir, hom_dir_name)

        # os.chdir(config["scratch_dir"])
        if os.path.exists(hom_dir_abs_path):
            shutil.rmtree(hom_dir_abs_path)

        os.mkdir(hom_dir_abs_path)
        os.chdir(hom_dir_abs_path)

        h_dir = os.getcwd()
        print("h_dir os.getcwd() ", os.getcwd())

        os.mkdir("dataset")
        dataset_path = os.path.join(h_dir, "dataset")
        sim_config = config["sim_config"]

        #print("sim config ", sim_config)
        #print("rasterize at once ", config["sim_config"]["rasterize_at_once"])

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

        #print("n subdomains ", n_subdomains)
        #exit()

        n_subdomains = int(np.floor(np.sqrt(n_subdomains)))

        cond_tensors = {}
        pred_cond_tensors = {}
        percentage_sym_tn_diff = []
        time_measurements = []
        sample_center = {}

        fine_flow = None

        k = 0
        #
        # if n_subdomains == 1:
        #     DFMSim.run_single_subdomain()

        # print("domain box ", domain_box)
        # print("subdomain box ", subdomain_box)
        # exit()

        #if rasterize_at_once

        #print("lx ", lx)
        #print("n subdomains ", n_subdomains)

        if "rasterize_at_once" in config["sim_config"] and config["sim_config"]["rasterize_at_once"]:
            pred_cond_tensors = DFMSim.rasterize_at_once(sample_dir, dataset_path, config, n_subdomains, fractures, h_dir, seed=seed)
            #print("pred cond tensors ", pred_cond_tensors)
        else:

            create_hom_samples_start_time = time.time()
            for i in range(n_subdomains):
                #print("subdomain box ", subdomain_box)
                #if "outer_polygon" not in sim_config["geometry"]:
                # print("i ", i)
                # print("(lx - subdomain_box[0]) ", (lx - subdomain_box[0]))
                # print("(n_subdomains - 1) * i ", (n_subdomains - 1) * i)
                #i = 3

                center_x = subdomain_box[0] / 2 + (lx - subdomain_box[0]) / (n_subdomains - 1) * i - lx / 2
                #print("center x ", center_x)

                for j in range(n_subdomains):
                    start_time = time.time()
                    k += 1
                    #j = 8
                    #k = 53
                    #print("k ", k)
                    subdir_name = "i_{}_j_{}_k_{}".format(i, j, k)
                    #print("subdir_name ", subdir_name)
                    os.mkdir(subdir_name)
                    os.chdir(subdir_name)

                    center_y = subdomain_box[1] / 2 + (lx - subdomain_box[1]) / (n_subdomains - 1) * j - lx / 2

                    #print("center x:{} y:{}, k: {}".format(center_x, center_y, k))
                    # print("subdomain box ", subdomain_box)
                    # exit()
                    #center_x += center_x *0.01
                    #center_y += center_y * 0.01
                    #print("new center x:{} y:{}".format(center_x, center_y))

                    box_size_x = subdomain_box[0]
                    box_size_y = subdomain_box[1]

                    outer_polygon = DFMSim.get_outer_polygon(center_x, center_y, box_size_x, box_size_y)
                    sim_config["geometry"]["outer_polygon"] = outer_polygon
                    #print("work_dir ", work_dir)

                    #print("outer polygon ", outer_polygon)
                    #print("center x:{} y:{}".format(center_x, center_y))

                    sim_config["work_dir"] = work_dir
                    #config["homogenization"] = True

                    # if fractures is None:
                    #     fractures = DFMSim.generate_fractures(config)
                    #
                    #
                    # fine_flow = FlowProblem.make_fine((config["fine"]["step"],
                    #                                    config["sim_config"]["geometry"]["fr_max_size"]),
                    #                                   fractures,
                    #                                   config)
                    # fine_flow.fr_range = [config["fine"]["step"], config["coarse"]["step"]]
                    #
                    # cond_fields = config["center_cond_field"]
                    # if "center_larger_cond_field" in config:
                    #     cond_fields = config["center_larger_cond_field"]
                    #
                    # print("=== HOMOGENIZATION SAMPLE ===")
                    #
                    # # try:
                    # center_cond_field = DFMSim.eliminate_far_points(outer_polygon,
                    #                                                 cond_fields,
                    #                                                 fine_step=config["fine"]["step"])
                    # try:
                    #     fine_flow.make_mesh()
                    # except:
                    #     pass

                    #sim_config["work_dir"] = work_dir

                    while True:
                        if fractures is None:
                            fractures = DFMSim.generate_fractures(config)

                        if fine_flow is not None:
                            bulk_model = fine_flow.bulk_model
                            fine_flow = FlowProblem("fine", (config["fine"]["step"],
                                                           config["sim_config"]["geometry"]["fr_max_size"]), fractures, bulk_model, config)
                        else:
                            fine_flow = FlowProblem.make_fine((config["fine"]["step"],
                                                               config["sim_config"]["geometry"]["fr_max_size"]),
                                                              fractures,
                                                              config)
                        fine_flow.fr_range = [config["fine"]["step"], config["coarse"]["step"]]

                        cond_fields = config["center_cond_field"]
                        if "center_larger_cond_field" in config:
                            cond_fields = config["center_larger_cond_field"]

                        center_cond_field = DFMSim.eliminate_far_points(outer_polygon,
                                                                        cond_fields,
                                                                        fine_step=config["fine"]["step"])
                        try:
                            fine_flow.make_mesh(center_box=([center_x, center_y], subdomain_box))
                        except:
                            pass

                        if not os.path.exists("mesh_fine.msh"):
                            box_size_x += box_size_x * 0.05
                            box_size_y += box_size_y * 0.05
                            outer_polygon = DFMSim.get_outer_polygon(center_x, center_y, box_size_x, box_size_y)
                            sim_config["geometry"]["outer_polygon"] = outer_polygon
                            print("new outer polygon make_mesh failed", outer_polygon)
                            continue

                        #print("center cond field ", center_cond_field)
                        #fine_flow.interpolate_fields(center_cond_field, mode="linear")

                        fine_flow.interpolate_fields(center_cond_field, mode="linear")

                        # fine_flow.make_fields()

                        if "nn_path" in config["sim_config"] and "run_only_hom" in config["sim_config"] and config["sim_config"]["run_only_hom"]:
                            break

                        done = []
                        # exit()
                        # fine_flow.run() # @TODO replace fine_flow.run by DFMSim._run_sample()
                        # print("run samples ")
                        status, p_loads, outer_reg_names, conv_check = DFMSim._run_homogenization_sample(fine_flow, config)

                        done.append(fine_flow)
                        try:
                            if not conv_check:
                                raise Exception("homogenization sample not converged")
                            cond_tn, diff = fine_flow.effective_tensor_from_bulk(p_loads, outer_reg_names, fine_flow.basename)
                        except:
                            box_size_x += box_size_x * 0.05
                            box_size_y += box_size_y * 0.05
                            outer_polygon = DFMSim.get_outer_polygon(center_x, center_y, box_size_x, box_size_y)
                            sim_config["geometry"]["outer_polygon"] = outer_polygon
                            print("new outer polygon flow123d failed", outer_polygon)
                            os.remove("mesh_fine.msh")
                            continue
                        DFMSim.make_summary(done)
                        percentage_sym_tn_diff.append(diff)
                        break

                    # if n_subdomains == 1:
                    #     mesh_file = "/home/martin/Desktop/mesh_fine.msh"
                    #     shutil.copy(mesh_file, os.getcwd())
                    #     elids_same_value = {18: 17, 20: 19, 22: 21, 24: 23, 26: 25, 28: 27, 30: 29, 32: 31, 34: 33}
                    #
                    #     fine_flow.make_mesh(mesh_file)
                    #     fine_flow.make_fields(elids_same_value=elids_same_value)
                    #
                    #     #fine_flow.make_mesh()
                    #     #fine_flow.make_fields()
                    #     done = []
                    #     #exit()
                    #     # fine_flow.run() # @TODO replace fine_flow.run by DFMSim._run_sample()
                    #     #print("run samples ")
                    #     status, p_loads, outer_reg_names, conv_check = DFMSim._run_homogenization_sample(fine_flow, config)
                    #
                    #     done.append(fine_flow)
                    #     cond_tn, diff = fine_flow.effective_tensor_from_bulk(p_loads, outer_reg_names, fine_flow.basename, elids_same_value)
                    #     #print("cond_tn ", cond_tn)
                    #     #exit()
                    #     DFMSim.make_summary(done)
                    #     percentage_sym_tn_diff.append(diff)
                    # else:

                    #fine_flow.mesh = config["fine_mesh"]
                    #fine_flow.get_subdomain(outer_polygon)


                        #except Exception as e:
                        #    print("make mesh error", str(e))
                        #    exit()
                        #    continue


                    # except Exception as e:
                    #    print(str(e))

                    #exit()

                    pred_cond_tn = None
                    ######################
                    ## Employ metamodel ##
                    #####################
                    if "nn_path" in config["sim_config"]:
                        # import cProfile
                        # import pstats

                        # print("cwd ", os.getcwd())
                        # pr = cProfile.Profile()
                        # pr.enable()

                        start_time_nn = time.time()
                        sample_name = "sample_{}".format(k - 1)
                        dset_sample_dir = os.path.join(dataset_path, sample_name)
                        os.mkdir(dset_sample_dir)
                        shutil.copy("fields_fine.msh", dset_sample_dir)
                        shutil.copy("mesh_fine.msh", dset_sample_dir)
                        # shutil.copy("summary.yaml", dset_dir)
                        sample_center[sample_name] = (center_x, center_y)

                        # pred_cond_tensors[] = (center_x, center_y)

                        # print("cond_tn ", cond_tn)
                        # @TODO: save cond tn and center to npz file

                    if "run_only_hom" not in config["sim_config"] or \
                            ("run_only_hom" in config["sim_config"] and not config["sim_config"]["run_only_hom"]):
                        if n_subdomains > 1:
                            cond_tn_flatten = cond_tn[0].flatten()
                            if not np.any(np.isnan(cond_tn_flatten)):
                                #print("cond_tn_flatten ", cond_tn_flatten)
                                cond_tensors[(center_x, center_y)] = [cond_tn_flatten[0], (cond_tn_flatten[1]+cond_tn_flatten[2])/2, cond_tn_flatten[3]]
                                #print("symmetric cond tn ",  cond_tensors[(center_x, center_y)])

                                # if pred_cond_tn is not None:
                                #     pred_cond_tn_flatten = pred_cond_tn.flatten()
                                #     pred_cond_tensors[(center_x, center_y)] = pred_cond_tn_flatten
                                # print("pred cond tn ", pred_cond_tn_flatten)
                                # print("config[coarse common_files_dir] ", config["coarse"]["common_files_dir"])


                                # cond_tn_pop_file = os.path.join(config["coarse"]["common_files_dir"],
                                #                                 DFMSim.COND_TN_POP_FILE)
                                # with NpyAppendArray(cond_tn_pop_file, delete_if_exists=False) as npaa:
                                #     npaa.append(cond_tn_flatten)

                            dir_name = os.path.join(work_dir, subdir_name)
                            config["dir_name"] = dir_name

                    try:
                        shutil.move("fine", dir_name)
                        if os.path.exists(os.path.join(dir_name, "fields_fine.msh")):
                            os.remove(os.path.join(dir_name, "fields_fine.msh"))
                        shutil.move("fields_fine.msh", dir_name)
                        if os.path.exists(os.path.join(dir_name, "summary.yaml")):
                            shutil.move("summary.yaml", dir_name)
                        if os.path.exists(os.path.join(dir_name, "flow_fields.msh")):
                            shutil.move("flow_fields.msh", dir_name)
                        shutil.move("mesh_fine.msh", dir_name)
                        shutil.move("mesh_fine.brep", dir_name)
                        shutil.move("mesh_fine.tmp.geo", dir_name)
                        shutil.move("mesh_fine.tmp.msh", dir_name)
                        shutil.rmtree("fine")
                    except:
                        pass
                    os.chdir(h_dir)
                    time_measurements.append(time.time() - start_time)

            create_hom_samples_end_time = time.time()

            print("create hom samples time ", create_hom_samples_end_time - create_hom_samples_start_time)

            if "nn_path" in config["sim_config"]:
                print("CREATE DATASET SCRIPT RUN")

                cr_dset_start_time = time.time()
                process = subprocess.run(["bash", config["sim_config"]["create_dataset_script"], dataset_path, "{}".format(256)],
                                         capture_output=True, text=True)

                # exit_code = subprocess.call('./sample.sh')

                # print("process.stdout.decode('ascii') ", process.stdout.decode('ascii'))

                if process.returncode != 0:
                    raise Exception(process.stderr)
                # else:
                #     print("dataset sample created ", os.getcwd())
                cr_dset_end_time = time.time()

                # bulk_data_array, fractures_data_array = create_dataset.create_input(os.getcwd())
                # print("bulk_data_array ", bulk_data_array)

                load_model_start_time = time.time()
                if DFMSim.model is None:
                    nn_path = config["sim_config"]["nn_path"]
                    study = load_study(nn_path)

                    # print(" study.user_attrs ",  study.user_attrs)
                    model_path = get_saved_model_path(nn_path, study.best_trial)
                    model_kwargs = study.best_trial.user_attrs["model_kwargs"]
                    DFMSim.model = study.best_trial.user_attrs["model_class"](**model_kwargs)
                    if not torch.cuda.is_available():
                        DFMSim.checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                    else:
                        DFMSim.checkpoint = torch.load(model_path)
                    # DFMSim.checkpoint = torch.load(model_path)

                    DFMSim.inverse_transform = get_inverse_transform(study, results_dir=nn_path)
                    DFMSim.transform = get_transform(study, results_dir=nn_path)
                    #print("DFMSim.transform ", DFMSim.transform)

                    #DFMSim.dataset = joblib.load(os.path.join(config["sim_config"]["nn_path"], "dataset.pkl"))

                load_model_end_time = time.time()

                ##
                # Create dataset
                ##
                # print("os.getcwd() ", os.getcwd())
                cr_dset_loader_start_time = time.time()
                dataset_for_prediction = DFMDataset(data_dir=dataset_path,
                                                    # output_file_name=output_file_name,
                                                    init_transform=DFMSim.transform[0],
                                                    input_transform=DFMSim.transform[1],
                                                    output_transform=DFMSim.transform[2],
                                                    two_dim=True,
                                                    # input_channels=config[
                                                    #     "input_channels"] if "input_channels" in config else None,
                                                    # output_channels=config[
                                                    #     "output_channels"] if "output_channels" in config else None,
                                                    # fractures_sep=config[
                                                    #     "fractures_sep"] if "fractures_sep" in config else False,
                                                    # vel_avg=config["vel_avg"] if "vel_avg" in config else False
                                                    )

                # print("dataset for prediction ", dataset_for_prediction)
                # print("len(dataset for predictin) ", len(dataset_for_prediction))

                dset_prediction_loader = torch.utils.data.DataLoader(dataset_for_prediction, shuffle=False)

                cr_dset_loader_end_time = time.time()

                # print("dset prediction loader ", dset_prediction_loader)

                with torch.no_grad():
                    DFMSim.model.load_state_dict(DFMSim.checkpoint['best_model_state_dict'])
                    DFMSim.model.eval()

                    dset_pred_start_time = time.time()

                    for i, sample in enumerate(dset_prediction_loader):
                        inputs, targets = sample
                        # print("intputs ", inputs)
                        inputs = inputs.float()
                        sample_n = dataset_for_prediction._bulk_file_paths[i].split('/')[-2]
                        center = sample_center[sample_n]
                        # if args.cuda and torch.cuda.is_available():
                        #    inputs = inputs.cuda()
                        predictions = DFMSim.model(inputs)
                        predictions = np.squeeze(predictions)

                        inv_predictions = torch.squeeze(
                            DFMSim.inverse_transform(torch.reshape(predictions, (*predictions.shape, 1, 1))))

                        if dataset_for_prediction.init_transform is not None:
                            inv_predictions *= dataset_for_prediction._bulk_features_avg

                        pred_cond_tn = np.array([[inv_predictions[0], inv_predictions[1]],
                                                 [inv_predictions[1], inv_predictions[2]]])

                        if pred_cond_tn is not None:
                            pred_cond_tn_flatten = pred_cond_tn.flatten()

                            if not np.any(np.isnan(pred_cond_tn_flatten)):
                                pred_cond_tensors[center] = [pred_cond_tn_flatten[0], (pred_cond_tn_flatten[1]+pred_cond_tn_flatten[2])/2, pred_cond_tn_flatten[3]]
                                # pred_cond_tn_pop_file = os.path.join(config["coarse"]["common_files_dir"],
                                #                                      DFMSim.PRED_COND_TN_POP_FILE)
                                # with NpyAppendArray(pred_cond_tn_pop_file, delete_if_exists=False) as npaa:
                                #     npaa.append(pred_cond_tn_flatten)

                                #print("cond tn: {}, pred cond tn: {}".format(cond_tensors[center],pred_cond_tensors[center]))

                    dset_pred_end_time = time.time()

                end_time_nn = time.time()

                # pr.disable()
                # ps = pstats.Stats(pr).sort_stats('cumtime')
                # ps.print_stats()

                print("NN total time {}, cr_dset: {}, load_model: {}, dset_loader: {}, dset pred: {}".format(
                    end_time_nn - start_time_nn,
                    cr_dset_end_time - cr_dset_start_time,
                    load_model_end_time - load_model_start_time,
                    cr_dset_loader_end_time - cr_dset_loader_start_time,
                    dset_pred_end_time - dset_pred_start_time
                    ))

        # print("np.mean(percentage_sym_tn_diff) ", np.mean(percentage_sym_tn_diff))
        # print("time_measurements ", time_measurements)
        os.chdir(sample_dir)

        # print("cond tensors ", cond_tensors)
        # print("pred cond tensors ", pred_cond_tensors)
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
    def _calculate_subdomains(coarse_step, geometry):
        if coarse_step > 0:
            hom_box_size = coarse_step * 1.5
            domain_box_size = geometry["orig_domain_box"][0]
            n_centers = np.round(domain_box_size / hom_box_size)
            n_total_centers = n_centers * 2 + 1
            hom_box_size = domain_box_size / n_centers
        else:
            return geometry["subdomain_box"], 4
        return [hom_box_size, hom_box_size], n_total_centers**2

    @staticmethod
    def calculate(config, seed):
        """
        Method that actually run the calculation, it's called from mlmc.tool.pbs_job.PbsJob.calculate_samples()
        Calculate fine and coarse sample and also extract their results
        :param config: dictionary containing simulation configuration, LevelSimulation.config_dict (set in level_instance)
        :param seed: random seed, int
        :return: List[fine result, coarse result], both flatten arrays (see mlmc.sim.synth_simulation.calculate())
        """
        # print("config ", config)
        # exit()
        #print("seed ", seed)
        #print("sim_config seed", config["sim_config"]["seed"])
        #@TODO: check sample dir creation
        np.random.seed(config["sim_config"]["seed"] + seed)
        fractures = DFMSim.generate_fractures(config)
        #fractures = DFMSim.generate_fractures(config)

        coarse_step = config["coarse"]["step"]
        fine_step = config["fine"]["step"]

        #print("fine_step", config["fine"]["step"])
        #print("coarse_step", config["coarse"]["step"])

        times = {}
        fine_res = 0

        # try:
        #shutil.rmtree("fine")
        DFMSim._remove_files()
        # if os.path.exists("fields_fine.msh")
        #shutil.rmtree("fine")
        # except:
        #     pass

        # if coarse_step == 0:
        #     print("config fine", config["fine"])
        #     exit()
        gen_hom_samples = False
        if "generate_hom_samples" in config["sim_config"] and config["sim_config"]["generate_hom_samples"]:
            gen_hom_samples = True

        cond_tn_pop = os.path.join(config["fine"]["common_files_dir"], DFMSim.COND_TN_POP_FILE)
        if os.path.exists(cond_tn_pop):
            config["fine"]["cond_tn_pop_file"] = cond_tn_pop

        if "nn_path" in config["sim_config"]:
            pred_cond_tn_pop = os.path.join(config["fine"]["common_files_dir"], DFMSim.PRED_COND_TN_POP_FILE)
            if os.path.exists(pred_cond_tn_pop):
                config["fine"]["pred_cond_tn_pop_file"] = pred_cond_tn_pop

        sample_cond_tns = os.path.join(config["fine"]["common_files_dir"], DFMSim.SAMPLES_COND_TNS_DIR)
        if os.path.exists(sample_cond_tns):
            config["fine"]["sample_cond_tns"] = sample_cond_tns

        #print("config fine ", config["fine"])

        ####################
        ### fine problem ###
        ####################
        if "steps" in config["sim_config"]:
            fine_step_key = np.round(fine_step, 3)
            if fine_step_key in config["sim_config"]["steps"]:
                #print("fine step key ", fine_step_key)
                geometry = config["sim_config"]["steps"][fine_step_key]["geometry"]
                config["sim_config"]["geometry"]["domain_box"] = geometry["domain_box"]
                config["sim_config"]["geometry"]["subdomain_box"] = geometry["subdomain_box"]
                config["sim_config"]["geometry"]["fractures_box"] = geometry["fractures_box"]
                config["sim_config"]["geometry"]["n_subdomains"] = geometry["n_subdomains"]
            else:
                print("domain box", config["sim_config"]["geometry"]["domain_box"])
                # print("orig domain box from config", orig_domain_box)
                print("fractures box ", config["sim_config"]["geometry"]["fractures_box"])

                subdomain_box, n_subdomains = DFMSim._calculate_subdomains(coarse_step, config["sim_config"]["geometry"])

                print("calculated subdomain box: {}, n subdomains: {}".format(subdomain_box, n_subdomains))

                config["sim_config"]["geometry"]["domain_box"] = config["sim_config"]["geometry"]["orig_domain_box"]
                config["sim_config"]["geometry"]["subdomain_box"] = subdomain_box
                config["sim_config"]["geometry"]["n_subdomains"] = n_subdomains

        orig_domain_box = config["sim_config"]["geometry"]["orig_domain_box"]
        if coarse_step > 0 and "use_larger_domain" in config["sim_config"] and config["sim_config"]["use_larger_domain"]:
            # Larger bulk domain
            #print("orig domain box from config", orig_domain_box)
            sub_domain_box = config["sim_config"]["geometry"]["subdomain_box"]
            orig_frac_box = config["sim_config"]["geometry"]["fractures_box"]

            config["sim_config"]["geometry"]["domain_box"] = [orig_domain_box[0] + 2 * sub_domain_box[0], orig_domain_box[1]+ 2 * sub_domain_box[1]]
            hom_domain_box = [orig_domain_box[0] + sub_domain_box[0], orig_domain_box[1]+ sub_domain_box[1]]
            # if config["sim_config"]["rasterize_at_once"]:
            #     hom_domain_box = [orig_domain_box[0] + sub_domain_box[0]/2, orig_domain_box[1] + sub_domain_box[1]/2]

            #print("use larger domain domain box ", config["sim_config"]["geometry"]["domain_box"])

            #larger_domain_box = config["sim_config"]["geometry"]["domain_box"]

            # Larger fracture domain
            #print("orig frac box from config ", orig_frac_box)
            #sub_frac_domain_box = config["sim_config"]["geometry"]["subdomain_box"]
            config["sim_config"]["geometry"]["fractures_box"] = [orig_frac_box[0] + 2 * sub_domain_box[0],
                                                              orig_frac_box[1] + 2 * sub_domain_box[1]]

            #hom_domain_box = [orig_domain_box[0] + sub_domain_box[0], orig_domain_box[1] + sub_domain_box[1]]
            #print("domain box ", config["sim_config"]["geometry"]["domain_box"])

            fractures = DFMSim.generate_fractures(config)

            larger_dom_obj = FlowProblem.make_fine((config["fine"]["step"], config["sim_config"]["geometry"]["fr_max_size"]),
                                              fractures, config, seed=None)#config["sim_config"]["seed"] + seed)
            larger_dom_obj.fr_range = [config["fine"]["step"], larger_dom_obj.fr_range[1]]

            larger_dom_obj.make_mesh()
            larger_dom_obj.make_fields()
            config["center_larger_cond_field"] = larger_dom_obj._center_cond

            if os.path.exists("fields_fine.msh"):
                shutil.move("fields_fine.msh", "fields_fine_large.msh")
            if os.path.exists("mesh_fine.msh"):
                shutil.move("mesh_fine.msh", "mesh_fine_large.msh")

            #print("orig domain box ", orig_domain_box)
            config["sim_config"]["geometry"]["domain_box"] = orig_domain_box
            config["sim_config"]["geometry"]["fractures_box"] = orig_domain_box
            DFMSim._remove_files()

        fine_res = [0,0,0]

        if coarse_step > 0 and ("rasterize_at_once" in config["sim_config"] and config["sim_config"]["rasterize_at_once"]):
            # print("domain box ", config["sim_config"]["geometry"]["domain_box"])
            # print("hom domain box ", hom_domain_box)
            config["sim_config"]["geometry"]["domain_box"] = hom_domain_box
            config["sim_config"]["geometry"]["fractures_box"] = hom_domain_box
            fine_flow = FlowProblem.make_fine(
                (config["fine"]["step"], config["sim_config"]["geometry"]["fr_max_size"]), fractures, config, seed=None)#config["sim_config"]["seed"] + seed)
            fine_flow.fr_range = [config["fine"]["step"], config["coarse"]["step"]]
            fine_flow.make_mesh()
            if "center_larger_cond_field" in config:
                # fine_flow.make_mesh()
                center_cond_field = np.array(config["center_larger_cond_field"][0]), \
                                    np.array(config["center_larger_cond_field"][1])
                #print("rast center cond field ", center_cond_field)
                fine_flow.interpolate_fields(center_cond_field, mode="linear")
            shutil.move("fields_fine.msh", "fields_fine_to_rast.msh")
            shutil.move("mesh_fine.msh", "mesh_fine_to_rast.msh")

        config["sim_config"]["geometry"]["domain_box"] = orig_domain_box
        config["sim_config"]["geometry"]["fractures_box"] = orig_domain_box

        #print("domain box for fine ", config["sim_config"]["geometry"]["domain_box"])
        fine_flow = FlowProblem.make_fine((config["fine"]["step"], config["sim_config"]["geometry"]["fr_max_size"]), fractures, config, seed=None)#config["sim_config"]["seed"] + seed)
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
            if "center_larger_cond_field" in config:
                #fine_flow.make_mesh()
                center_cond_field = np.array(config["center_larger_cond_field"][0]), np.array(config["center_larger_cond_field"][1])
                fine_flow.interpolate_fields(center_cond_field, mode="linear")
                #print("fine center cond field ", center_cond_field)
            else:
                fine_flow.make_fields()

            config["center_cond_field"] = fine_flow._center_cond
            times['make_fields'] = time.time() - make_fields_start

        #fine_flow.run() # @TODO replace fine_flow.run by DFMSim._run_sample()
        #print("run samples ")

        fractures_for_coarse_sample = copy.deepcopy(fractures._reduced_fractures)

        if not gen_hom_samples:
            if "flow_sim" in config["sim_config"] and config["sim_config"]["flow_sim"]:
                fine_res, status, conv_check = DFMSim._run_sample(fine_flow, config)
                if not conv_check:
                    raise Exception("fine sample not converged")

                fine_res = [fine_res[0], fine_res[0], fine_res[0]]
                if os.path.exists("flow_fields.pvd"):
                    shutil.move("flow_fields.pvd", "flow_field_fine.pvd")
                if os.path.exists("flow_fields"):
                    shutil.move("flow_fields", "flow_fields_fine")
                if os.path.exists("flow_fields.msh"):
                    shutil.move("flow_fields.msh", "flow_fields_fine.msh")
            else:
                done = []
                status, p_loads, outer_reg_names, conv_check = DFMSim._run_homogenization_sample(fine_flow, config)
                if not conv_check:
                    raise Exception("fine sample not converged")
                done.append(fine_flow)
                cond_tn, diff = fine_flow.effective_tensor_from_bulk(p_loads, outer_reg_names, fine_flow.basename)
                #print("fine cond tn ", cond_tn)
                cond_tn = cond_tn[0]
                cond_tn[0, 1] = (cond_tn[0, 1] + cond_tn[1, 0]) / 2
                fine_res = cond_tn.flatten()
                fine_res = [fine_res[0], fine_res[1], fine_res[3]]

            print("fine res ", fine_res)

            if os.path.exists("flow_fields.pvd"):
                os.remove("flow_fields.pvd")
            if os.path.exists("flow_fields"):
                shutil.rmtree("flow_fields")

            if os.path.exists("flow_fields.msh"):
                shutil.move("flow_fields.msh", "flow_fields_fine.msh")

            # if "nn_path" in config["sim_config"] and \
            #         ("run_only_hom" not in config["sim_config"] or not config["sim_config"]["run_only_hom"]):
            #
            #     status, p_loads, outer_reg_names, conv_check = DFMSim._run_homogenization_sample(fine_flow, config, format="vtk")
            #     DFMSim.make_summary(done)
            #
            #     ff_fine_vtk = os.path.join(os.getcwd(), "flow_field_fine_vtk")
            #     if os.path.exists(ff_fine_vtk):
            #         shutil.rmtree(ff_fine_vtk)
            #     os.mkdir(ff_fine_vtk)
            #
            #     if os.path.exists("flow_fields.pvd"):
            #         shutil.move("flow_fields.pvd", ff_fine_vtk)
            #         # shutil.move("flow_fields.pvd", "flow_fields_coarse_vtk.pvd")
            #     if os.path.exists("flow_fields"):
            #         shutil.move("flow_fields", ff_fine_vtk)
            # if os.path.exists("flow_fields.pvd"):
            #     shutil.move("flow_fields.pvd", "flow_fields_fine_vtk.pvd")
            # if os.path.exists("flow_fields"):
            #     shutil.move("flow_fields", "flow_fields_fine_vtk")
            if os.path.exists("summary.yaml"):
                shutil.move("summary.yaml", "summary_fine.yaml")

        coarse_res = [0, 0, 0]

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
            print("=== COARSE PROBLEM ===")
            fractures.fractures = fractures_for_coarse_sample
            if config["sim_config"]["use_larger_domain"]:
                #print("hom domain box ", hom_domain_box)
                config["sim_config"]["geometry"]["domain_box"] = hom_domain_box
                config["sim_config"]["geometry"]["fractures_box"] = hom_domain_box
                #print("config geometry ", config["sim_config"]["geometry"])

            #print("domain box for homogenization ", config["sim_config"]["geometry"]["domain_box"])
            cond_tensors, pred_cond_tensors = DFMSim.homogenization(copy.deepcopy(config), fractures)
            #
            # pr.disable()
            # ps = pstats.Stats(pr).sort_stats('cumtime')
            # ps.print_stats(100)
            # exit()

            if config["sim_config"]["use_larger_domain"]:
                config["sim_config"]["geometry"]["domain_box"] = orig_domain_box
                config["sim_config"]["geometry"]["fractures_box"] = orig_frac_box

            if not gen_hom_samples:
                # print("os.getcwd() ", os.getcwd())
                # print("cond tensors ", cond_tensors)
                # print("pred cond tensors ", pred_cond_tensors)

                DFMSim._save_tensors(cond_tensors, file=DFMSim.COND_TN_FILE)
                DFMSim._save_tensors(pred_cond_tensors, file=DFMSim.PRED_COND_TN_FILE)

                #print("os.path.abspath(DFMSim.COND_TN_FILE) ", os.path.abspath(DFMSim.COND_TN_FILE))
                # print("config[sample_cond_tns] ", config["fine"]["sample_cond_tns"])

                file_path_split = os.path.split(os.path.split(os.path.abspath(DFMSim.COND_TN_FILE))[0])
                sample_id = file_path_split[1].split("_")[1]

                # shutil.copy(os.path.abspath(DFMSim.COND_TN_FILE), config["coarse"]["sample_cond_tns"])
                # os.rename(os.path.join(config["coarse"]["sample_cond_tns"], DFMSim.COND_TN_FILE),
                #           os.path.join(config["coarse"]["sample_cond_tns"], sample_id + "_" + DFMSim.COND_TN_FILE))

                # shutil.copy(os.path.abspath(DFMSim.PRED_COND_TN_FILE), config["coarse"]["sample_cond_tns"])
                # os.rename(os.path.join(config["coarse"]["sample_cond_tns"], DFMSim.PRED_COND_TN_FILE),
                #           os.path.join(config["coarse"]["sample_cond_tns"], sample_id + "_" + DFMSim.PRED_COND_TN_FILE))

                config["cond_tns_yaml_file"] = os.path.abspath(DFMSim.COND_TN_FILE)
                config["pred_cond_tns_yaml_file"] = os.path.abspath(DFMSim.PRED_COND_TN_FILE)

                ######################
                ### coarse problem ###
                ######################
                #coarse_ref = FlowProblem.make_microscale((config["fine"]["step"], config["coarse"]["step"]), fractures, fine_flow, config)
                # coarse_ref.pressure_loads = p_loads
                # coarse_ref.reg_to_group = fine_flow.reg_to_group
                # coarse_ref.regions = fine_flow.regions
                #print("coarse problem ")
                # print("fractures ", fractures)
                # exit()

                # if "nn_path" in config["sim_config"] and \
                #         ("run_only_hom" not in config["sim_config"] or not config["sim_config"]["run_only_hom"]):

                coarse_flow = FlowProblem.make_coarse((config["coarse"]["step"], config["sim_config"]["geometry"]["fr_max_size"]), fractures, config)
                coarse_flow.fr_range = [config["coarse"]["step"], coarse_flow.fr_range[1]]
                coarse_flow.make_mesh()
                coarse_flow.make_fields()
                # print("coarse flow ", coarse_flow)
                # print("config ", config)

                if os.path.exists("flow_fields"):
                    shutil.rmtree("flow_fields")

                if "flow_sim" in config["sim_config"] and config["sim_config"]["flow_sim"]:
                    coarse_res, status, conv_check = DFMSim._run_sample(coarse_flow, config)
                    if not conv_check:
                        raise Exception("coarse sample not converged")
                    coarse_res = [coarse_res[0], coarse_res[0], coarse_res[0]]
                    if os.path.exists("flow_fields.pvd"):
                        shutil.move("flow_fields.pvd", "flow_field_coarse.pvd")
                    if os.path.exists("flow_fields"):
                        shutil.move("flow_fields", "flow_fields_coarse")
                    if os.path.exists("flow_fields.msh"):
                        shutil.move("flow_fields.msh", "flow_fields_coarse.msh")
                else:
                    done = []
                    status, p_loads, outer_reg_names, conv_check = DFMSim._run_homogenization_sample(coarse_flow, config)
                    if not conv_check:
                        raise Exception("coarse sample not converged")
                    done.append(coarse_flow)
                    cond_tn, diff = coarse_flow.effective_tensor_from_bulk(p_loads, outer_reg_names, coarse_flow.basename)
                    #print("coarse cond tn ", cond_tn)
                    cond_tn = cond_tn[0]
                    cond_tn[0, 1] = (cond_tn[0, 1] + cond_tn[1, 0]) / 2
                    coarse_res = cond_tn.flatten()
                    coarse_res = [coarse_res[0], coarse_res[1], coarse_res[3]]
                print("coarse res ", coarse_res)

                # print("centers cond[0].shape ", np.array(coarse_flow._center_cond[0]).shape)
                # print("centers cond[1].shape ", np.array(coarse_flow._center_cond[1]).shape)
                #
                # centers_cond = np.array(coarse_flow._center_cond)

                #print("np.array(coarse_flow._center_cond[1])shape ", np.array(coarse_flow._center_cond[1]).shape)

                shutil.copy(os.path.abspath(DFMSim.COND_TN_FILE), config["coarse"]["sample_cond_tns"])
                np.save(os.path.join(config["coarse"]["sample_cond_tns"], sample_id + "_centers_" + DFMSim.COND_TN_FILE),
                        np.array(coarse_flow._center_cond[0]))

                shutil.copy(os.path.abspath(DFMSim.COND_TN_FILE), config["coarse"]["sample_cond_tns"])
                np.save(os.path.join(config["coarse"]["sample_cond_tns"],
                                     sample_id + "_cond_tns_" + DFMSim.COND_TN_FILE),
                        np.array(coarse_flow._center_cond[1])[:,[0, 1, 3, 4]])

                # #@TODO: RM ASAP
                # shutil.copy(os.path.abspath(DFMSim.COND_TN_FILE), config["coarse"]["sample_cond_tns"])
                # np.save(
                #     os.path.join(config["coarse"]["sample_cond_tns"], sample_id + "_centers_" + DFMSim.COND_TN_FILE),
                #     np.array(fine_flow._center_cond[0]))
                #
                # shutil.copy(os.path.abspath(DFMSim.COND_TN_FILE), config["coarse"]["sample_cond_tns"])
                # np.save(os.path.join(config["coarse"]["sample_cond_tns"],
                #                      sample_id + "_cond_tns_" + DFMSim.COND_TN_FILE),
                #         np.array(fine_flow._center_cond[1])[:, [0, 1, 3, 4]])

                if os.path.exists("flow_fields.pvd"):
                    os.remove("flow_fields.pvd")
                if os.path.exists("flow_fields"):
                    shutil.rmtree("flow_fields")

                # if "nn_path" in config["sim_config"] and \
                #         ("run_only_hom" not in config["sim_config"] or not config["sim_config"]["run_only_hom"]):
                #     status, p_loads, outer_reg_names, conv_check = DFMSim._run_homogenization_sample(coarse_flow, config, format="vtk")
                #
                #     DFMSim.make_summary(done)
                #     if os.path.exists("flow_fields.msh"):
                #         shutil.move("flow_fields.msh", "flow_fields_coarse.msh")
                #
                #     ff_coarse_vtk = os.path.join(os.getcwd(), "flow_field_coarse_vtk")
                #     if os.path.exists(ff_coarse_vtk):
                #         shutil.rmtree(ff_coarse_vtk)
                #     os.mkdir(ff_coarse_vtk)
                #
                #     if os.path.exists("flow_fields.pvd"):
                #         shutil.move("flow_fields.pvd", ff_coarse_vtk)
                #         #shutil.move("flow_fields.pvd", "flow_fields_coarse_vtk.pvd")
                #     if os.path.exists("flow_fields"):
                #         shutil.move("flow_fields", ff_coarse_vtk)
                if os.path.exists("summary.yaml"):
                    shutil.move("summary.yaml", "summary_coarse.yaml")


                ############################################
                ## Coarse sample - predicted cond tensors ##
                ############################################
                if len(pred_cond_tensors) > 0:
                    pred_coarse_dir = os.path.join(os.getcwd(), "pred_coarse")
                    os.mkdir(pred_coarse_dir)
                    os.chdir(pred_coarse_dir)
                    # print("os.getcwd() ", os.getcwd())
                    # exit()
                    config["back_up_cond_tns_yaml_file"] = config["cond_tns_yaml_file"]
                    config["cond_tns_yaml_file"] = config["pred_cond_tns_yaml_file"]
                    coarse_flow = FlowProblem.make_coarse(
                        (config["coarse"]["step"], config["sim_config"]["geometry"]["fr_max_size"]), fractures, config)
                    coarse_flow.fr_range = [config["coarse"]["step"], coarse_flow.fr_range[1]]
                    coarse_flow.make_mesh()
                    coarse_flow.make_fields()

                    if "flow_sim" in config["sim_config"] and config["sim_config"]["flow_sim"]:
                        coarse_res, status, conv_check = DFMSim._run_sample(coarse_flow, config)
                        if not conv_check:
                            raise Exception("pred coarse sample not converged")
                        coarse_res = [coarse_res[0], coarse_res[0], coarse_res[0]]
                        if os.path.exists("flow_fields.pvd"):
                            shutil.move("flow_fields.pvd", "flow_field_coarse_pred.pvd")
                        if os.path.exists("flow_fields"):
                            shutil.move("flow_fields", "flow_fields_coarse_pred")
                        if os.path.exists("flow_fields.msh"):
                            shutil.move("flow_fields.msh", "flow_fields_coarse_pred.msh")
                    else:
                        done = []
                        status, p_loads, outer_reg_names, conv_check = DFMSim._run_homogenization_sample(coarse_flow, config)
                        if not conv_check:
                            raise Exception("pred coarse sample not converged")
                        done.append(coarse_flow)
                        cond_tn, diff = coarse_flow.effective_tensor_from_bulk(p_loads, outer_reg_names, coarse_flow.basename)
                        cond_tn = cond_tn[0]
                        cond_tn[0, 1] = (cond_tn[0, 1] + cond_tn[1, 0]) / 2
                        coarse_res = cond_tn.flatten()
                        coarse_res = [coarse_res[0], coarse_res[1], coarse_res[3]]
                    print("coarse res ", coarse_res)

                    if os.path.exists("flow_fields.pvd"):
                        os.remove("flow_fields.pvd")
                    if os.path.exists("flow_fields"):
                        shutil.rmtree("flow_fields")

                    # if "nn_path" in config["sim_config"] and \
                    #         ("run_only_hom" not in config["sim_config"] or not config["sim_config"]["run_only_hom"]):
                    #
                    #     status, p_loads, outer_reg_names, conv_check = DFMSim._run_homogenization_sample(coarse_flow, config, format="vtk")
                    #
                    #     DFMSim.make_summary(done)
                    #
                    #     ff_coarse_vtk = os.path.join(os.getcwd(), "flow_field_coarse_vtk")
                    #     if os.path.exists(ff_coarse_vtk):
                    #         shutil.rmtree(ff_coarse_vtk)
                    #     os.mkdir(ff_coarse_vtk)
                    #
                    #     if os.path.exists("flow_fields.pvd"):
                    #         shutil.move("flow_fields.pvd", ff_coarse_vtk)
                    #         # shutil.move("flow_fields.pvd", "flow_fields_coarse_vtk.pvd")
                    #     if os.path.exists("flow_fields"):
                    #         shutil.move("flow_fields", ff_coarse_vtk)
                    if os.path.exists("summary.yaml"):
                        shutil.move("summary.yaml", "summary_coarse.yaml")

                    if os.path.exists("flow_fields.msh"):
                        shutil.move("flow_fields.msh", "flow_fields_coarse.msh")

        #print("fine res ", fine_res)
        return fine_res, coarse_res

    @staticmethod
    def _save_tensors(cond_tensors, file):
        #print("save tensors cond tensors ", cond_tensors)
        with open(file, "w") as f:
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
    def _run_homogenization_sample(flow_problem, config, format="gmsh"):
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
        start_time = time.time()
        outer_reg_names = []
        for reg in flow_problem.side_regions:
            outer_reg_names.append(reg.name)
            outer_reg_names.append(reg.sub_reg.name)

        base = flow_problem.basename
        base = base + "_hom"
        if format == "vtk":
            base += "_vtk"
        outer_regions_list = outer_reg_names
        # flow_args = config["flow123d"]
        n_steps = config["sim_config"]["n_pressure_loads"]
        t = np.pi * np.arange(0, n_steps) / n_steps
        p_loads = np.array([np.cos(t), np.sin(t)]).T

        in_f = in_file(base)
        out_dir =base
        # n_loads = len(self.p_loads)
        # flow_in = "flow_{}.yaml".format(self.base)
        params = dict(
            mesh_file=mesh_file(flow_problem.basename),
            fields_file=fields_file(flow_problem.basename),
            outer_regions=str(outer_regions_list),
            n_steps=len(p_loads))

        out_dir = os.getcwd()

        common_files_dir = config["fine"]["common_files_dir"]

        if format == "vtk":
            substitute_placeholders(os.path.join(common_files_dir, DFMSim.YAML_TEMPLATE_H_VTK), in_f, params)
            #print("os.path.join(common_files_dir, DFMSim.YAML_TEMPLATE_H_VTK) ", os.path.join(common_files_dir, DFMSim.YAML_TEMPLATE_H_VTK))
        else:
            substitute_placeholders(os.path.join(common_files_dir, DFMSim.YAML_TEMPLATE_H), in_f, params)

        flow_args = ["docker", "run", "-v", "{}:{}".format(os.getcwd(), os.getcwd()), *config["flow123d"]]
        #flow_args = ["singularity", "exec", "/storage/liberec3-tul/home/martin_spetlik/flow_3_1_0.sif", "flow123d"]


        flow_args.extend(['--output_dir', out_dir, os.path.join(out_dir, in_f)])

        # print("out dir ", out_dir)
        #print("flow_args ", flow_args)

        # if os.path.exists(os.path.join(out_dir, "flow123.0.log")):
        #     os.remove(os.path.join(out_dir, "flow123.0.log"))

        # if os.path.exists(os.path.join(out_dir, DFMSim.FIELDS_FILE)):
        #     return True
        with open(base + "_stdout", "w") as stdout:
            with open(base + "_stderr", "w") as stderr:
                completed = subprocess.run(flow_args, stdout=stdout, stderr=stderr)
            print("Exit status: ", completed.returncode)
            status = completed.returncode == 0
        print("run homogenization sample TIME: {}".format(time.time() - start_time))
        conv_check = DFMSim.check_conv_reasons(os.path.join(out_dir, "flow123.0.log"))
        print("converged: ", conv_check)

        return status, p_loads, outer_reg_names, conv_check  # and conv_check
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
        #print("yaml file ", os.path.join(common_files_dir, DFMSim.YAML_TEMPLATE))
        substitute_placeholders(os.path.join(common_files_dir, DFMSim.YAML_TEMPLATE), in_f, params)
        flow_args = ["docker", "run", "-v", "{}:{}".format(os.getcwd(), os.getcwd()), *config["flow123d"]]
        #flow_args = ["singularity", "exec", "/storage/liberec3-tul/home/martin_spetlik/flow_3_1_0.sif", "flow123d"]

        flow_args.extend(['--output_dir', out_dir, os.path.join(out_dir, in_f)])

        if os.path.exists(os.path.join(out_dir, DFMSim.FIELDS_FILE)):
            print("os.path.join(out_dir, DFMSim.FIELDS_FILE) ", os.path.join(out_dir, DFMSim.FIELDS_FILE))
            return True
        with open(base + "_stdout", "w") as stdout:
            with open(base + "_stderr", "w") as stderr:
                completed = subprocess.run(flow_args, stdout=stdout, stderr=stderr)
            print("Exit status: ", completed.returncode)
            status = completed.returncode == 0
        conv_check = DFMSim.check_conv_reasons(os.path.join(out_dir, "flow123.0.log"))
        print("converged: ", conv_check)

        return DFMSim._extract_result(os.getcwd()), status, conv_check # and conv_check
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
                #if flux_in > 1e-10:
                #    raise Exception("Possitive inflow at outlet region.")
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
        #spec1 = QuantitySpec(name="cond_tn", unit="m", shape=(1, 1), times=[1], locations=['0'])
        spec1 = QuantitySpec(name="cond_tn", unit="m", shape=(3, 1), times=[1], locations=['0'])

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
            print("intensity ", intensity)

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

                print("fr size range ", fr_size_range)

                pop.set_sample_range(fr_size_range,
                                     sample_size=n_frac_limit)
            elif pow_law_sample_range:
                pop.set_sample_range(pow_law_sample_range)

        #print("total mean size: ", pop.mean_size())
        #print("size range:", pop.families[0].size.sample_range)


        pos_gen = fracture.UniformBoxPosition(fracture_box)
        fractures = pop.sample(pos_distr=pos_gen, keep_nonempty=False)

        print("fractures len ", len(fractures))

        #print("fractures ", fractures)
        #print("fr_size_range[0] ", fr_size_range[0])

        fr_set = fracture.Fractures(fractures, fr_size_range[0] / 2)

        return fr_set

    def make_summary(done_list):
        results = {problem.basename: problem.summary() for problem in done_list}

        #print("results ", results)

        with open("summary.yaml", "w") as f:
            yaml.dump(results, f)



